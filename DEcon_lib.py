import copy
import math

import numpy as np
import torch

"""Python library for PyTorch-based image / optics calculation

a collection of convenient Python functions commonly used in optical
calculations in PyTorch (for its easy and convenient interface to GPU).
All of the functions in this module is compatible with tensor.cuda()
input array/tensors.

Contains several classes used for convenient handling of deconvolution
problems:
ComplexOps
    A static container class for doing complex arithmetics. Currently
    supported are : exp(z), sqrt(z), abs(z), abs(z)^2, mult(z1,z2)
    , jmult(x), conj(z)

TODO:
    Remove inconsistent usage of NumPy arrays as input in Complex operations.
    Write another interface to handle NumPy <--> PyTorch usage

    Vectorial implementation for the PupilFunction for computing PSF is not yet
    completed

History:
        18 September 2018   D.E.    First version
        21 September 2018   D.E.    Added few functions:
                                    - zernikePolynomials
        31 May 2022         D.E     pytorch FFT api has been updated
                                    now consistent with numpy. Updating all
                                    convolution functions
"""

# constants
PI = torch.pi
SMALL = torch.tensor(torch.finfo(torch.float32).eps)


if torch.cuda.is_available():
    SMALL_GPU = SMALL.cuda()


fftfreq = torch.fft.fftfreq
rfftfreq = torch.fft.rfftfreq
# meshgrid = torch.meshgrid


def meshgrid(*args):
    return torch.meshgrid(*args, indexing="ij")


# function definitions
def fftshift_1d(x):
    """Shifts samples so that origin is at the 0-index.

    Args:
        x (torch.tensor): input tensor. Must be 1-D

    Returns:
        shifted_x (torch.tensor): origin-centered at x[0]
        shifted_id (torch.tensor): the index for shifting the origin. Use with Tensor.index_select()

    """

    Nx = len(x)
    # determine whether sample is even or odd
    offset = (1, 0)[Nx % 2 == 0]
    Nhalf = (int((Nx - 1) / 2), int(Nx / 2))[Nx % 2 == 0]

    shifted_x = torch.cat([x[Nhalf + offset :], x[0 : Nhalf + offset]])
    # also generate shifted indexing
    x_id = torch.arange(Nx, device=x.device)
    shifted_id = torch.cat([x_id[Nhalf + offset :], x_id[0 : Nhalf + offset]])
    return shifted_x, shifted_id


def fourier_meshgrid(*N, d=1.0, real=False):
    """Computes meshgrid of frequency spacing

    Computes Fourier coordinate matrices given vectors of sizes N1, N2, N3, ...
    and its corresponding real-space increment d1,d2,d3, ...

    Parameters
    ----------
    N1,N2,N3,... : Python int
        Size of sample N for computing fourier coordinate grids
    d1,d2,d3,... : Python float, optional
        Real-space spacing corresponding to each sample size N1,N2,N3... Otherwise,
        the default is each dimension has a spacing of 1.0
    real(bool): if False, then it will return the Fourier coordinates for fully
        complex samples. If true, it will return the Fourier coordinates with the
        last axis halved.

    """
    Ndim = len(N)
    dimrange = range(Ndim)

    # if spacing is not supplied, use 1.0 for all dimensions
    if d == 1.0:
        dvec = [1.0 for n in dimrange]
        Nd = len(dvec)
    else:
        dvec = d
        Nd = len(dvec)

    assert (
        Ndim == Nd
    ), "Number of spacing d ({:d}) must be equal to number of \
    sample sizes ({:d})".format(
        Nd, Ndim
    )

    if real:
        spacings = [
            fftfreq(N[n], d=dvec[n]) if n < (Ndim - 1) else rfftfreq(N[n], d=dvec[n])
            for n in dimrange
        ]
    else:
        spacings = [fftfreq(N[n], d=dvec[n]) for n in dimrange]

    return meshgrid(*spacings)


def imshift(img, *shifts):
    """Translate n-dimensional image given shifts by Fourier shift theorem

    Translate an image specified by shifts using Fourier-shift theorem.
    Can take non-integer shifts.

    Args:
        img(PyTorch torch.FloatTensor):
            Input image to be shifted. Can be 2D/3D real image
        shifts1, shifts2, [shifts3](Python float):
            Shift amount, in pixel units, for the corresponding axes as they appear
            in order of tensor.shape. Must provide all shifts for each axes. To keep
            image fixed along one axis, pass 0 as the shift value. Axis order is
            "row-major", conventionally, this is Z, Y, X or according to 'ij' or
            matrix indexing. Last dimension is usually reserved for imaginary part
            of a torch.Tensor

    Returns:
        shifted torch.FloatTensor

    """
    Ndim = img.dim()
    inputshape = img.shape
    Nshifts = len(shifts)
    dimrange = range(Ndim)
    assert (
        Ndim == Nshifts
    ), "Number of shifts ({:d}) must be equal to \
    equal to dimensionality ({:d})".format(
        Nshifts, Ndim
    )

    # compute fourier grids
    img_FT = torch.fft.rfftn(img)
    k_space = fourier_meshgrid(*img.shape, real=True)

    k_shift = torch.zeros(k_space[0].shape, device=img.device)

    # compute shift for each axes/dimension
    for n in dimrange:
        k_shift += k_space[n].to(img.device) * shifts[n]

    # compute phase shifting term, the convention here is positive shifts are
    # applied to increment pixel position along each corresponding axes
    phase_shift = torch.exp(-2.0 * PI * 1j * k_shift)

    return torch.fft.irfftn(img_FT * phase_shift, s=inputshape)


def pad_zero_origin(img, *outputdims):
    Ndim = len(img.shape)
    assert Ndim < 4, "Only up to 4D images are supported"
    assert len(img.shape) == len(outputdims), (
        "output dims must have %d \
    integers"
        % (len(img.shape))
    )

    # zero-pad origin-centered <Tensor> using Fourier shift theorem
    halfdims = [-(int((N - 1) / 2), int(N / 2))[N % 2 == 0] for N in img.shape]
    shifts = [(halfdims[n], 0)[img.shape[n] == outputdims[n]] for n in range(Ndim)]
    paddedimg = torch.zeros(*outputdims, device=img.device)
    inputshape = img.shape
    if Ndim == 2:
        paddedimg[0 : inputshape[0], 0 : inputshape[1]] = img
    elif Ndim == 3:
        paddedimg[0 : inputshape[0], 0 : inputshape[1], 0 : inputshape[2]] = img
    elif Ndim == 4:
        paddedimg[
            0 : inputshape[0], 0 : inputshape[1], 0 : inputshape[2], 0 : inputshape[3]
        ] = img

    return imshift(paddedimg, *shifts)


def pad_centered_3D_array(img, Pz, Py, Px):
    # Pz, Py, Pz is the output (padded) array size
    out = torch.zeros(Pz, Py, Px, device=img.device)
    Nz, Ny, Nx = img.shape
    evenZ = Nz % 2 == 0
    evenY = Ny % 2 == 0
    evenX = Nx % 2 == 0
    halfZ = (int((Nz - 1) / 2), int(Nz / 2))[evenZ]
    halfY = (int((Ny - 1) / 2), int(Ny / 2))[evenY]
    halfX = (int((Nx - 1) / 2), int(Nx / 2))[evenX]
    endZ = (Pz - halfZ) - (1, 0)[evenZ]
    endY = (Py - halfY) - (1, 0)[evenY]
    endX = (Px - halfX) - (1, 0)[evenX]
    # fill in zero-array with original image
    out[0:halfZ, 0:halfY, 0:halfX] = img[0:halfZ, 0:halfY, 0:halfX]
    out[0:halfZ, 0:halfY, endX:] = img[0:halfZ, 0:halfY, halfX:]
    out[0:halfZ, endY:, 0:halfX] = img[0:halfZ, halfY:, 0:halfX]
    out[0:halfZ, endY:, endX:] = img[0:halfZ, halfY:, halfX:]
    out[halfZ:, 0:halfY, 0:halfX] = img[halfZ:, 0:halfY, 0:halfX]
    out[halfZ:, 0:halfY, endX:] = img[halfZ:, 0:halfY, halfX:]
    out[halfZ:, endY:, 0:halfX] = img[halfZ:, halfY:, 0:halfX]
    out[halfZ:, endY:, endX:] = img[halfZ:, halfY:, halfX:]

    return out


def resize_pupil(PupilFunction, Ny_out, Nx_out):
    """Resizes PupilFunction to desired Ny,Nx

    Args:
        PupilFunction(PupilFunction): Input pupil function
        Ny_out(int): output pupil image height
        Nx_out(int): output pupil image width

    Returns:
        resized PupilFunction

    """
    pupil_data = PupilFunction.pupil
    out = torch.zeros(Ny_out, Nx_out, 2, device=pupil_data.device)
    Ny, Nx, _ = pupil_data.shape  # last dimension should be 2 for real, imaginary
    evenY = Ny % 2 == 0
    evenX = Nx % 2 == 0
    # determine half-sizes for pupil array
    halfY = (int((Ny - 1) / 2), int(Ny / 2))[evenY]
    halfX = (int((Nx - 1) / 2), int(Nx / 2))[evenX]
    endY = (Ny_out - halfY) - (1, 0)[evenY]
    endX = (Nx_out - halfX) - (1, 0)[evenX]
    # compute amplitude PSF
    psfa = torch.fft.ifft2(pupil_data)
    # fill in zero-array with PSFA
    out[0:halfY, 0:halfX, :] = psfa[0:halfY, 0:halfX, :]
    out[0:halfX, endX:, :] = psfa[0:halfY, halfX:, :]
    out[endY:, 0:halfX, :] = psfa[halfY:, 0:halfX, :]
    out[endY:, endX:, :] = psfa[halfY:, halfX:, :]
    # transform back to pupil and assign to copied pupil function object
    PupilFunction_out = copy.deepcopy(PupilFunction)
    PupilFunction_out.Ny = Ny_out
    PupilFunction_out.Nx = Nx_out
    # compute pupil quantities, we need to apply mask to resized pupil
    PupilFunction_out._compute_pupil_quantities()
    # apply mask to resized pupil
    PupilFunction_out.pupil = torch.fft(out, 2) * PupilFunction_out.mask.repeat(
        2, 1, 1
    ).permute(1, 2, 0)

    return PupilFunction_out


def zernikePolynomials(rho, phi, nmax):
    """Computes Zernike polynomials given rho, phi, and order

    Function to compute Zernike polynomials (2D circular polynomial). The order
    numbering uses the ANSI convention for mapping the single index j to m,n.

    Parameters
    ==========
    rho : torch.tensor (2D)
        Radial coordinate of the pupil function. Must be between 0 and 1.
        PupilFunction.mask * PupilFunction.kxy / (PupilFunction.NA/PupilFunction.wvlen)
    phi : torch.tensor (2D)
        Angular coordinate of the pupil function. Can be compute from arctan2(ky/kx)
        Can pass PupilFunction.phi
    mask : torch.tensor (2D)
        Pupil support function with 1 within support and 0 elsewhere.
        Can pass PupilFunction.mask
    nmax : int
        Maximum number of polynomial order to be computed, minus one

    Returns
    =======
    Zmn : torch.tensor (3D)
        Returns circular Zernike polynomial computed for given rho, phi and nmax

    """

    def _Rmn(m, n, rho):
        # internal function, compute R_m^n term
        Rmn = torch.zeros_like(rho)
        Nk = int((n - m) / 2)
        for k in range(Nk + 1):
            _num = (-1.0) ** k * math.factorial(n - k)
            try:
                _den = (
                    math.factorial(k)
                    * math.factorial((n + m) / 2 - k)
                    * math.factorial((n - m) / 2 - k)
                )
            except ValueError:
                print("m={:d}, n={:d}, k={:d}".format(m, n, k))
                raise
            Rmn += (float(_num) / float(_den)) * rho ** (float(n - 2 * k))
        return Rmn

    def _recursive_sum(n):
        if n == 0:
            return 0
        else:
            return n + _recursive_sum(n - 1)

    Norders = _recursive_sum(nmax + 1)
    Zmn = torch.zeros((Norders,) + rho.shape, dtype=torch.float32)
    n_range = torch.arange(nmax, -1, -1)

    for n in n_range:
        m_range = torch.arange(-n, n + 2, 2)
        for m in m_range:
            normfact = (float(2 * (n + 1)) / (1.0 + float(m == 0))) ** 0.5
            # ANSI notation
            j = int((n * (n + 2) + m) / 2)
            if m >= 0:
                # even function
                Zmn[j, :, :] = normfact * _Rmn(m, n, rho) * torch.cos(float(m) * phi)
            elif m < 0:
                # odd function
                Zmn[j, :, :] = -normfact * _Rmn(-m, n, rho) * torch.sin(-float(m) * phi)
    return Zmn


def compute_dwt_filters(*N, J=3):
    """Compute Fourier-space a trous wavelet filters

    This function uses NumPy for computing an N-dimensional discrete
    wavelet filters (a trous - undecimated wavelet transform) in the Fourier
    space with a B3 (cubic-spline). It then uses PyTorch to convert them into
    a complex tensor

    Args:
        N1,N2,...,Nn(ints): length of samples.
        J(int): number of levels in wavelet decomposition. Default is 4. The
            output coefficients will be J+1.

    Returns:
        J+1,N1,N2,...,Nn,2(complex torch.tensor)

    """

    ndim = len(N)
    j2PI = 1j * 2.0 * np.pi

    # For real signals only the last axis is halved
    f = [
        np.fft.fftfreq(n) if i < (ndim - 1) else np.fft.rfftfreq(n)
        for i, n in enumerate(N)
    ]

    # generate the fourier meshgrid, k = k_1 + k_2 + k_3 + ...
    # we can sum them all because the zero-insertion is isotropic in all axes
    if ndim > 1:
        k = np.meshgrid(*f, indexing="ij")
        filter_shape = k[0].shape
    elif ndim == 1:
        k = f[0]
        filter_shape = [len(k)]

    # pre-allocate dilated filters
    psi = np.zeros((J,) + filter_shape, dtype=np.complex64)

    for j in range(J):
        _psi = np.ones(filter_shape, dtype=np.complex64)
        for _k in k:
            _psi *= (
                0.375
                + 0.25 * (np.exp(-j2PI * -(2**j) * _k) + np.exp(-j2PI * 2**j * _k))
                + 0.0625
                * (
                    np.exp(-j2PI * -(2 ** (j + 1)) * _k)
                    + np.exp(-j2PI * 2 ** (j + 1) * _k)
                )
            )
        psi[j, ...] = _psi

    # form wavelet filter; pre-allocate fourier filter arrays
    filters = np.zeros((J + 1,) + filter_shape, dtype=np.complex64)

    for j in range(J):
        if j == 0:
            filters[j, ...] = 1.0 - psi[j, ...]
        else:
            filters[j, ...] = np.prod(psi[0:j, ...], axis=0) - np.prod(
                psi[0 : j + 1, ...], axis=0
            )
    filters[-1, ...] = np.prod(psi, axis=0)

    # convert to pytorch tensor
    return torch.from_numpy(filters)


class PupilFunction:
    """Pupil function for 3D optical microscopy

    This is an implementation of the pupil function as described in
    Hanser et al. 2003. The current implementation does not include polarization
    effects. It is meant to be used as part of a deconvolution application
    on the GPU using FFT routines in PyTorch (torch.rfft, etc.)

    Reference:
    Hanser, Bridget M., et al. "Phase‐retrieved pupil functions in wide‐field fluorescence microscopy." Journal of microscopy 216.1 (2004): 32-48.

    """

    def __init__(
        self,
        Nx=256,
        Ny=256,
        dx=0.085,
        dy=0.085,
        wavelength=0.525,
        ns=1.334,
        ni=1.515,
        NA=1.40,
        vectorial=False,
        apodize=False,
    ):
        """

        All physical dimensions are given in micron. The pupil function is
        initialized by these parameters

        Args:

            Nx(int): image width or number of pixels along columns
            Ny(int): image height or number of pixels along rows
            dx(float): pixel size in x
            dy(float): pixel size in y
            wavelength(float): emission wavelength in micron
            ns(float): refractive index of the sample medium. Default is 1.334
            ni(float): refractive index of the immersion medium. This depends on the
                objective lens (air, glycerin, water, or oil). Default is oil, which
                is 1.515.
            vectorial(bool): account for vectorial effects. Currently this is only
                implemented as far as computing how each polarization vector in xyz
                contributes at the pupil plane in xy. It is currently incomplete.
                Default is False.
            apodize(bool): account for the apodization of the pupil plane due to
                sine condition. This does not impact the PSF/OTF calculation
                significantly. Default is False.

        Note:
            For convenience, this class has a __str__ method that will allow printing
            relevant optical parameters.

            ..

                >>> pf = PupilFunction()
                >>> print(pf)
                Nx = 256, Ny = 256
                NA = 1.40, wavelength = 0.525
                dx = 0.085, dy = 0.085
                ni = 1.515, ns = 1.334

        """
        self.dx = dx
        self.dy = dy
        self.Nx = Nx
        self.Ny = Ny
        self.wvlen = wavelength
        self.dkx = 1.0 / dx
        self.dky = 1.0 / dy
        self.ni = ni
        self.ns = ns
        self.NA = NA
        # flag to indicate whether experimental pupil function is used
        # implement self.loadPupil("pupil_function_file.mrc")
        # followed by resizing of pupil, etc.
        self.pupil = None
        # flag to indicate whether pupil quantities have been computed
        self.ready = False
        # initialize pupil function quantities
        self._compute_pupil_quantities(vectorial=vectorial, apodize=apodize)

    def _compute_pupil_quantities(self, vectorial=False, apodize=False):
        """Computes required quantities for computing PSF and OTF

        This is an internal function. Users should not need to call this.

        The quantities are

            - kxy, fourier radial coordinate
            - kz, fourier-spacing in z
            - mask, support of pupil function 1.0 inside, 0 outside
            - theta_1, emission angle in immersion medium
            - theta_2, emission angle in sample medium
            - A, amplitude transmission/wave compression factor
            - apodization, 1/sqrt(cos(theta_1)) apodization factor

        If vectorial is True, it computes 6 polarization-dependent factor
        for a point emitter in x,y,z emitting electric field to x,y axes
        (e.g. Px -> Ex & Px -> Ey, etc.)

        """

        # prepare radial and angular pupil coordinates
        self.ky, self.kx = fourier_meshgrid(self.Ny, self.Nx, d=(self.dy, self.dx))
        # radial coordinate
        self.kxy = torch.sqrt(self.kx * self.kx + self.ky * self.ky)
        # azimuth angle
        self.phi = torch.atan2(self.ky, self.kx)
        # radial coordinate for z
        zarg = (self.ni / self.wvlen) ** 2 - self.kxy * self.kxy
        self.kz = torch.sqrt(zarg)
        self.kz[zarg < 0] = 0.0

        # mask on diffraction limit of circular aperture
        pupil_limit = self.NA / self.wvlen

        self.mask = self.kxy.clone().detach() <= pupil_limit
        self.mask = self.mask.float()

        # ideal pupil with perfect (zero) phase
        self.pupil0 = torch.zeros(self.mask.shape, dtype=torch.complex64)
        self.pupil0.real = self.mask.clone()

        # compute angular coordinate
        a = self.ni / self.ns  # ratio of refractive indices
        sin_theta_1 = (self.wvlen / self.ni) * self.kxy
        self.theta_1 = torch.asin(sin_theta_1)
        self.theta_2 = torch.asin(a * sin_theta_1)

        # -1 <= x <= 1 for arcsin(x)
        outside_domain_1 = torch.abs(sin_theta_1) > 1
        outside_domain_2 = torch.abs(a * sin_theta_1) > 1

        self.theta_1[outside_domain_1] = 0.0
        self.theta_2[outside_domain_2] = 0.0

        # compute At
        # sin(theta_1) * cos(theta_2)
        sinT1mcosT2 = sin_theta_1 * torch.cos(self.theta_2)
        # sin(theta_1 + theta_2)
        sinT1pT2 = torch.sin(self.theta_1 + self.theta_2)
        # cos(theta_2 - theta_1)
        cosT2mT1 = torch.cos(self.theta_2 - self.theta_1)
        # Avoid dividing by zero
        arg1 = sinT1mcosT2 / torch.maximum(sinT1pT2, SMALL)
        arg2 = 1.0 + 1.0 / torch.maximum(cosT2mT1, SMALL)
        At = arg1 * arg2

        # compute Aw
        Aw = (self.ni * torch.tan(self.theta_2)) / torch.maximum(
            self.ns * torch.tan(self.theta_1), SMALL
        )

        # compute apodization factor, sine condition
        # do pupil *= self.apodization (phase-retrieved pupil doesnt need it)
        self.apodization = 1.0 / torch.sqrt(torch.cos(self.theta_1))

        if apodize:
            self.pupil0[..., 0] *= self.apodization

        # Amplitude factor
        self.A = At * Aw

        # Currently not being used ... #
        # For each dipole orientation Px, Py, Pz
        if vectorial:
            # compute vectorial factors, to account for polarization
            cos_theta = torch.cos(self.theta_1)
            cos_phi = torch.cos(self.phi)
            sin_phi = torch.sin(self.phi)
            Pvec = torch.zeros((6,) + self.kxy.shape)
            # Px -> Ex
            Pvec[0, :, :] = cos_theta * cos_phi * cos_phi + sin_phi * sin_phi
            # Px -> Ey
            Pvec[1, :, :] = (cos_theta - 1.0) * sin_phi * cos_phi
            # Py -> Ex
            Pvec[2, :, :] = (cos_theta - 1.0) * sin_phi * cos_phi
            # Py -> Ey
            Pvec[3, :, :] = cos_theta * sin_phi * sin_phi + cos_phi * cos_phi
            # Pz -> Ex
            Pvec[4, :, :] = sin_theta_1 * cos_phi
            # Pz -> Ey
            Pvec[5, :, :] = sin_theta_1 * sin_phi
            # this results in a 6-component PSF that needs to be summed
            # for phase retrieval, multiply each pupil function with each one

    def retrieve_phase(
        self,
        observed_magnitudes,
        z_planes,
        max_iter=500,
        beta=0.95,
        method="ER",
        center_xy=True,
    ):
        """Retrieve the pupil function given measured magnitude data

        The magnitude is obtained by taking the square-root of the collected
        intensity. There are currently two methods that are implemented: the
        classic "Error-Reduction" (ER) and "Hybrid Input-Output" (HIO) algorithms

        Args:
            observed_magnitudes(3D torch.tensor): the z-dimension should be on
                the first axis. Each z-slice must correspond to the given physical
                position specified through the z_planes parameter.
            z_planes(torch.tensor): a 1-D vector of physical positions in z. The
                plane in-focus is at 0. Under-focussed is negative and over-focussed
                z-planes are positive.
            max_iter(int): the maximum number of iterations for phase retrieval
            beta(float): relaxation parameter for the "HIO" algorithm. Should be
                 number less than 1.0. Typically this is kept at >=0.9.
            method(str): "ER" or "HIO".
            center_xy(bool): specify whether the PSF magnitude data has its
                center in the middle of the image. If False, then the input
                magnitude data should have its origin (brightest PSF peak) at
                position 0,0. If True, then its position should be at Nx/2,Ny/2.


        """
        # for method two options are implemented, "ER" and "HIO"
        # "ER" is the classic error reduction and "HIO" is Fienup's Hybrid-
        # input-output algorithm with parameter beta.
        # the observed magnitude, sqrt(PSF_intensity) must have equal number
        # of Nz with the given z_planes

        Nz = len(z_planes)

        assert observed_magnitudes.shape[0] == Nz, "Number of data must match z_planes"

        # compute defocus term
        defocus_term = (
            1j * 2.0 * PI * z_planes.reshape(Nz, 1, 1) * self.kz.repeat(Nz, 1, 1)
        )

        sum_intensity = torch.sum(observed_magnitudes * observed_magnitudes)

        if center_xy:
            sx = (self.Nx * self.dx) / 2.0
            sy = (self.Ny * self.dy) / 2.0
            shift_term = 1j * -2.0 * PI * (sx * self.kx + sy * self.ky)
            # add centering term into complex defocus-term array
            defocus_term += shift_term.repeat(Nz, 1, 1)

        pupil = self.pupil0.clone()

        # initialize random phase in pupil?
        init_phase = (
            (torch.rand(self.mask.shape, device=self.pupil0.device) - 0.5) * PI / 2.0
        )

        pupil = pupil * torch.exp(1j * init_phase)

        _defocus = torch.exp(defocus_term)
        _focus = torch.exp(-defocus_term)

        outside_support = (
            torch.tensor(
                1.0 * ~(self.mask == 1.0), dtype=torch.float32, device=self.mask.device
            )
            .repeat(2, 1, 1)
            .permute(1, 2, 0)
        )
        within_support = self.mask.repeat(2, 1, 1).permute(1, 2, 0)

        it = 0
        iMSE_list = []
        iviolation_list = []

        while it < max_iter:
            it += 1
            # input pupil, g
            g = pupil.repeat(Nz, 1, 1) * _defocus
            # compute amplitude PSF
            psfa = torch.fft.ifft2(g)
            # compute intensity Error
            error = torch.abs(torch.abs(psfa) - observed_magnitudes)
            error2 = torch.sum(error * error)
            iMSE = error2 / sum_intensity
            iMSE_list.append(iMSE)

            # swap amplitude
            # doing this with computing the angle with arctan is awkward and
            # not efficient. Do projection onto a unit vector instead
            _unit = psfa / torch.abs(psfa).repeat(2, 1, 1, 1).permute(1, 2, 3, 0)
            wrk = observed_magnitudes.repeat(2, 1, 1, 1).permute(1, 2, 3, 0) * _unit
            # transform back into pupil
            gprime = torch.fft.fft2(wrk)
            # refocus pupil
            gprime = gprime * _focus
            # average pupil function
            gprime = gprime.mean(dim=0)

            # compute object domain error
            violation = gprime * outside_support
            # compute the intensity of violation, abs(violation)^2
            iviolation = torch.sum(torch.abs(violation) ** 2) / torch.sum(
                torch.abs(gprime) ** 2
            )
            iviolation_list.append(iviolation)

            print(
                "\rIteration {:d}, iMSE = {:12.5e}, E_support = {:12.5e}".format(
                    it, iMSE, iviolation
                ),
                end="",
            )

            # apply object modification
            if method == "ER":
                pupil = gprime * within_support
            elif method == "HIO":
                feedback = pupil - beta * gprime
                pupil = gprime * within_support + feedback * outside_support

        print("\n")
        # apply support before returning
        pupil = pupil * within_support
        return pupil, iMSE_list, iviolation_list

    def compute_3D_psf(
        self,
        focal_planes,
        emitter_pos=0.0,
        center_xy=False,
        confocal=False,
        use_pupil=False,
    ):
        """Returns the 3D intensity PSF

        Args:
            focal_planes(1D torch.tensor): physical positions of the focal planes
            emitter_pos(float): distance of the point emitter from the coverslip.
                Negative number points in the direction of the excitation light
                (or into the sample). Default is 0.0, assuming PSF at the coverslip.
            center_xy(bool): If True, the output PSF will be centered in the
                middle of the image. Otherwise, center of the PSF is placed at 0,0.
                Default is False.
            confocal(bool): If True, then a simulated confocal PSF is returned
                by taking the square of the intensity PSF (simulating light
                going through two apertures of the same size). Otherwise,
                the widefield PSF is returned. Default is False.
            use_pupil(bool): If True, will use phase-retrieved pupil function
                to compute the 3D PSF.

        Returns:
            the 3D PSF (torch.tensor)

        Example:

            >>> pf = PupilFunction()
            >>> Nz = 50
            >>> dz = 0.2
            >>> z_sections = dz * fftfreq(Nz) * float(Nz)
            >>> psf3d = pf.compute_3D_psf(z_sections)

        """
        Nz = len(focal_planes)

        if self.pupil is None:
            # use naive pupil
            pupil = self.pupil0
        else:
            if use_pupil:
                # use phase-retrieved or whatever pupil was assigned by user
                pupil = self.pupil
            else:
                pupil = self.pupil0

        # compute defocus term
        defocus_term = (
            1j * 2.0 * PI * focal_planes.reshape(Nz, 1, 1) * self.kz.repeat(Nz, 1, 1)
        )

        if center_xy:
            sx = (self.Nx * self.dx) / 2.0
            sy = (self.Ny * self.dy) / 2.0
            shift_term = 1j * -2.0 * PI * (sx * self.kx + sy * self.ky)
            # add centering term into complex defocus-term array
            defocus_term += shift_term.repeat(Nz, 1, 1)

        # compute z-dependent RI mismatch term, OP(kx,ky,ni,ns,d)
        OPd = emitter_pos * (
            self.ns * torch.cos(self.theta_2) - self.ni * torch.cos(self.theta_1)
        )

        iOPd = 1j * 2.0 * PI * OPd / self.wvlen

        # compute pupil modifier
        psi = self.A * torch.exp(iOPd)

        # broadcast and modify pupil
        pupil = pupil.repeat(Nz, 1, 1) * psi

        # defocus pupil, no need to broadcast
        pupil = pupil * torch.exp(defocus_term)
        psfa = torch.fft.ifft2(pupil)

        # take the squared-absolute value to get intensity
        psfi = torch.abs(psfa) ** 2

        if confocal:
            psfi = psfi * psfi
        # return the normalized PSF
        return psfi / torch.sum(psfi)

    def compute_OTF(self, focal_planes, emitter_pos=0.0, use_pupil=False):
        """Returns the 3D OTF

        This computes the same quantity as compute_3D_psf() but returns the OTF
        instead by taking the auto-correlation of the pupil function. Which is
        equivalent to taking the Fourier Transform of the PSF.

        Args:
            focal_planes(1D torch.tensor): physical positions of the focal planes
            emitter_pos(float): distance of the point emitter from the coverslip.
                Negative number points in the direction of the excitation light
                (or into the sample). Default is 0.0, assuming PSF at the coverslip.
            use_pupil(bool): If True, will use phase-retrieved pupil function
                to compute the 3D PSF.

        Returns:
            the 3D OTF (torch.tensor)

        """
        Nz = len(focal_planes)

        if use_pupil:
            if self.pupil is None:
                # print("Experimental pupil is not present. Using ideal pupil")
                pupil = self.pupil0
            else:
                pupil = self.pupil
        elif not use_pupil:
            pupil = self.pupil0

        # compute defocus term
        defocus_term = (
            1j * 2.0 * PI * focal_planes.reshape(Nz, 1, 1) * self.kz.repeat(Nz, 1, 1)
        )

        # compute z-dependent RI mismatch term, OP(kx,ky,ni,ns,d)
        OPd = emitter_pos * (
            self.ns * torch.cos(self.theta_2) - self.ni * torch.cos(self.theta_1)
        )

        iOPd = 1j * 2.0 * PI * OPd / self.wvlen
        # compute pupil modifier
        psi = self.A.repeat(Nz, 1, 1), torch.exp(iOPd)
        # modify pupil and broadcast
        pupil = pupil.repeat(Nz, 1, 1) * psi
        # defocus pupil, no need to broadcast
        pupil = pupil * torch.exp(defocus_term)
        # compute OTF by autocorrelation
        pFT = torch.fft.fft2(pupil)
        OTF = torch.fft.ifft2(pFT * torch.conj(pFT))

        # normalize OTF by maximum along each section
        OTFmag = torch.abs(OTF)
        # to do this, collapse y & x axes using 'view', get max along dim=1
        OTFmax, max_id = torch.max(OTFmag.view(Nz, -1), dim=1)
        return OTF / OTFmax.reshape(Nz, 1, 1, 1)

    def to_cuda(self, GPU_device):
        """Move the PupilFunction to the GPU for fast computation

        This is meant to emulate the .cuda() function for torch.tensor.
        Under the hood, this moves all of the relevant quantities for computing
        the PupilFunction-derived PSF into the GPU. Call this function before
        doing any calculations. When called it should notify users that the
        PupilFunction has been moved to the GPU.

        """
        # call before computing PSF
        # before moving, recompute all quantities and move relevant quantities
        # into the GPU
        self.kx = self.kx.to(GPU_device)
        self.ky = self.ky.to(GPU_device)
        self.kz = self.kz.to(GPU_device)
        self.theta_1 = self.theta_1.to(GPU_device)
        self.theta_2 = self.theta_2.to(GPU_device)
        self.A = self.A.to(GPU_device)
        self.mask = self.mask.to(GPU_device)

        # do the same for pupil function
        if self.pupil is None:
            # if there are no loaded pupil functions, get an ideal one
            self.pupil0 = self.pupil0.to(GPU_device)
        else:
            # if the current pupil is in CPU, move to GPU
            if self.pupil.device.type == "cpu":
                self.pupil = self.pupil.to(GPU_device)
                self.pupil0 = self.pupil0.to(GPU_device)
            elif self.pupil.device.type == "cuda":
                print("self.pupil is already in the GPU")
                pass
        print("PupilFunction now in GPU")

    def __str__(self):
        # for convenience, this will inform user about PupilFunction parameters
        # without having to check every variables
        logstr = "Nx = {:d}, Ny = {:d}\n"
        logstr += "NA = {:.2f}, wavelength = {:.3f}\n"
        logstr += "dx = {:.3f}, dy = {:.3f}\n"
        logstr += "ni = {:.3f}, ns = {:.3f}\n"
        return logstr.format(
            self.Nx, self.Ny, self.NA, self.wvlen, self.dx, self.dy, self.ni, self.ns
        )
