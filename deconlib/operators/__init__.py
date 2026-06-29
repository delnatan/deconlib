"""Clean operator construction for deconvolution.

This module provides a transparent, composable API for building forward operators
for deconvolution. The philosophy is:

1. Each operator does ONE thing explicitly
2. Operators are composed using the `compose()` function
3. All shape and padding information is explicit
4. The adjoint is automatically correct when operators are composed

The forward model for deconvolution is:

    hidden-space -> [ICF transform] -> visible-space -> [PSF convolution] -> 
    [binning/downsampling] -> [finite detector crop] -> data-space

For standard deconvolution (no super-resolution):
    - visible-space = data-space (same pixel size)
    - No binning needed

For super-resolution:
    - visible-space has finer pixels than data-space
    - Binning/downsampling is required

For finite detector modeling:
    - visible-space is larger than the field of view
    - Allows objects outside the detector to contribute to edge pixels
"""

from .io import save_restored_as_ims

__all__ = [
    "save_restored_as_ims",
]
