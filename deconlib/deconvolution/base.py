"""Base types for deconvolution algorithms."""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import mlx.core as mx

__all__ = [
    "DeconvolutionResult",
    "MLXDeconvolutionResult",
    "SICGConfig",
    "PDHGConfig",
]


@dataclass
class DeconvolutionResult:
    """Result from a deconvolution algorithm.

    Attributes:
        restored: The restored image tensor.
        iterations: Number of iterations performed.
        loss_history: Loss/objective value at each iteration (if tracked).
        converged: Whether the algorithm converged to tolerance.
        metadata: Optional algorithm-specific metadata.
    """

    restored: torch.Tensor
    iterations: int
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    metadata: dict = field(default_factory=dict)


@dataclass
class MLXDeconvolutionResult:
    """Result from an MLX-based deconvolution algorithm.

    Attributes:
        restored: The restored image as MLX array.
        iterations: Number of iterations performed.
        loss_history: Loss/objective value at each iteration (if tracked).
        converged: Whether the algorithm converged to tolerance.
        tau_history: Primal step size history (for adaptive algorithms).
        sigma_history: Dual step size history (for adaptive algorithms).
        metadata: Optional algorithm-specific metadata.
    """

    restored: "mx.array"
    iterations: int
    loss_history: List[float] = field(default_factory=list)
    converged: bool = False
    tau_history: List[float] = field(default_factory=list)
    sigma_history: List[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class SICGConfig:
    """Configuration for the SI-CG (Spatially Invariant Conjugate Gradient) solver.

    SI-CG uses square-root parametrization (f = c²) to ensure non-negativity,
    with Fletcher-Reeves conjugate gradient and Newton-Raphson line search.

    Example:
        ```python
        from deconlib.deconvolution import (
            make_fft_convolver, solve_sicg, SICGConfig
        )

        # Create operators
        C, C_adj = make_fft_convolver(psf, device="cuda")

        # Create config
        config = SICGConfig(
            beta=0.01,
            background=100.0,
            restart_interval=10,
            line_search_iter=5,
        )

        # Use with solver - unpack config to kwargs
        result = solve_sicg(
            observed, C, C_adj,
            num_iter=100,
            **config.to_solver_kwargs()
        )

        # Create variants using dataclasses.replace
        from dataclasses import replace
        stronger_reg = replace(config, beta=0.05)
        ```

    Attributes:
        beta: Regularization weight. Controls smoothness vs data fidelity.
            Larger values produce smoother results. Typical range: 1e-4 to 1e-2.
        background: Constant background value in the forward model.
        spacing: Physical grid spacing (dz, dy, dx) or (dy, dx). Used for
            volume-consistent regularization in super-resolution mode. If None,
            uses unit spacing.
        restart_interval: Reset conjugate direction every N iterations to
            prevent direction degradation.
        line_search_iter: Number of Newton-Raphson iterations for step size.
    """

    beta: float = 0.001
    background: float = 0.0
    spacing: Optional[Tuple[float, ...]] = None
    restart_interval: int = 5
    line_search_iter: int = 3

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Convert config to keyword arguments for solve_sicg."""
        return {
            "beta": self.beta,
            "background": self.background,
            "spacing": self.spacing,
            "restart_interval": self.restart_interval,
            "line_search_iter": self.line_search_iter,
        }


@dataclass
class PDHGConfig:
    """Configuration for the Chambolle-Pock PDHG (Primal-Dual Hybrid Gradient) solver.

    PDHG solves Poisson deconvolution with sparse regularization:
        min_{x>=0}  KL(b || Ax + bg) + alpha * R(Lx)

    where R is either L1 or L2 norm applied to the regularization operator L.

    Example:
        ```python
        from deconlib.deconvolution import (
            make_fft_convolver, make_binned_convolver,
            solve_chambolle_pock, PDHGConfig
        )

        # Standard deconvolution with Hessian regularization
        C, C_adj = make_fft_convolver(psf, device="cuda")

        config = PDHGConfig(
            alpha=0.001,
            regularization="hessian",
            norm="L2",                     # Isotropic, avoids blocky artifacts
            spacing=(0.3, 0.1, 0.1),       # (dz, dy, dx) in microns
            background=50.0,
        )

        result = solve_chambolle_pock(
            observed, C, C_adj,
            num_iter=200,
            **config.to_solver_kwargs()
        )

        # Super-resolution with binned operators
        A, A_adj, op_norm_sq = make_binned_convolver(psf_fine, bin_factor=2)

        config_sr = PDHGConfig(
            alpha=0.001,
            blur_norm_sq=op_norm_sq,       # Important: use returned norm
            spacing=(0.05, 0.05),          # Fine grid spacing
            norm="L1",
        )

        result = solve_chambolle_pock(
            observed, A, A_adj,
            num_iter=200,
            init_shape=psf_fine.shape,     # Fine grid shape
            **config_sr.to_solver_kwargs()
        )

        # Reuse configs across multiple images
        widefield_config = PDHGConfig(
            alpha=0.005,
            regularization="hessian",
            norm="L2",
            spacing=(0.3, 0.1, 0.1),
        )

        for stack in stacks:
            C, C_adj = make_fft_convolver(psf, device="cuda")
            result = solve_chambolle_pock(
                stack, C, C_adj,
                num_iter=150,
                **widefield_config.to_solver_kwargs()
            )

        # Create variants using dataclasses.replace
        from dataclasses import replace
        stronger_reg = replace(widefield_config, alpha=0.05)
        l1_variant = replace(widefield_config, norm="L1")
        ```

    Attributes:
        alpha: Regularization weight. Larger = smoother/sparser result.
        regularization: Type of regularization operator L:
            - "hessian": All second derivatives, weighted by spacing
            - "identity": L=I, yielding sparsity penalty on x directly
        norm: Type of norm for regularization:
            - "L1": Anisotropic, soft-thresholds each component independently.
                    Can produce sharper edges but may have axis-aligned artifacts.
            - "L2": Isotropic, joint soft-thresholding across components at each
                    pixel. Promotes sparse derivatives while avoiding blocky artifacts.
        spacing: Physical grid spacing (dz, dy, dx) or (dy, dx). Used to weight
            derivative terms for isotropic regularization in physical space.
            Coarser spacing = less weight. If None, assumes unit spacing.
        background: Constant background in forward model (Ax + background).
        blur_norm_sq: Squared operator norm ||A||². Default 1.0 is correct for
            convolution with normalized PSF. When using make_binned_convolver,
            pass the returned norm_sq for correct step sizes.
        accelerate: If True (default), use FISTA-style momentum with adaptive
            restart for O(1/k²) convergence. Typically 2-3x faster.
        theta: Overrelaxation parameter. Only used when accelerate=False.
            With accelerate=True, this is ignored and momentum is adaptive.
    """

    alpha: float = 0.01
    regularization: Literal["hessian", "identity"] = "hessian"
    norm: Literal["L1", "L2"] = "L1"
    spacing: Optional[Tuple[float, ...]] = None
    background: float = 0.0
    blur_norm_sq: float = 1.0
    accelerate: bool = True
    theta: float = 1.0  # Only used when accelerate=False

    def to_solver_kwargs(self) -> Dict[str, Any]:
        """Convert config to keyword arguments for solve_chambolle_pock."""
        return {
            "alpha": self.alpha,
            "regularization": self.regularization,
            "norm": self.norm,
            "spacing": self.spacing,
            "background": self.background,
            "blur_norm_sq": self.blur_norm_sq,
            "accelerate": self.accelerate,
            "theta": self.theta,
        }
