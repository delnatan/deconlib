"""Phase retrieval algorithms for pupil function recovery.

This module is temporarily separated from the main deconlib API while we
focus on stabilizing the deconvolution API. It contains:

- Gerchberg-Saxton (GS), Error Reduction (ER), and HIO phase retrieval
- Gradient-based MAP retrieval with MLX backend
- Pupil real-space regularization filters
- Pupil I/O (save/load .pupil.h5 files)

These functions may be reintegrated into the main API in a future release.
"""

from .io import (
    Pupil,
    load_pupil,
    save_pupil,
)
from .retrieval import (
    PhaseRetrievalResult,
    make_pupil_real_filter,
    retrieve_phase,
    retrieve_phase_vectorial,
)
from .retrieval_mlx import (
    MLXRetrievalConfig,
    retrieve_phase_vectorial_mlx,
)

__all__ = [
    # Pupil I/O
    "Pupil",
    "load_pupil",
    "save_pupil",
    # Phase retrieval
    "PhaseRetrievalResult",
    "make_pupil_real_filter",
    "retrieve_phase",
    "retrieve_phase_vectorial",
    # MLX retrieval
    "MLXRetrievalConfig",
    "retrieve_phase_vectorial_mlx",
]
