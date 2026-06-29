"""I/O utilities for deconvolution results.

This module provides helper functions to save deconvolution results
to various formats, including Imaris .ims files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    from pyvistra.io import save_imaris
    HAS_PYVISTRA = True
except ImportError:
    HAS_PYVISTRA = False
    save_imaris = None

def save_restored_as_ims(
    filepath: Union[str, Path],
    restored: np.ndarray,
    *,
    original_data: Optional[np.ndarray] = None,
    original_metadata: Optional[Dict[str, Any]] = None,
    pixel_spacing: Optional[Tuple[float, ...]] = None,
    channel_names: Optional[list] = None,
    resolution_levels: bool = True,
    progress_cb: Optional[callable] = None,
) -> None:
    """Save deconvolution result as Imaris .ims file.
    
    This function takes a restored array (in visible-space) and saves it in 
    Imaris format (.ims). The restored array is assumed to be in visible-space
    (the reconstruction domain), not data-space. For super-resolution, this
    means the array has finer pixels than the original data.
    
    The restored array will be converted to the required 5D (T, Z, C, Y, X) format.
    
    Args:
        filepath: Output path (should end with .ims).
        restored: The deconvolved image array in visible-space. Can be:
            - 2D: (Y, X)
            - 3D: (Z, Y, X) - most common for single-channel 3D deconvolution
            - 4D: (C, Z, Y, X)
            - 5D: (T, Z, C, Y, X)
        original_data: Optional original data array to copy metadata from.
        original_metadata: Optional metadata dict (should contain 'scale' key).
        pixel_spacing: Optional tuple of pixel spacing values in micrometers
            for the visible-space. If None and original_metadata is provided,
            uses 'scale' from metadata. Otherwise, defaults to (1.0, 1.0, ...).
        channel_names: Optional list of channel names. Defaults to ['Channel 0'], etc.
        resolution_levels: Whether to generate downsampled pyramid levels.
            Default is True for better viewing in Imaris.
        progress_cb: Optional callback function for progress updates.
    
    Raises:
        ImportError: If pyvistra is not installed.
        ValueError: If the restored array has more than 5 dimensions.
    
    Note:
        For super-resolution deconvolution, the pixel_spacing should be the
        visible-space pixel spacing (finer than data-space), not the data-space
        pixel spacing.
    
    Example:
        >>> import numpy as np
        >>> from deconlib.operators.io import save_restored_as_ims
        >>> # restored is in visible-space (41, 125, 125) for super-resolution
        >>> restored = np.random.rand(41, 125, 125)
        >>> # Use visible-space pixel spacing
        >>> save_restored_as_ims(
        ...     'output.ims',
        ...     restored,
        ...     pixel_spacing=(0.196, 0.08344, 0.08344),  # visible-space
        ...     channel_names=['Deconvolved']
        ... )
    """
    if not HAS_PYVISTRA:
        raise ImportError(
            "pyvistra is required to save .ims files. "
            "Install with: pip install pyvistra"
        )
    
    filepath = Path(filepath)
    
    # Convert restored to 5D format (T, Z, C, Y, X)
    restored_5d = _to_5d(restored)
    
    # Build metadata
    metadata = _build_metadata(
        restored_5d.shape,
        original_metadata=original_metadata,
        pixel_spacing=pixel_spacing,
        channel_names=channel_names,
    )
    
    # Save using pyvistra
    save_imaris(
        str(filepath),
        restored_5d,
        metadata=metadata,
        resolution_levels=resolution_levels,
        progress_cb=progress_cb,
    )
    print(f"Saved restored data to {filepath}")

def _to_5d(arr: np.ndarray) -> np.ndarray:
    """Convert array to 5D (T, Z, C, Y, X) format.
    
    The Imaris format expects:
    - axis 0: T (time)
    - axis 1: Z (depth)
    - axis 2: C (channel)
    - axis 3: Y (height)
    - axis 4: X (width)
    
    For common cases:
    - 2D (Y, X) -> (1, 1, 1, Y, X)
    - 3D (Z, Y, X) -> (1, Z, 1, Y, X)
    - 3D (C, Y, X) -> (1, 1, C, Y, X)
    - 4D (C, Z, Y, X) -> (1, Z, C, Y, X)
    - 4D (T, Z, Y, X) -> (T, Z, 1, Y, X)
    - 5D (T, Z, C, Y, X) -> (T, Z, C, Y, X)
    """
    ndim = arr.ndim
    if ndim > 5:
        raise ValueError(f"Array has {ndim} dimensions, max 5 supported")
    
    # Strategy: assume the last 2 dimensions are always (Y, X)
    # and the last 3 are (Z, Y, X) or (C, Y, X)
    # We need to add T and/or C dimensions at the front
    
    # For 2D: (Y, X) -> (1, 1, 1, Y, X)
    # For 3D: (Z, Y, X) -> (1, Z, 1, Y, X) OR (C, Y, X) -> (1, 1, C, Y, X)
    # For 4D: (T, Z, Y, X) -> (T, Z, 1, Y, X) OR (C, Z, Y, X) -> (1, Z, C, Y, X)
    # For 5D: already (T, Z, C, Y, X)
    
    # Default assumption: 
    # - If 3D, assume it's (Z, Y, X) and add T and C
    # - If 4D, assume it's (C, Z, Y, X) or (T, Z, Y, X) - add the missing one
    # This is a bit ambiguous, but we'll use a heuristic
    
    # Convert to numpy if needed
    arr_np = np.asarray(arr, dtype=np.float32)
    
    if ndim == 5:
        return arr_np
    elif ndim == 4:
        # Could be (T, Z, C, Y), (T, Z, Y, X), (T, C, Y, X), (Z, C, Y, X), (C, Z, Y, X)
        # Assume it's missing one of T or C
        # Try to detect: if dims are roughly equal, assume spatial
        # For now, just add T=1 at front: (1, ..., ..., ..., ...)
        return arr_np[np.newaxis, ...]
    elif ndim == 3:
        # Assume (Z, Y, X) -> (1, Z, 1, Y, X)
        return arr_np[np.newaxis, :, np.newaxis, :, :]
    elif ndim == 2:
        # Assume (Y, X) -> (1, 1, 1, Y, X)
        return arr_np[np.newaxis, np.newaxis, np.newaxis, :, :]
    elif ndim == 1:
        # (X) -> (1, 1, 1, 1, X)
        return arr_np[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    else:
        return arr_np

def _build_metadata(
    shape_5d: Tuple[int, ...],
    *,
    original_metadata: Optional[Dict[str, Any]] = None,
    pixel_spacing: Optional[Tuple[float, ...]] = None,
    channel_names: Optional[list] = None,
) -> Dict[str, Any]:
    """Build metadata dict for Imaris file.
    
    Args:
        shape_5d: 5D shape (T, Z, C, Y, X)
        original_metadata: Optional original metadata to copy from
        pixel_spacing: Optional pixel spacing tuple
        channel_names: Optional channel names list
    
    Returns:
        Metadata dict with 'scale', 'channels', etc.
    """
    T, Z, C, Y, X = shape_5d
    
    metadata: Dict[str, Any] = {}
    
    # Build scale (pixel spacing) metadata
    # pyvistra expects scale as 3-tuple: (dz, dy, dx) for (Z, Y, X) spacing
    if pixel_spacing is not None:
        # pixel_spacing should be (dz, dy, dx) or (dt, dz, dy, dx) or (dz, dy, dx, ...)
        # We need to extract the spatial (Z, Y, X) components
        if len(pixel_spacing) >= 3:
            # Take the last 3 as (z, y, x) spacing
            dz = pixel_spacing[-3]
            dy = pixel_spacing[-2]
            dx = pixel_spacing[-1]
        elif len(pixel_spacing) == 2:
            dz = pixel_spacing[0]
            dy = dx = pixel_spacing[1]
        elif len(pixel_spacing) == 1:
            dz = dy = dx = pixel_spacing[0]
        else:
            dz = dy = dx = 1.0
    elif original_metadata and 'scale' in original_metadata:
        # Copy from original - pyvistra uses 3-tuple (z, y, x)
        scale = original_metadata['scale']
        if isinstance(scale, (list, tuple)) and len(scale) >= 3:
            dz = scale[0]
            dy = scale[1]
            dx = scale[2]
        elif isinstance(scale, (list, tuple)) and len(scale) == 3:
            dz, dy, dx = scale
        else:
            dz = dy = dx = 1.0
    else:
        dz = dy = dx = 1.0
    
    # pyvistra expects scale as 3-tuple: (dz, dy, dx)
    metadata['scale'] = (dz, dy, dx)
    
    # Build channels metadata
    if channel_names is not None:
        channels = []
        for i, name in enumerate(channel_names[:C]):
            channels.append({'name': name, 'color': _get_channel_color(i)})
        metadata['channels'] = channels
    else:
        channels = []
        for i in range(C):
            channels.append({'name': f'Channel {i}', 'color': _get_channel_color(i)})
        metadata['channels'] = channels
    
    # Add other standard metadata
    metadata['name'] = 'Deconvolution Result'
    metadata['axes'] = 'TZCYX'
    
    return metadata

def _get_channel_color(index: int) -> Tuple[int, int, int]:
    """Get a default color for a channel."""
    colors = [
        (255, 0, 0),      # Red
        (0, 255, 0),      # Green
        (0, 0, 255),      # Blue
        (255, 255, 0),    # Yellow
        (255, 0, 255),    # Magenta
        (0, 255, 255),    # Cyan
        (255, 128, 0),    # Orange
        (128, 0, 255),    # Purple
    ]
    return colors[index % len(colors)]
