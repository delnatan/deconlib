"""Shared deconvolution domain and finite-detector padding helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DetectorPadding = tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class DeconvolutionDomains:
    """Resolved array domains for detector-aware deconvolution.

    The domains are named in the adjoint direction:

    ``data_shape``
        Measured detector samples.
    ``detector_domain_shape``
        Measured samples embedded in the padded detector domain. This is where
        finite-detector padding for objects outside the measured chip lives.
    ``visible_shape``
        Object-space samples after optional detector-resampling adjoint.

    PSF FFT padding is intentionally not part of this structure. Linear blur
    operators decide their internal FFT canvas from ``visible_shape`` and the
    actual PSF shape.
    """

    data_shape: tuple[int, ...]
    detector_padding: DetectorPadding
    detector_domain_shape: tuple[int, ...]
    visible_shape: tuple[int, ...]
    resampling_factor: tuple[int, ...]


def _shape_tuple(name: str, shape: tuple[Any, ...]) -> tuple[int, ...]:
    result = tuple(int(s) for s in shape)
    if any(s <= 0 for s in result):
        raise ValueError(f"{name} entries must be positive")
    return result


def normalize_detector_padding(
    padding: tuple[Any, ...],
    ndim: int,
) -> DetectorPadding:
    """Normalize detector padding to explicit ``(before, after)`` pairs.
    """
    if not padding:
        return tuple((0, 0) for _ in range(ndim))
    if len(padding) != ndim:
        raise ValueError("detector_padding length must match data_shape ndim")

    pairs: list[tuple[int, int]] = []
    for item in padding:
        if isinstance(item, (str, bytes)):
            raise ValueError("detector_padding entries must be (before, after) pairs")
        try:
            item_len = len(item)
        except TypeError as exc:
            raise ValueError(
                "detector_padding entries must be (before, after) pairs"
            ) from exc
        if item_len != 2:
            raise ValueError("detector_padding pair entries must have length 2")
        before, after = (int(item[0]), int(item[1]))
        if before < 0 or after < 0:
            raise ValueError("detector_padding values must be non-negative")
        pairs.append((before, after))
    return tuple(pairs)


def detector_padding_from_domain(
    *,
    data_shape: tuple[int, ...],
    detector_domain_shape: tuple[int, ...],
    padding: tuple[Any, ...],
) -> DetectorPadding:
    """Resolve finite-detector padding from explicit padding."""
    data_shape = _shape_tuple("data_shape", data_shape)
    detector_domain_shape = _shape_tuple(
        "detector_domain_shape",
        detector_domain_shape,
    )
    if len(data_shape) != len(detector_domain_shape):
        raise ValueError("data_shape and detector domain shape must have same ndim")
    if any(v < d for d, v in zip(data_shape, detector_domain_shape)):
        raise ValueError("detector domain shape cannot be smaller than data_shape")

    explicit = normalize_detector_padding(padding, len(data_shape))
    if any(before or after for before, after in explicit):
        expected = detector_domain_shape_from_padding(
            data_shape=data_shape,
            padding=explicit,
        )
        if expected != detector_domain_shape:
            raise ValueError(
                "detector_padding implies detector domain shape "
                f"{expected}, but geometry implies {detector_domain_shape}"
            )
        return explicit

    if detector_domain_shape != data_shape:
        raise ValueError(
            "detector_padding is required when detector domain shape "
            f"{detector_domain_shape} differs from data_shape {data_shape}"
        )
    return explicit


def detector_domain_shape_from_padding(
    *,
    data_shape: tuple[int, ...],
    padding: tuple[Any, ...],
) -> tuple[int, ...]:
    """Return the padded detector-domain shape implied by detector padding."""
    data_shape = _shape_tuple("data_shape", data_shape)
    padding_pairs = normalize_detector_padding(padding, len(data_shape))
    return tuple(
        d + before + after
        for d, (before, after) in zip(data_shape, padding_pairs)
    )


def normalize_resampling_factor(
    factor: tuple[int, ...],
    ndim: int,
) -> tuple[int, ...]:
    """Normalize optional detector-resampling factors."""
    result = tuple(int(f) for f in (factor or ()))
    if not result:
        return (1,) * ndim
    if len(result) != ndim:
        raise ValueError("resampling_factor length must match data_shape ndim")
    if any(f <= 0 for f in result):
        raise ValueError("resampling_factor values must be positive")
    return result


def detector_domain_from_visible_shape(
    *,
    visible_shape: tuple[int, ...],
    resampling_factor: tuple[int, ...],
) -> tuple[int, ...]:
    """Return the low-res detector domain that resamples from visible space."""
    visible_shape = _shape_tuple("visible_shape", visible_shape)
    factor = normalize_resampling_factor(resampling_factor, len(visible_shape))
    detector_domain = []
    for visible_n, f in zip(visible_shape, factor):
        if visible_n % f:
            raise ValueError(
                "visible_shape must be divisible by resampling_factor"
            )
        detector_domain.append(visible_n // f)
    return tuple(detector_domain)


def resolve_deconvolution_domains(
    *,
    data_shape: tuple[int, ...],
    visible_shape: tuple[int, ...],
    detector_padding: tuple[Any, ...] = (),
    resampling_factor: tuple[int, ...] = (),
) -> DeconvolutionDomains:
    """Resolve detector and visible domains for a deconvolution recipe."""
    data_shape = _shape_tuple("data_shape", data_shape)
    visible_shape = _shape_tuple("visible_shape", visible_shape)
    if len(data_shape) != len(visible_shape):
        raise ValueError("data_shape and visible_shape must have same ndim")

    factor = normalize_resampling_factor(resampling_factor, len(data_shape))
    detector_domain_shape = detector_domain_from_visible_shape(
        visible_shape=visible_shape,
        resampling_factor=factor,
    )
    padding = detector_padding_from_domain(
        data_shape=data_shape,
        detector_domain_shape=detector_domain_shape,
        padding=detector_padding,
    )
    expected_detector = detector_domain_shape_from_padding(
        data_shape=data_shape,
        padding=padding,
    )
    if expected_detector != detector_domain_shape:
        raise ValueError(
            "detector padding and geometry imply detector domain "
            f"{expected_detector}, but resolved {detector_domain_shape}"
        )
    expected_visible = tuple(
        d * f for d, f in zip(detector_domain_shape, factor)
    )
    if expected_visible != visible_shape:
        raise ValueError(
            "detector domain and resampling_factor imply visible_shape "
            f"{expected_visible}, but geometry has {visible_shape}"
        )

    return DeconvolutionDomains(
        data_shape=data_shape,
        detector_padding=padding,
        detector_domain_shape=detector_domain_shape,
        visible_shape=visible_shape,
        resampling_factor=factor,
    )
