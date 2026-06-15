"""Tests for shared deconvolution domain shape helpers."""

import pytest

from deconlib.domains import (
    detector_domain_shape_from_padding,
    normalize_detector_padding,
    resolve_deconvolution_domains,
)


def test_resolve_domains_places_padding_on_detector_domain():
    domains = resolve_deconvolution_domains(
        data_shape=(5, 7),
        detector_padding=((1, 2), (0, 3)),
        visible_shape=(16, 30),
        resampling_factor=(2, 3),
    )

    assert domains.data_shape == (5, 7)
    assert domains.detector_padding == ((1, 2), (0, 3))
    assert domains.detector_domain_shape == (8, 10)
    assert domains.visible_shape == (16, 30)
    assert domains.resampling_factor == (2, 3)


def test_detector_padding_requires_explicit_before_after_pairs():
    with pytest.raises(ValueError, match="before, after"):
        normalize_detector_padding((1, 2), ndim=2)
    assert normalize_detector_padding(((1, 2), (3, 4)), ndim=2) == (
        (1, 2),
        (3, 4),
    )
    assert detector_domain_shape_from_padding(
        data_shape=(5, 7),
        padding=((1, 2), (3, 4)),
    ) == (8, 14)


def test_resolve_domains_rejects_visible_shape_not_matching_resampling():
    with pytest.raises(ValueError, match="visible_shape must be divisible"):
        resolve_deconvolution_domains(
            data_shape=(5, 7),
            detector_padding=((1, 1), (1, 1)),
            visible_shape=(15, 18),
            resampling_factor=(2, 2),
        )


def test_resolve_domains_requires_padding_when_detector_domain_is_larger():
    with pytest.raises(ValueError, match="detector_padding is required"):
        resolve_deconvolution_domains(
            data_shape=(5, 7),
            visible_shape=(7, 9),
        )
