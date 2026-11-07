import numpy as np
import pytest
import torchxrayvision as xrv

from medrax_taixray_audit.adapter import TAIXRAY_UINT16_RULE, normalization_maxval


def test_uint8_behavior_matches_upstream_configuration():
    image = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    expected = xrv.datasets.normalize(image, 255)
    actual = xrv.datasets.normalize(image, normalization_maxval(image))
    np.testing.assert_allclose(actual, expected)


def test_full_range_uint16_matches_rescaled_uint8():
    image_8bit = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    image_16bit = image_8bit.astype(np.uint16) * 257
    normalized_8bit = xrv.datasets.normalize(image_8bit, normalization_maxval(image_8bit))
    normalized_16bit = xrv.datasets.normalize(
        image_16bit, normalization_maxval(image_16bit, 65535)
    )
    np.testing.assert_allclose(normalized_8bit, normalized_16bit)


def test_12bit_uint16_requires_explicit_scale():
    image = np.array([[0, 2048], [4095, 100]], dtype=np.uint16)
    with pytest.raises(ValueError, match="require input_maxval"):
        normalization_maxval(image)
    assert normalization_maxval(image, 4095) == 4095.0


def test_dataset_specific_rule_uses_fixed_known_ranges_not_image_maximum():
    low_range = np.array([[0, 2000], [4095, 100]], dtype=np.uint16)
    wider_range = np.array([[16, 32000], [65280, 40]], dtype=np.uint16)
    assert normalization_maxval(low_range, TAIXRAY_UINT16_RULE) == 4095.0
    assert normalization_maxval(wider_range, TAIXRAY_UINT16_RULE) == 65535.0
