
"""Tests for Sentinel-2 SCL cloud and invalid-pixel masking."""

from unittest.mock import MagicMock, call

from app.services.ndvi_pipeline import mask_sentinel2_clouds


def test_cloud_mask_excludes_expected_scl_classes():
    """Mask no-data, defective, shadow, cloud, cirrus and snow classes."""
    image = MagicMock()
    scl = MagicMock()
    valid_pixels = MagicMock()

    image.select.return_value = scl

    # Chain the expected Earth Engine mask expression.
    scl.neq.return_value = valid_pixels
    valid_pixels.And.return_value = valid_pixels

    result = mask_sentinel2_clouds(image)

    image.select.assert_called_once_with("SCL")

    expected_excluded_classes = [0, 1, 3, 7, 8, 9, 10, 11]
    actual_calls = scl.neq.call_args_list

    assert actual_calls == [
        call(class_value)
        for class_value in expected_excluded_classes
    ]

    # The mask must be applied to the original image.
    image.updateMask.assert_called_once_with(valid_pixels)
    assert result == image.updateMask.return_value


def test_cloud_mask_keeps_clear_classes_unexcluded():
    """Clear vegetation, bare soil and water classes are not excluded."""
    image = MagicMock()
    scl = MagicMock()
    valid_pixels = MagicMock()

    image.select.return_value = scl
    scl.neq.return_value = valid_pixels
    valid_pixels.And.return_value = valid_pixels

    mask_sentinel2_clouds(image)

    excluded_classes = {
        args[0]
        for args, _ in scl.neq.call_args_list
    }

    # SCL class 2 (dark area), 4 (vegetation), 5 (bare soil),
    # 6 (water), and 7 (unclassified) are not explicitly excluded.
    assert excluded_classes == {0, 1, 3, 7, 8, 9, 10, 11}
    assert 4 not in excluded_classes
    assert 5 not in excluded_classes
    assert 6 not in excluded_classes
