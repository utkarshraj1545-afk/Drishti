
from app.services import ndvi_pipeline
from unittest.mock import MagicMock, patch

import pytest

from app.services.ndvi_pipeline import get_valid_pixel_coverage


def build_ee_mocks(
    total_count,
    valid_count,
    total_area=10000,
    valid_area=None,
):
    """Mock the Earth Engine calls used by the coverage function."""
    if valid_area is None:
        valid_area = (
            total_area * valid_count / total_count
            if total_count
            else 0
        )

    ndvi_image = MagicMock()
    aoi = MagicMock()
    projection = "test-projection"

    reduction = MagicMock()
    reduction.getInfo.return_value = {
        "total_pixels": total_count,
        "valid_pixels": valid_count,
        "total_area": total_area,
        "valid_area": valid_area,
    }

    with patch(
        "app.services.ndvi_pipeline.ee.Image.constant"
    ) as constant_image, patch(
        "app.services.ndvi_pipeline.ee.Image.pixelArea"
    ) as pixel_area_image, patch(
        "app.services.ndvi_pipeline.ee.Reducer.sum"
    ) as sum_reducer:

        total_image = MagicMock()
        constant_image.return_value = total_image

        pixel_area = MagicMock()
        pixel_area_image.return_value = pixel_area

        # Keep image operations chainable.
        for image in (total_image, pixel_area):
            image.rename.return_value = image
            image.setDefaultProjection.return_value = image
            image.clip.return_value = image
            image.addBands.return_value = image

        # The total and valid pixel bands are combined.
        total_image.addBands.return_value = total_image

        ndvi_image.projection.return_value = projection

        mask = MagicMock()
        ndvi_image.mask.return_value = mask
        mask.gt.return_value.unmask.return_value.rename.return_value.setDefaultProjection.return_value.clip.return_value = MagicMock()

        # Valid-area image uses pixelArea().updateMask(...).
        pixel_area.updateMask.return_value = pixel_area

        # All chained bands ultimately lead to the same mocked reduction.
        total_image.reduceRegion.return_value = reduction
        pixel_area.reduceRegion.return_value = reduction

        # Route the final combined reduction through the mocked result.
        with patch(
            "app.services.ndvi_pipeline.ee.Image",
        ) as image_class:
            image_class.constant.return_value = total_image
            image_class.pixelArea.return_value = pixel_area

            result = get_valid_pixel_coverage(
                ndvi_image,
                aoi,
            )

    return result, reduction


def test_coverage_calculates_percentage_from_pixel_area():
    result, _ = build_ee_mocks(
        total_count=1000,
        valid_count=750,
        total_area=10000,
        valid_area=7500,
    )

    assert result["total_pixel_count"] == 1000
    assert result["valid_pixel_count"] == 750
    assert result["coverage_percent"] == 75.0


def test_coverage_is_100_percent_when_all_sampled_pixels_are_valid():
    result, _ = build_ee_mocks(
        total_count=1000,
        valid_count=1000,
        total_area=10000,
        valid_area=10000,
    )

    assert result["coverage_percent"] == 100.0


def test_coverage_is_zero_when_no_pixels_are_valid():
    result, _ = build_ee_mocks(
        total_count=1000,
        valid_count=0,
        total_area=10000,
        valid_area=0,
    )

    assert result["coverage_percent"] == 0.0


def test_coverage_rejects_nonpositive_scale():
    with pytest.raises(
        ValueError,
        match="scale must be greater than zero",
    ):
        get_valid_pixel_coverage(
            MagicMock(),
            MagicMock(),
            scale=0,
        )


def test_coverage_uses_area_even_when_pixel_counts_suggest_other_value():
    result, _ = build_ee_mocks(
        total_count=1000,
        valid_count=900,
        total_area=10000,
        valid_area=8000,
    )

    # Pixel counts suggest 90%, but the area-based result is 80%.
    assert result["valid_pixel_count"] == 900
    assert result["total_pixel_count"] == 1000
    assert result["coverage_percent"] == 80.0


def test_low_coverage_produces_warning():
    """Warn when valid-pixel coverage is below the threshold."""
    aoi = MagicMock()
    collection = MagicMock()
    collection.size.return_value.getInfo.return_value = 3
    composite = MagicMock()
    ndvi_image = MagicMock()

    low_coverage = {
        "valid_pixel_count": 600,
        "total_pixel_count": 1000,
        "coverage_percent": 60.0,
    }

    with patch.object(
        ndvi_pipeline,
        "get_cloud_filtered_collection",
        return_value=collection,
    ), patch.object(
        ndvi_pipeline,
        "build_composite",
        return_value=composite,
    ), patch.object(
        ndvi_pipeline,
        "compute_ndvi_image",
        return_value=ndvi_image,
    ), patch.object(
        ndvi_pipeline,
        "get_valid_pixel_coverage",
        return_value=low_coverage,
    ), patch.object(
        ndvi_pipeline,
        "get_mean_ndvi",
        return_value=0.25,
    ), patch.object(
        ndvi_pipeline,
        "COVERAGE_WARNING_THRESHOLD",
        70,
    ), patch("builtins.print") as mock_print:

        result = ndvi_pipeline.get_ndvi_for_period(
            aoi,
            "2024-10-01",
            "2024-10-31",
        )

    assert result["coverage_warning"] is not None
    assert "60.00%" in result["coverage_warning"]
    assert "cautiously" in result["coverage_warning"]
    mock_print.assert_called_once()


def test_acceptable_coverage_does_not_produce_warning():
    """Do not warn when coverage meets the configured threshold."""
    aoi = MagicMock()
    collection = MagicMock()
    collection.size.return_value.getInfo.return_value = 3
    composite = MagicMock()
    ndvi_image = MagicMock()

    acceptable_coverage = {
        "valid_pixel_count": 800,
        "total_pixel_count": 1000,
        "coverage_percent": 80.0,
    }

    with patch.object(
        ndvi_pipeline,
        "get_cloud_filtered_collection",
        return_value=collection,
    ), patch.object(
        ndvi_pipeline,
        "build_composite",
        return_value=composite,
    ), patch.object(
        ndvi_pipeline,
        "compute_ndvi_image",
        return_value=ndvi_image,
    ), patch.object(
        ndvi_pipeline,
        "get_valid_pixel_coverage",
        return_value=acceptable_coverage,
    ), patch.object(
        ndvi_pipeline,
        "get_mean_ndvi",
        return_value=0.25,
    ), patch.object(
        ndvi_pipeline,
        "COVERAGE_WARNING_THRESHOLD",
        70,
    ), patch("builtins.print") as mock_print:

        result = ndvi_pipeline.get_ndvi_for_period(
            aoi,
            "2024-10-01",
            "2024-10-31",
        )

    assert result["coverage_warning"] is None
    mock_print.assert_not_called()
