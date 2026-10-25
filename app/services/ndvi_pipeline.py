
"""
NDVI Pipeline for DRISHTI
=========================

Deterministic Sentinel-2 vegetation analysis using Google Earth Engine.

Features:
- Sentinel-2 Surface Reflectance Harmonized imagery
- Per-pixel SCL cloud, shadow, cirrus, and snow masking
- Median image compositing
- NDVI calculation using Sentinel-2 B8 and B4
- Mean NDVI and valid-pixel coverage statistics
- Low-coverage warnings and empty-data validation
- Percentage change and vegetation-change classification
"""

import math
import ee


# Initial coverage warning threshold.
# This is a configurable project threshold, not a universal scientific standard.
COVERAGE_WARNING_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_earth_engine(service_account_email: str, key_path: str):
    """
    Authenticate and initialize Google Earth Engine using a service account.
    """
    credentials = ee.ServiceAccountCredentials(
        service_account_email,
        key_path,
    )
    ee.Initialize(credentials)


# ---------------------------------------------------------------------------
# Step 1: Resolve area of interest
# ---------------------------------------------------------------------------

def resolve_aoi(
    west: float,
    south: float,
    east: float,
    north: float,
) -> ee.Geometry:
    """
    Convert a longitude/latitude bounding box into an Earth Engine rectangle.
    """
    if not all(
        math.isfinite(value)
        for value in (west, south, east, north)
    ):
        raise ValueError("AOI coordinates must be finite numbers.")

    if not (-180 <= west <= 180 and -180 <= east <= 180):
        raise ValueError("Longitude must be between -180 and 180.")

    if not (-90 <= south <= 90 and -90 <= north <= 90):
        raise ValueError("Latitude must be between -90 and 90.")

    if west >= east:
        raise ValueError("West longitude must be less than east longitude.")

    if south >= north:
        raise ValueError("South latitude must be less than north latitude.")

    return ee.Geometry.Rectangle([west, south, east, north])


# ---------------------------------------------------------------------------
# Step 2: Sentinel-2 cloud and invalid-pixel masking
# ---------------------------------------------------------------------------

def mask_sentinel2_clouds(image: ee.Image) -> ee.Image:
    """
    Mask invalid and cloud-contaminated Sentinel-2 pixels using SCL.

    Excluded SCL classes:
        0  = No data
        1  = Saturated or defective pixels
        3  = Cloud shadows
        7  = Low-probability cloud or unclassified
        8  = Clouds, medium probability
        9  = Clouds, high probability
        10 = Cirrus
        11 = Snow or ice

    The mask is applied to every image before compositing.
    """
    scl = image.select("SCL")

    valid_pixels = (
        scl.neq(0)
        .And(scl.neq(1))
        .And(scl.neq(3))
        .And(scl.neq(7))
        .And(scl.neq(8))
        .And(scl.neq(9))
        .And(scl.neq(10))
        .And(scl.neq(11))
    )

    return image.updateMask(valid_pixels)


def get_cloud_filtered_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 20,
) -> ee.ImageCollection:
    """
    Fetch Sentinel-2 SR Harmonized scenes for an AOI and date range.

    First filters scenes by their overall cloud percentage, then applies
    the per-pixel SCL mask.

    Note: scene-level cloud percentage does not represent the cloud
    percentage inside the selected AOI.
    """
    if not 0 <= max_cloud_pct <= 100:
        raise ValueError("max_cloud_pct must be between 0 and 100.")

    if start_date >= end_date:
        raise ValueError("Start date must be earlier than end date.")

    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                max_cloud_pct,
            )
        )
        .map(mask_sentinel2_clouds)
    )

    return collection


# ---------------------------------------------------------------------------
# Step 3: Build median composite
# ---------------------------------------------------------------------------

def build_composite(collection: ee.ImageCollection) -> ee.Image:
    """
    Build a per-pixel median composite from the masked image collection.
    """
    return collection.median()


# ---------------------------------------------------------------------------
# Step 4: Compute NDVI
# ---------------------------------------------------------------------------

def compute_ndvi_image(composite: ee.Image) -> ee.Image:
    """
    Calculate NDVI using Sentinel-2 bands:

        NDVI = (B8 - B4) / (B8 + B4)

    B8 = Near-infrared
    B4 = Red

    Earth Engine normalizedDifference() masks pixels where either input
    band is negative. Such pixels will therefore not count as valid NDVI
    pixels in the coverage calculation.
    """
    return composite.normalizedDifference(
        ["B8", "B4"]
    ).rename("NDVI")


# ---------------------------------------------------------------------------
# Step 5: Calculate mean NDVI
# ---------------------------------------------------------------------------

def get_mean_ndvi(
    ndvi_image: ee.Image,
    aoi: ee.Geometry,
    scale: int = 10,
) -> float:
    """
    Calculate the mean NDVI over the AOI using valid pixels only.

    Raises ValueError if the AOI contains no usable NDVI pixels.
    """
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e9,
        tileScale=4,
    ).getInfo()

    mean_ndvi = stats.get("NDVI")

    if mean_ndvi is None:
        raise ValueError(
            "No valid NDVI pixels remain for this AOI and date range. "
            "Try different dates or review the cloud mask."
        )

    mean_ndvi = float(mean_ndvi)

    if not math.isfinite(mean_ndvi):
        raise ValueError("The calculated mean NDVI is not finite.")

    return mean_ndvi


# ---------------------------------------------------------------------------
# Step 6: Calculate valid-pixel coverage
# ---------------------------------------------------------------------------


def get_valid_pixel_coverage(
    ndvi_image: ee.Image,
    aoi: ee.Geometry,
    scale: int = 10,
) -> dict:
    """
    Calculate valid-pixel coverage using pixel area.
    Pixel counts are returned for reporting, while coverage_percent
    is calculated from valid area divided by total area.

    """
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")

    projection = ndvi_image.projection()

    total_image = (
        ee.Image.constant(1)
        .rename("total_pixels")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    valid_image = (
        ndvi_image.mask()
        .gt(0)
        .unmask(0, False)
        .rename("valid_pixels")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    pixel_area = ee.Image.pixelArea().setDefaultProjection(
        projection
    )

    total_area_image = (
        pixel_area
        .updateMask(total_image)
        .rename("total_area")
        .clip(aoi)
    )

    valid_area_image = (
        pixel_area
        .updateMask(valid_image)
        .rename("valid_area")
        .clip(aoi)
    )

    combined = (
        total_image
        .addBands(valid_image)
        .addBands(total_area_image)
        .addBands(valid_area_image)
    )

    stats = combined.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=scale,
        crs=projection,
        maxPixels=1e9,
        tileScale=4,
    ).getInfo()

    total_count = stats.get("total_pixels")
    valid_count = stats.get("valid_pixels")
    total_area = stats.get("total_area")
    valid_area = stats.get("valid_area")

    if (
        total_count is None
        or valid_count is None
        or total_area is None
        or valid_area is None
    ):
        raise ValueError(
            "Could not calculate pixel coverage for this AOI. "
            "Check the coordinates and satellite imagery."
        )

    total_count = int(total_count)
    valid_count = int(valid_count)
    total_area = float(total_area)
    valid_area = float(valid_area)

    if total_count <= 0 or total_area <= 0:
        raise ValueError(
            "No sampled pixels were found inside the AOI."
        )

    valid_count = min(max(valid_count, 0), total_count)
    valid_area = min(max(valid_area, 0.0), total_area)

    coverage_percent = valid_area / total_area * 100.0

    return {
        "valid_pixel_count": valid_count,
        "total_pixel_count": total_count,
        "coverage_percent": round(coverage_percent, 2),
    }




# ---------------------------------------------------------------------------
# Steps 2-7: Analyze one date period
# ---------------------------------------------------------------------------

def get_ndvi_for_period(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 20,
) -> dict:
    """
    Analyze one date period and return NDVI, scene count, and coverage.

    Returns:
        mean_ndvi
        composite_image
        scene_count
        valid_pixel_count
        total_pixel_count
        coverage_percent
        coverage_warning

    Raises ValueError if no scenes or no usable NDVI pixels are available.
    """
    collection = get_cloud_filtered_collection(
        aoi,
        start_date,
        end_date,
        max_cloud_pct,
    )

    scene_count = collection.size().getInfo()

    if scene_count == 0:
        raise ValueError(
            f"No Sentinel-2 scenes met the cloud filter for this AOI "
            f"between {start_date} and {end_date}. "
            f"The scene cloud threshold is {max_cloud_pct}%. "
            "Try a wider date range or a higher cloud threshold."
        )

    composite = build_composite(collection)
    ndvi_image = compute_ndvi_image(composite)

    # Check coverage before reporting the mean NDVI.
    coverage = get_valid_pixel_coverage(ndvi_image, aoi)

    if coverage["valid_pixel_count"] == 0:
        raise ValueError(
            f"No valid NDVI pixels remain for {start_date} to {end_date}. "
            "Cloud masking may have removed all usable pixels. "
            "Try a different date range or review the AOI."
        )

    mean_ndvi = get_mean_ndvi(ndvi_image, aoi)

    coverage_warning = None

    if coverage["coverage_percent"] < COVERAGE_WARNING_THRESHOLD:
        coverage_warning = (
            f"Low valid-pixel coverage: "
            f"{coverage['coverage_percent']:.2f}% of sampled AOI pixels "
            f"were usable. Interpret this period's NDVI cautiously."
        )

        print(
            f"WARNING: {coverage_warning} "
            f"Period: {start_date} to {end_date}"
        )

    return {
        "mean_ndvi": mean_ndvi,
        "composite_image": composite,
        "scene_count": scene_count,
        "valid_pixel_count": coverage["valid_pixel_count"],
        "total_pixel_count": coverage["total_pixel_count"],
        "coverage_percent": coverage["coverage_percent"],
        "coverage_warning": coverage_warning,
    }


# ---------------------------------------------------------------------------
# Step 8: Compute change between two periods
# ---------------------------------------------------------------------------

def compute_change(
    before_ndvi: float,
    after_ndvi: float,
) -> dict:
    """
    Compare mean NDVI between two periods.

    Percentage change:
        ((after_ndvi - before_ndvi) / before_ndvi) * 100

    A negative change means NDVI decreased.
    A positive change means NDVI increased.

    The percentage is rounded to one decimal place before classification,
    ensuring the displayed percentage and classification use the same value.

    Note:
        NDVI change alone does not prove flooding, vegetation damage,
        or whether an area is safe.
    """
    if not math.isfinite(before_ndvi) or not math.isfinite(after_ndvi):
        raise ValueError("NDVI values must be finite numbers.")

    if abs(before_ndvi) < 1e-6:
        return {
            "pct_change": None,
            "direction": "undetermined",
            "note": (
                "Baseline NDVI is too close to zero. Percentage change "
                "may not be meaningful for this area. Review the absolute "
                "NDVI values and land cover."
            ),
        }

    pct_change = ((after_ndvi - before_ndvi) / before_ndvi) * 100.0
    rounded_change = round(pct_change, 1)

    if rounded_change < -5.0:
        direction = "vegetation loss"
    elif rounded_change > 5.0:
        direction = "vegetation gain"
    else:
        direction = "no significant change"

    return {
        "pct_change": rounded_change,
        "direction": direction,
        "note": (
            "NDVI change indicates a change in vegetation greenness. "
            "It does not independently confirm flooding, vegetation "
            "damage, or whether an area is safe. Cloud cover, seasonal "
            "changes, and image availability can affect the results."
        ),
    }

# ---------------------------------------------------------------------------
# Step 9: Estimate water extent using MNDWI
# ---------------------------------------------------------------------------

def get_mndwi_water_extent(
    composite: ee.Image,
    aoi: ee.Geometry,
) -> dict:
    """
    Estimate water-like surface area using the Modified Normalized
    Difference Water Index (MNDWI).

    This is an estimate, not definitive flood detection.
    """
    green = composite.select("B3")
    swir = composite.select("B11")

    denominator = green.add(swir)

    mndwi = (
        green.subtract(swir)
        .divide(denominator.where(denominator.eq(0), 1))
        .rename("MNDWI")
    )

    # Positive MNDWI is a simple indicator of water-like surfaces.
    water_mask = mndwi.gt(0).rename("water")

    pixel_area = ee.Image.pixelArea()

    total_area = pixel_area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=aoi,
        scale=20,
        maxPixels=1_000_000_000,
        bestEffort=True,
    ).get("area")

    water_area = (
        pixel_area.updateMask(water_mask)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=20,
            maxPixels=1_000_000_000,
            bestEffort=True,
        )
        .get("area")
    )

    total_area_m2 = ee.Number(total_area).getInfo()
    water_area_m2 = ee.Number(
        ee.Algorithms.If(
            ee.Algorithms.IsEqual(water_area, None),
            0,
            water_area,
        )
    ).getInfo()

    total_area_km2 = total_area_m2 / 1_000_000
    water_area_km2 = water_area_m2 / 1_000_000

    water_percent = (
        (water_area_m2 / total_area_m2) * 100
        if total_area_m2 > 0
        else 0
    )

    valid_area_km2 = total_area_km2 - water_area_km2

    return {
        "water_area_km2": round(water_area_km2, 4),
        "valid_area_km2": round(valid_area_km2, 4),
        "water_percent_valid": round(water_percent, 2),
    }
