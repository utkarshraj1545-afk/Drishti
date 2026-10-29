
"""
NDVI and MNDWI Pipeline for DRISHTI / PRAKRITI
==============================================

Deterministic Sentinel-2 analysis using Google Earth Engine.

Features:
- Sentinel-2 Surface Reflectance Harmonized imagery
- Per-pixel SCL cloud, shadow, cirrus, and snow masking
- Median image compositing
- NDVI calculation using Sentinel-2 B8 and B4
- Mean NDVI and valid-pixel coverage statistics
- Low-coverage warnings and empty-data validation
- Percentage change and vegetation-change classification
- MNDWI-based estimated water-like area using B3 and B11
"""

import math
import ee


# Configurable project threshold, not a universal scientific standard.
COVERAGE_WARNING_THRESHOLD = 70.0


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_earth_engine(service_account_email: str, key_path: str):
    """Authenticate and initialize Earth Engine using a service account."""
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
    """Convert a longitude/latitude bounding box into an EE rectangle."""
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
    Mask invalid and potentially cloud-contaminated pixels using SCL.

    Excluded Sentinel-2 SCL classes:
        0  = No data
        1  = Saturated or defective pixels
        3  = Cloud shadows
        7  = Low-probability clouds / unclassified pixels
        8  = Clouds, medium probability
        9  = Clouds, high probability
        10 = Cirrus
        11 = Snow or ice

    Class 7 is excluded conservatively because it may contain
    ambiguous pixels. This can reduce valid-pixel coverage.
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

    Filters scenes by overall cloud percentage and then applies
    the per-pixel SCL mask. Scene cloud percentage does not necessarily
    represent the cloud percentage inside the selected AOI.
    """
    if not 0 <= max_cloud_pct <= 100:
        raise ValueError("max_cloud_pct must be between 0 and 100.")

    if start_date >= end_date:
        raise ValueError("Start date must be earlier than end date.")

    return (
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


# ---------------------------------------------------------------------------
# Step 3: Build median composite
# ---------------------------------------------------------------------------

def build_composite(collection: ee.ImageCollection) -> ee.Image:
    """Build a per-pixel median composite from the masked collection."""
    return collection.median()


# ---------------------------------------------------------------------------
# Step 4: Compute NDVI
# ---------------------------------------------------------------------------

def compute_ndvi_image(composite: ee.Image) -> ee.Image:
    """
    Calculate NDVI = (B8 - B4) / (B8 + B4).

    B8 = Near-infrared
    B4 = Red
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
    """Calculate mean NDVI over the AOI using valid pixels only."""
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
    Calculate valid NDVI coverage over the sampled AOI.

    Coverage is calculated from valid pixel area divided by
    total sampled pixel area. Pixel counts are rounded to avoid
    reporting fractional pixel counts caused by boundary weighting.
    """
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")

    projection = ndvi_image.projection()

    # Count the sampled grid pixels inside the AOI.
    total_pixels = (
        ee.Image.constant(1)
        .rename("total_pixels")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    valid_pixels = (
        ndvi_image.mask()
        .gt(0)
        .unmask(0, False)
        .rename("valid_pixels")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    # Measure total sampled area and the area with valid NDVI.
    total_area = (
        ee.Image.pixelArea()
        .rename("total_area")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    valid_area = (
        ee.Image.pixelArea()
        .updateMask(ndvi_image.mask().gt(0))
        .rename("valid_area")
        .setDefaultProjection(projection)
        .clip(aoi)
    )

    # Calculate all four measurements in one Earth Engine request.
    result = (
        total_pixels
        .addBands(valid_pixels)
        .addBands(total_area)
        .addBands(valid_area)
        .reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=aoi,
            scale=scale,
            crs=projection,
            maxPixels=1e9,
            tileScale=4,
        )
        .getInfo()
    )

    total_value = result.get("total_pixels")
    valid_value = result.get("valid_pixels")
    total_area_value = result.get("total_area")
    valid_area_value = result.get("valid_area")

    if total_value is None or valid_value is None:
        raise ValueError(
            "Could not calculate pixel coverage for this AOI. "
            "Check the coordinates and satellite imagery."
        )

    if total_area_value is None or total_area_value <= 0:
        raise ValueError(
            "Could not calculate the sampled AOI area."
        )

    if valid_area_value is None:
        valid_area_value = 0

    total = int(round(total_value))
    valid = int(round(valid_value))

    if total <= 0:
        raise ValueError(
            "No sampled pixels were found inside the AOI."
        )

    valid = min(max(valid, 0), total)

    # Use areas for the coverage percentage, not rounded counts.
    coverage_percent = (
        valid_area_value / total_area_value
    ) * 100.0

    coverage_percent = min(
        max(coverage_percent, 0.0),
        100.0,
    )

    return {
        "valid_pixel_count": valid,
        "total_pixel_count": total,
        "coverage_percent": round(coverage_percent, 2),
    }



# ---------------------------------------------------------------------------
# Step 7: Estimate water extent using MNDWI
# ---------------------------------------------------------------------------

def get_mndwi_water_extent(
    composite: ee.Image,
    aoi: ee.Geometry,
    scale: int = 20,
) -> dict:
    """
    Estimate water-like area using MNDWI from an existing composite.

    MNDWI = (Green - SWIR) / (Green + SWIR)
          = (B3 - B11) / (B3 + B11)

    Sentinel-2 B3 is Green (10 m); B11 is SWIR (20 m).
    A 20 m scale is used to match B11's native resolution.

    MNDWI > 0 is an initial screening heuristic, not a definitive
    water classification. Results may be affected by season, mixed
    pixels, cloud masking, shadows, and image availability.

    Returns:
        water_area_km2: estimated water-like area among valid pixels
        valid_area_km2: area with valid MNDWI data
        water_percent_valid: water-like area as a percentage of
                             valid MNDWI area

    If there are no valid MNDWI pixels, areas are reported as unavailable
    rather than treating the entire AOI as non-water.
    """
    if scale <= 0:
        raise ValueError("scale must be greater than zero.")

    mndwi = composite.normalizedDifference(
        ["B3", "B11"]
    ).rename("MNDWI")

    pixel_area = ee.Image.pixelArea()

    valid_area_image = (
        pixel_area
        .updateMask(mndwi.mask())
        .rename("area")
    )

    water_area_image = (
        pixel_area
        .updateMask(mndwi.gt(0))
        .rename("area")
    )

    reduce_args = {
        "geometry": aoi,
        "scale": scale,
        "maxPixels": 1e9,
        "tileScale": 4,
    }

    valid_result = valid_area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        **reduce_args,
    ).getInfo()

    water_result = water_area_image.reduceRegion(
        reducer=ee.Reducer.sum(),
        **reduce_args,
    ).getInfo()

    valid_area_m2 = valid_result.get("area")
    water_area_m2 = water_result.get("area")

    if (
    valid_area_m2 is None
    or not isinstance(valid_area_m2, (int, float))
    or valid_area_m2 <= 0
):
        return {
            "water_area_km2": None,
            "valid_area_km2": 0.0,
            "water_percent_valid": None,
        }

    # A missing water sum means no pixels passed the threshold.
    water_area_m2 = water_area_m2 or 0.0

    # Guard against small numerical discrepancies.
    water_area_m2 = min(water_area_m2, valid_area_m2)

    return {
        "water_area_km2": round(water_area_m2 / 1_000_000, 6),
        "valid_area_km2": round(valid_area_m2 / 1_000_000, 6),
        "water_percent_valid": round(
            water_area_m2 / valid_area_m2 * 100.0,
            2,
        ),
    }


# ---------------------------------------------------------------------------
# Steps 2-8: Analyze one date period
# ---------------------------------------------------------------------------

def get_ndvi_for_period(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 20,
) -> dict:
    """
    Analyze one date period and return NDVI, composite, scene count,
    and valid-pixel coverage.

    Returns:
        mean_ndvi
        composite_image
        scene_count
        valid_pixel_count
        total_pixel_count
        coverage_percent
        coverage_warning

    The composite_image is returned so other indices, including MNDWI,
    can reuse the same imagery without fetching the collection again.
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
# Step 9: Compute change between two periods
# ---------------------------------------------------------------------------

def compute_change(
    before_ndvi: float,
    after_ndvi: float,
) -> dict:
    """
    Compare mean NDVI between two periods.

    Percentage change:
        ((after_ndvi - before_ndvi) / before_ndvi) * 100

    Negative change means NDVI decreased.
    Positive change means NDVI increased.

    NDVI change alone does not prove flooding, vegetation damage,
    or whether an area is safe.
    """
    if (
        not math.isfinite(before_ndvi)
        or not math.isfinite(after_ndvi)
    ):
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
