"""
NDVI Pipeline for SatQuery AI
==============================
Core deterministic remote-sensing module. Every function here produces a
number or image derived directly from satellite band math -- nothing here
is estimated by an AI model. This is the "ground truth" layer that the
VLM will later explain in plain language.

Maps to the 8-step pipeline:
  1. resolve_aoi          - turn a bbox into an ee.Geometry
  2. get_cloud_filtered_collection - fetch valid scenes for a date range
  3. build_composite       - median-composite the collection into one image
  4. compute_ndvi_image    - band math: (NIR - Red) / (NIR + Red)
  5. get_mean_ndvi         - reduce the NDVI image to a single number
  6/7. get_ndvi_for_period - steps 2-5 combined for one date window
  8. compute_change        - compare two periods, get % vegetation change
"""
import ee


def init_earth_engine(service_account_email: str, key_path: str):
    """
    Call once at backend startup. Authenticates using the service account
    JSON key (never the personal browser login -- that only works for
    the Code Editor, not a server process).
    """
    credentials = ee.ServiceAccountCredentials(service_account_email, key_path)
    ee.Initialize(credentials)


# ---------------------------------------------------------------------------
# Step 1: Resolve AOI
# ---------------------------------------------------------------------------
def resolve_aoi(west: float, south: float, east: float, north: float) -> ee.Geometry:
    """
    Converts a bounding box into an ee.Geometry.Rectangle.

    Coordinates are in standard lon/lat order (NOT lat/lon -- this trips
    people up constantly). West/east are longitude, south/north are latitude.

    Example: Delhi region
        resolve_aoi(west=76.8, south=28.4, east=77.4, north=28.9)
    """
    return ee.Geometry.Rectangle([west, south, east, north])


# ---------------------------------------------------------------------------
# Step 2: Cloud-filtered collection
# ---------------------------------------------------------------------------
def get_cloud_filtered_collection(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 20,
) -> ee.ImageCollection:
    """
    Returns every Sentinel-2 scene covering the AOI within the date range,
    filtered to only scenes where cloud cover is below max_cloud_pct.

    We fetch a COLLECTION, not a single image, because any one satellite
    pass over an AOI is likely to have some cloud cover. This is what lets
    step 3 (compositing) smooth that out.

    dates as 'YYYY-MM-DD' strings.
    """
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud_pct))
    )
    return collection


# ---------------------------------------------------------------------------
# Step 3: Build composite
# ---------------------------------------------------------------------------
def build_composite(collection: ee.ImageCollection) -> ee.Image:
    """
    Merges a collection of images into ONE image using a per-pixel median.

    Why median, not average? If a handful of images in the collection still
    have leftover cloud/shadow (our cloud filter isn't perfect), those pixels
    become extreme outliers. Median is far less sensitive to outliers than
    mean -- it picks the "middle" value per pixel across the whole stack,
    so a few bad frames get outvoted rather than skewing the result.

    Returns a single ee.Image with all the original bands (B1-B12 etc.)
    still available -- we haven't touched NDVI yet, that's step 4.
    """
    return collection.median()


# ---------------------------------------------------------------------------
# Step 4: Compute NDVI
# ---------------------------------------------------------------------------
def compute_ndvi_image(composite: ee.Image) -> ee.Image:
    """
    NDVI = (NIR - Red) / (NIR + Red)

    For Sentinel-2:
        B8 = Near-Infrared (NIR)
        B4 = Red

    Why this formula works (the physics, not just math):
    Chlorophyll in healthy leaves strongly ABSORBS red light (for
    photosynthesis) but the leaf's internal cell structure strongly
    REFLECTS near-infrared light. So healthy vegetation = high NIR,
    low Red -> NDVI close to +1.

    Bare soil, roads, and buildings reflect Red and NIR fairly similarly
    -> NDVI close to 0.

    Water absorbs NIR strongly -> NDVI is negative.

    normalizedDifference() is Earth Engine's built-in helper -- you give
    it [band1, band2] and it computes (band1 - band2) / (band1 + band2)
    for every pixel. We rename the output band to "NDVI" so later
    functions can refer to it by name.
    """
    ndvi = composite.normalizedDifference(["B8", "B4"]).rename("NDVI")
    return ndvi


# ---------------------------------------------------------------------------
# Step 5: Reduce to a single number
# ---------------------------------------------------------------------------
def get_mean_ndvi(ndvi_image: ee.Image, aoi: ee.Geometry, scale: int = 10) -> float:
    """
    compute_ndvi_image() gives you an NDVI value PER PIXEL across the AOI --
    still an image, not a number. This function averages all those pixel
    values into one headline statistic: "the mean NDVI across this region."

    reduceRegion() is Earth Engine's tool for collapsing an image down to
    statistics over an area. We use Reducer.mean() to get the average.

    `scale=10` matters: it tells Earth Engine what pixel resolution to
    sample at, in meters. Sentinel-2's B4/B8 bands are natively 10m
    resolution -- using scale=10 means we sample at native resolution.
    Using a coarser scale (e.g. 100) would average over bigger blocks and
    give a less precise (though faster to compute) result.

    maxPixels is set high because reduceRegion refuses to run over very
    large areas by default, as a safety limit against runaway compute cost.

    .getInfo() is what actually pulls the result from Earth Engine's
    servers back into a normal Python value -- everything before this
    point is just "instructions," not yet executed. Earth Engine is lazy:
    it doesn't compute anything until you explicitly ask for the result.
    """
    stats = ndvi_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=scale,
        maxPixels=1e9,
    )
    return stats.get("NDVI").getInfo()


# ---------------------------------------------------------------------------
# Steps 2-5 combined: everything needed for ONE date window
# ---------------------------------------------------------------------------
def get_ndvi_for_period(
    aoi: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud_pct: int = 20,
) -> dict:
    """
    Runs the full pipeline (Functions 3, 4, 5, 6) for a single date window
    and returns everything you need for that period in one call:
      - the mean NDVI value
      - the composite image itself (so a thumbnail can be generated later)
      - how many scenes went into the composite (useful for transparency --
        "this result is based on N cloud-free scenes")

    Raises ValueError if no usable imagery was found, rather than silently
    returning a broken/empty result. This is the honest-failure-handling
    we talked about -- better to tell the user "widen your date range"
    than to hand back a meaningless number.
    """
    collection = get_cloud_filtered_collection(aoi, start_date, end_date, max_cloud_pct)

    scene_count = collection.size().getInfo()
    if scene_count == 0:
        raise ValueError(
            f"No cloud-free Sentinel-2 imagery found for this AOI between "
            f"{start_date} and {end_date} (cloud threshold: {max_cloud_pct}%). "
            f"Try widening the date range or raising max_cloud_pct."
        )

    composite = build_composite(collection)
    ndvi_image = compute_ndvi_image(composite)
    mean_ndvi = get_mean_ndvi(ndvi_image, aoi)

    return {
        "mean_ndvi": mean_ndvi,
        "composite_image": composite,
        "scene_count": scene_count,
    }


# ---------------------------------------------------------------------------
# Step 8: Compute change between two periods
# ---------------------------------------------------------------------------
def compute_change(before_ndvi: float, after_ndvi: float) -> dict:
    """
    Takes two mean NDVI values (from two calls to get_ndvi_for_period)
    and computes the percentage change between them.

    % change = (after - before) / before * 100

    Negative result = vegetation LOSS (NDVI dropped -- e.g. deforestation)
    Positive result = vegetation GAIN (e.g. regrowth, new crop cycle)

    We guard against before_ndvi == 0 (or very close to it) because
    dividing by a near-zero baseline produces a meaningless, wildly
    exaggerated percentage -- an honest edge case to handle rather than
    silently returning garbage like "38000% change."
    """
    if abs(before_ndvi) < 1e-6:
        return {
            "pct_change": None,
            "direction": "undetermined",
            "note": (
                "Baseline NDVI is too close to zero (likely bare ground, "
                "water, or urban area) -- percentage change isn't a "
                "meaningful metric here. Report absolute NDVI values instead."
            ),
        }

    pct_change = ((after_ndvi - before_ndvi) / before_ndvi) * 100

    if pct_change < -5:
        direction = "vegetation loss"
    elif pct_change > 5:
        direction = "vegetation gain"
    else:
        direction = "no significant change"

    return {
        "pct_change": round(pct_change, 1),
        "direction": direction,
        "note": None,
    }
