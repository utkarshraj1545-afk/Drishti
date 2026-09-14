"""
Test script: run the full NDVI pipeline against a real, documented
deforestation site to sanity-check the whole chain end-to-end.

Site: Rondonia, Brazil -- a well-known deforestation region along the
BR-364 highway corridor, showing the characteristic "fishbone" clearing
pattern. Chosen because it's well-documented and independently verifiable,
not because we're cherry-picking a fake result.

HOW TO RUN:
  1. pip install earthengine-api
  2. Place your service account JSON key somewhere local, e.g. ./secrets/key.json
  3. Fill in SERVICE_ACCOUNT_EMAIL and KEY_PATH below
  4. python test_ndvi_pipeline.py
"""
from app.services.ndvi_pipeline import (
    init_earth_engine,
    resolve_aoi,
    get_ndvi_for_period,
    compute_change,
)

# --- Fill these in with YOUR values ---
SERVICE_ACCOUNT_EMAIL = "satquery-backend@satquery-ai-507509.iam.gserviceaccount.com"
KEY_PATH = "./secrets/key.json"  # path to your (regenerated, local-only) key file
AOI_BOUNDS = dict(west=77.87, south=30.15, east=77.93, north=30.21)

BEFORE_RANGE = ("2018-01-01", "2018-03-01")
AFTER_RANGE = ("2024-01-01", "2024-03-01")

def main():
    print("Initializing Earth Engine...")
    init_earth_engine(SERVICE_ACCOUNT_EMAIL, KEY_PATH)

    aoi = resolve_aoi(**AOI_BOUNDS)

    print(f"\nFetching 'before' period: {BEFORE_RANGE}")
    before = get_ndvi_for_period(aoi, *BEFORE_RANGE)
    print(f"  Mean NDVI: {before['mean_ndvi']:.4f}  (from {before['scene_count']} scenes)")

    print(f"\nFetching 'after' period: {AFTER_RANGE}")
    after = get_ndvi_for_period(aoi, *AFTER_RANGE)
    print(f"  Mean NDVI: {after['mean_ndvi']:.4f}  (from {after['scene_count']} scenes)")

    change = compute_change(before["mean_ndvi"], after["mean_ndvi"])

    print("\n--- RESULT ---")
    print(f"Before NDVI: {before['mean_ndvi']:.4f}")
    print(f"After NDVI:  {after['mean_ndvi']:.4f}")
    print(f"Change:      {change['pct_change']}%  ({change['direction']})")
    if change["note"]:
        print(f"Note: {change['note']}")


if __name__ == "__main__":
    main()
