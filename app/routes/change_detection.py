from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.ndvi_pipeline import (
    init_earth_engine,
    resolve_aoi,
    get_ndvi_for_period,
    compute_change,
)


router = APIRouter()


class ChangeDetectionRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float

    before_start: str
    before_end: str

    after_start: str
    after_end: str


@router.post("/analyze/change")
def analyze_change(request: ChangeDetectionRequest):

    try:
        # Earth Engine authentication
        init_earth_engine(
            "satquery-backend@satquery-ai-507509.iam.gserviceaccount.com",
            "./secrets/key.json",
        )

        # Convert bounding box into Earth Engine geometry
        aoi = resolve_aoi(
            request.west,
            request.south,
            request.east,
            request.north,
        )

        # Calculate before-period NDVI
        before_ndvi, before_scenes = get_ndvi_for_period(
            aoi,
            request.before_start,
            request.before_end,
        )

        # Calculate after-period NDVI
        after_ndvi, after_scenes = get_ndvi_for_period(
            aoi,
            request.after_start,
            request.after_end,
        )

        # Calculate percentage change
        change_percent, classification = compute_change(
            before_ndvi,
            after_ndvi,
        )

        return {
            "before": {
                "mean_ndvi": before_ndvi,
                "scenes": before_scenes,
            },
            "after": {
                "mean_ndvi": after_ndvi,
                "scenes": after_scenes,
            },
            "change_percent": change_percent,
            "classification": classification,
        }

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}",
        )