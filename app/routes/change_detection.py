import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.report_generator import generate_report
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

    audience: str = "general"
    language: str = "en"


@router.post("/analyze/change")
def analyze_change(request: ChangeDetectionRequest):

    try:
        # Earth Engine authentication
        key_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../secrets/key.json")
        )
        if not os.path.exists(key_path):
            key_path = "./secrets/key.json"

        init_earth_engine(
            "satquery-backend@satquery-ai-507509.iam.gserviceaccount.com",
            key_path,
        )

        # Convert bounding box into Earth Engine geometry
        aoi = resolve_aoi(
            request.west,
            request.south,
            request.east,
            request.north,
        )

        # Calculate before-period NDVI
        before_result = get_ndvi_for_period(
            aoi,
            request.before_start,
            request.before_end,
        )

        before_ndvi = before_result["mean_ndvi"]
        before_scenes = before_result["scene_count"]

        # Calculate after-period NDVI
        after_result = get_ndvi_for_period(
            aoi,
            request.after_start,
            request.after_end,
        )

        after_ndvi = after_result["mean_ndvi"]
        after_scenes = after_result["scene_count"]

        # Calculate percentage change
        change_result = compute_change(
            before_ndvi,
            after_ndvi,
        )

        change_percent = change_result.get("pct_change")
        classification = change_result.get("direction", "undetermined")

        report = generate_report(
            change_percent,
            classification,
            request.audience,
            request.language,           
        )

        before_stats = {
            "mean_ndvi": round(before_ndvi, 4) if before_ndvi is not None else None,
            "scene_count": before_scenes,
        }
        after_stats = {
            "mean_ndvi": round(after_ndvi, 4) if after_ndvi is not None else None,
            "scene_count": after_scenes,
        }

        return {
            "analysis": {
                "before": before_stats,
                "after": after_stats,
                "change_percent": change_percent,
                "classification": classification,
                "report": report,
                "note": change_result.get("note"),
            },
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