
from io import BytesIO
from datetime import date
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.ndvi_pipeline import (
    init_earth_engine,
    resolve_aoi,
    get_ndvi_for_period,
    compute_change,
    COVERAGE_WARNING_THRESHOLD,
)

router = APIRouter()

# Configure these with environment variables. Keep credential files outside
# version control and never include them in shared ZIP files.
SERVICE_ACCOUNT = os.getenv(
    "EARTH_ENGINE_SERVICE_ACCOUNT",
    "satquery-backend@satquery-ai-507509.iam.gserviceaccount.com",
)
_DEFAULT_KEY_PATH = Path(__file__).resolve().parents[2] / "secrets" / "key.json"
SERVICE_ACCOUNT_KEY = os.getenv("EARTH_ENGINE_KEY_PATH", str(_DEFAULT_KEY_PATH))


class ChangeDetectionRequest(BaseModel):
    west: float
    south: float
    east: float
    north: float

    before_start: str
    before_end: str

    after_start: str
    after_end: str

    language: str = "en"
    max_cloud_pct: int = Field(default=80, ge=1, le=100)


def run_analysis(request: ChangeDetectionRequest) -> dict:
    """Run the shared analysis used by both API endpoints."""

    if request.west >= request.east:
        raise ValueError("West longitude must be less than east longitude.")

    if request.south >= request.north:
        raise ValueError("South latitude must be less than north latitude.")

    if not (
        -180 <= request.west <= 180
        and -180 <= request.east <= 180
        and -90 <= request.south <= 90
        and -90 <= request.north <= 90
    ):
        raise ValueError("The supplied coordinates are outside valid ranges.")

    # Require real ISO dates rather than relying on lexicographic string comparison.
    date_fields = {
        "before_start": request.before_start,
        "before_end": request.before_end,
        "after_start": request.after_start,
        "after_end": request.after_end,
    }
    parsed_dates = {}
    for field_name, value in date_fields.items():
        try:
            parsed_dates[field_name] = date.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field_name} must be a valid date in YYYY-MM-DD format."
            ) from exc

    if parsed_dates["before_start"] > parsed_dates["before_end"]:
        raise ValueError("Before-period start date must not follow its end date.")

    if parsed_dates["after_start"] > parsed_dates["after_end"]:
        raise ValueError("After-period start date must not follow its end date.")

    key_path = Path(SERVICE_ACCOUNT_KEY).expanduser()
    if not key_path.is_file():
        raise RuntimeError(
            "Google Earth Engine service-account key was not found. "
            "Set EARTH_ENGINE_KEY_PATH to the path of your local key JSON file. "
            "Keep that file private and outside shared ZIP files."
        )

    init_earth_engine(SERVICE_ACCOUNT, str(key_path))

    aoi = resolve_aoi(
        request.west,
        request.south,
        request.east,
        request.north,
    )

    before_result = get_ndvi_for_period(
        aoi,
        request.before_start,
        request.before_end,
        max_cloud_pct=request.max_cloud_pct,
    )

    after_result = get_ndvi_for_period(
        aoi,
        request.after_start,
        request.after_end,
        max_cloud_pct=request.max_cloud_pct,
    )

    change_result = compute_change(
        before_result["mean_ndvi"],
        after_result["mean_ndvi"],
    )

    def period_payload(period: dict) -> dict:
        return {
            "mean_ndvi": period["mean_ndvi"],
            "scenes": period["scene_count"],
            "valid_pixel_count": period["valid_pixel_count"],
            "total_pixel_count": period["total_pixel_count"],
            "coverage_percent": period["coverage_percent"],
            "coverage_warning": period["coverage_warning"],
        }

    before = period_payload(before_result)
    after = period_payload(after_result)

    coverage_warnings = []

    if before["coverage_warning"]:
        coverage_warnings.append(
            f"Before period: {before['coverage_warning']}"
        )

    if after["coverage_warning"]:
        coverage_warnings.append(
            f"After period: {after['coverage_warning']}"
        )

    low_coverage = (
        before["coverage_percent"] < COVERAGE_WARNING_THRESHOLD
        or after["coverage_percent"] < COVERAGE_WARNING_THRESHOLD
    )

    original_classification = change_result["direction"]

    if low_coverage and original_classification:
        classification = (
            f"Provisional result — {original_classification}; "
            "low valid-pixel coverage"
        )
        note = (
            "This comparison has low valid-pixel coverage in at least one "
            "period. Treat the apparent NDVI change as provisional. Seasonal "
            "variation, cloud masking, and image availability may affect the "
            "result. NDVI change alone does not confirm vegetation damage "
            "or flooding."
        )
    else:
        classification = original_classification
        note = (
            "NDVI change indicates a change in vegetation greenness. It does "
            "not independently confirm flooding, vegetation damage, or "
            "whether an area is safe. Seasonal differences and image "
            "availability can affect the results."
        )

    return {
        "before": before,
        "after": after,
        "change_percent": change_result["pct_change"],
        "classification": classification,
        "note": note,
        "max_cloud_pct": request.max_cloud_pct,
        "coverage_warning_threshold": COVERAGE_WARNING_THRESHOLD,
        "coverage_warnings": coverage_warnings,
    }


def handle_analysis_error(error: Exception) -> None:
    """Convert analysis failures into suitable HTTP responses."""

    if isinstance(error, HTTPException):
        raise error

    if isinstance(error, ValueError):
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    print(f"DRISHTI analysis error: {error}")

    raise HTTPException(
        status_code=500,
        detail="Analysis failed. Check the backend logs for details.",
    ) from error


@router.post("/analyze/change")
def analyze_change(request: ChangeDetectionRequest):
    try:
        return run_analysis(request)
    except Exception as error:
        handle_analysis_error(error)


@router.post("/analyze/change/pdf")
def analyze_change_pdf(request: ChangeDetectionRequest):
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle,
        )
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            KeepTogether,
        )

        result = run_analysis(request)
        buffer = BytesIO()

        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.55 * inch,
            leftMargin=0.55 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title="DRISHTI NDVI Change Report",
            author="DRISHTI",
        )

        styles = getSampleStyleSheet()
        styles["Title"].textColor = colors.HexColor("#172554")
        styles["Title"].alignment = TA_CENTER
        styles["Title"].spaceAfter = 5

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=16,
        )

        warning_style = ParagraphStyle(
            "CoverageWarning",
            parent=styles["Normal"],
            textColor=colors.HexColor("#92400e"),
            leading=14,
        )

        interpretation_style = ParagraphStyle(
            "Interpretation",
            parent=styles["Italic"],
            leading=14,
            spaceBefore=3,
        )

        story = [
            Paragraph(
                "DRISHTI — NDVI Change Analysis",
                styles["Title"],
            ),
            Paragraph(
                "Satellite-based vegetation comparison report",
                subtitle_style,
            ),
        ]

        def add_table(data, widths):
            table = Table(data, colWidths=widths, repeatRows=1)
            table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, 0),
                        colors.HexColor("#172554"),
                    ),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    (
                        "GRID",
                        (0, 0),
                        (-1, -1),
                        0.4,
                        colors.HexColor("#cbd5e1"),
                    ),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f8fafc")],
                    ),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ])
            )
            return table

        # Area of interest
        story.append(
            Paragraph("Area of Interest", styles["Heading2"])
        )

        aoi_data = [
            ["Coordinate", "Value"],
            ["West longitude", f"{request.west:.6f}"],
            ["South latitude", f"{request.south:.6f}"],
            ["East longitude", f"{request.east:.6f}"],
            ["North latitude", f"{request.north:.6f}"],
        ]

        story.append(
            add_table(aoi_data, [2.2 * inch, 3.5 * inch])
        )
        story.append(Spacer(1, 16))

        # Period comparison
        before = result["before"]
        after = result["after"]

        story.append(
            Paragraph("Comparison Periods", styles["Heading2"])
        )

        period_data = [
            ["Metric", "Before period", "After period"],
            [
                "Date range",
                f"{request.before_start} to {request.before_end}",
                f"{request.after_start} to {request.after_end}",
            ],
            [
                "Mean NDVI",
                (
                    f"{before['mean_ndvi']:.6f}"
                    if before["mean_ndvi"] is not None
                    else "Not available"
                ),
                (
                    f"{after['mean_ndvi']:.6f}"
                    if after["mean_ndvi"] is not None
                    else "Not available"
                ),
            ],
            [
                "Satellite scenes",
                str(before["scenes"]),
                str(after["scenes"]),
            ],
            [
                "Valid pixels",
                f"{before['valid_pixel_count']:,}",
                f"{after['valid_pixel_count']:,}",
            ],
            [
                "Total sampled pixels",
                f"{before['total_pixel_count']:,}",
                f"{after['total_pixel_count']:,}",
            ],
            [
                "Valid-pixel coverage",
                f"{before['coverage_percent']:.2f}%",
                f"{after['coverage_percent']:.2f}%",
            ],
        ]

        story.append(
            add_table(
                period_data,
                [1.75 * inch, 2.0 * inch, 2.0 * inch],
            )
        )
        story.append(Spacer(1, 14))

        # Highlight coverage warnings
        if result["coverage_warnings"]:
            warning_items = [
                Paragraph(
                    "Valid-Pixel Coverage Warnings",
                    styles["Heading2"],
                )
            ]

            for warning in result["coverage_warnings"]:
                warning_items.append(
                    Paragraph(warning, warning_style)
                )
                warning_items.append(Spacer(1, 4))

            warning_table = Table(
                [[warning_items]],
                colWidths=[5.7 * inch],
            )
            warning_table.setStyle(
                TableStyle([
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        colors.HexColor("#fffbeb"),
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.8,
                        colors.HexColor("#f59e0b"),
                    ),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ])
            )

            story.append(
                KeepTogether([warning_table, Spacer(1, 10)])
            )

        story.append(
            Paragraph(
                "Coverage is the percentage of sampled AOI pixels with "
                "usable NDVI after masking. The warning threshold is "
                f"{COVERAGE_WARNING_THRESHOLD:.0f}%. Coverage alone does "
                "not guarantee imagery quality.",
                styles["Italic"],
            )
        )
        story.append(Spacer(1, 14))

        # Analysis results
        story.append(
            Paragraph("Analysis Results", styles["Heading2"])
        )

        change = result["change_percent"]
        change_text = (
            f"{change:+.1f}%"
            if isinstance(change, (int, float))
            else "Not available"
        )

        story.append(
            Paragraph(f"NDVI change: {change_text}", styles["Normal"])
        )
        story.append(
            Paragraph(
                f"Classification: {result['classification']}",
                styles["Normal"],
            )
        )

        # One concise interpretation paragraph, built from actual values.
        story.append(Spacer(1, 8))

        after_coverage = after["coverage_percent"]
        coverage_threshold = result["coverage_warning_threshold"]

        if isinstance(change, (int, float)):
            if change < 0:
                change_description = (
                    f"NDVI decreased by {abs(change):.1f}%"
                )
            elif change > 0:
                change_description = (
                    f"NDVI increased by {change:.1f}%"
                )
            else:
                change_description = "NDVI showed no relative change"
        else:
            change_description = "NDVI change could not be calculated"

        if isinstance(after_coverage, (int, float)):
            if after_coverage < coverage_threshold:
                coverage_sentence = (
                    f"However, only {after_coverage:.2f}% of sampled "
                    "pixels were usable in the after period, below the "
                    f"{coverage_threshold:.0f}% coverage threshold. "
                )
            else:
                coverage_sentence = (
                    f"Valid-pixel coverage in the after period was "
                    f"{after_coverage:.2f}%. "
                )
        else:
            coverage_sentence = (
                "After-period coverage was not available. "
            )

        low_coverage = bool(result["coverage_warnings"])

        if low_coverage:
            interpretation = (
                f"Interpretation: {change_description} between the two "
                "periods. "
                f"{coverage_sentence}"
                "Treat this comparison as provisional. Seasonal variation, "
                "cloud masking, and image availability may affect the result. "
                "NDVI change alone does not confirm vegetation damage "
                "or flooding."
            )
        else:
            interpretation = (
                f"Interpretation: {change_description} between the two "
                "periods. "
                f"{coverage_sentence}"
                "Seasonal variation, cloud masking, and image availability "
                "may affect the result. NDVI change alone does not confirm "
                "vegetation damage or flooding."
            )

        story.append(
            Paragraph(
                interpretation,
                interpretation_style,
            )
        )

        document.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": (
                    'attachment; filename="drishti-ndvi-report.pdf"'
                )
            },
        )

    except Exception as error:
        handle_analysis_error(error)
