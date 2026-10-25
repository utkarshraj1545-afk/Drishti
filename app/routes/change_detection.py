
from io import BytesIO
from datetime import date
import os
from pathlib import Path
from html import escape

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.services.ndvi_pipeline import (
    init_earth_engine,
    resolve_aoi,
    get_ndvi_for_period,
    compute_change,
    get_mndwi_water_extent,
    COVERAGE_WARNING_THRESHOLD,
)

router = APIRouter()

SERVICE_ACCOUNT = os.getenv(
    "EARTH_ENGINE_SERVICE_ACCOUNT",
    "satquery-backend@satquery-ai-507509.iam.gserviceaccount.com",
)

_DEFAULT_KEY_PATH = (
    Path(__file__).resolve().parents[2] / "secrets" / "key.json"
)

SERVICE_ACCOUNT_KEY = os.getenv(
    "EARTH_ENGINE_KEY_PATH",
    str(_DEFAULT_KEY_PATH),
)


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


def is_hindi(request: ChangeDetectionRequest) -> bool:
    """Check whether the request asks for a Hindi report."""
    language = str(request.language).strip().lower().replace("_", "-")
    return language in {"hi", "hindi"} or language.startswith("hi-")
def is_hindi_labels(labels: dict) -> bool:
    """Detect Hindi labels from the translated report text."""
    return any(
        any("\u0900" <= char <= "\u097f" for char in str(value))
        for value in labels.values()
    )


def report_labels(request: ChangeDetectionRequest) -> dict:
    """Return report labels in the selected language."""

    if is_hindi(request):
        return {
            "title": "PRAKRITI — NDVI परिवर्तन विश्लेषण",
            "subtitle": "उपग्रह आधारित वनस्पति तुलना रिपोर्ट",
            "aoi": "अध्ययन क्षेत्र",
            "coordinate": "निर्देशांक",
            "value": "मान",
            "west": "पश्चिमी देशांतर",
            "south": "दक्षिणी अक्षांश",
            "east": "पूर्वी देशांतर",
            "north": "उत्तरी अक्षांश",
            "periods": "तुलना की अवधि",
            "metric": "मापदंड",
            "before": "पहले की अवधि",
            "after": "बाद की अवधि",
            "date_range": "तिथि सीमा",
            "date_to": "से",
            "mean_ndvi": "औसत NDVI",
            "scenes": "उपग्रह चित्रों की संख्या",
            "valid_pixels": "उपयोग योग्य पिक्सेल",
            "total_pixels": "कुल नमूना पिक्सेल",
            "coverage": "उपयोग योग्य पिक्सेल का प्रतिशत",
            "warnings": "डेटा कवरेज संबंधी चेतावनी",
            "coverage_explanation": (
                 "यह प्रतिशत बताता है कि अध्ययन क्षेत्र के कितने "
                 "पिक्सेल में उपयोग योग्य NDVI डेटा उपलब्ध था। "
                 "अधिक कवरेज से पूरे क्षेत्र का आकलन बेहतर होता है।"
             ),
            "results": "विश्लेषण के परिणाम",
            "change": "NDVI में परिवर्तन",
            "classification": "वर्गीकरण",
            "not_available": "उपलब्ध नहीं",
            "decreased": "NDVI में कमी",
            "increased": "NDVI में वृद्धि",
            "no_change": "NDVI में कोई सापेक्ष परिवर्तन नहीं",
            "could_not_calculate": "NDVI परिवर्तन की गणना नहीं हो सकी",
            "interpretation": "व्याख्या",
            "provisional": "इस तुलना को प्रारंभिक मानें।",
            "caution": (
                "मौसम, बादल और उपग्रह चित्रों की उपलब्धता परिणामों "
                "को प्रभावित कर सकती है। NDVI में बदलाव वनस्पति की "
                "हरियाली में बदलाव का संकेत देता है, लेकिन इससे "
                "अकेले वनस्पति क्षति या किसी क्षेत्र की सुरक्षा "
                "की पुष्टि नहीं की जा सकती।"
            ),
            "before_warning": "पहले की अवधि",
            "after_warning": "बाद की अवधि",
            "provisional_classification": "प्रारंभिक परिणाम",
            "low_coverage": "कम उपयोग योग्य पिक्सेल कवरेज",
            "ndvi_decreased_by": "NDVI में कमी",
            "ndvi_increased_by": "NDVI में वृद्धि",
            "coverage_sentence_low": "हालाँकि, बाद की अवधि में केवल",
            "coverage_sentence_low_end": (
                "नमूना पिक्सेल उपयोग योग्य थे, जो निर्धारित सीमा से कम है।"
            ),
            "coverage_sentence_ok": (
                "बाद की अवधि में उपयोग योग्य पिक्सेल कवरेज"
            ),
            "coverage_unavailable": (
                "बाद की अवधि का कवरेज उपलब्ध नहीं था।"
            ),
            "threshold": "निर्धारित सीमा",
            "water_extent": "अनुमानित जल विस्तार (MNDWI)",
            "water_area": "जल जैसा क्षेत्रफल",
            "valid_area": "वैध विश्लेषण क्षेत्र",
            "water_percent": "वैध क्षेत्र का जल प्रतिशत",
            "water_not_available": "जल अनुमान उपलब्ध नहीं",
        }

    return {
        "title": "PRAKRITI — NDVI Change Analysis",
        "subtitle": "Satellite-based vegetation comparison report",
        "aoi": "Area of Interest",
        "coordinate": "Coordinate",
        "value": "Value",
        "west": "West longitude",
        "south": "South latitude",
        "east": "East longitude",
        "north": "North latitude",
        "periods": "Comparison Periods",
        "metric": "Metric",
        "before": "Before period",
        "after": "After period",
        "date_range": "Date range",
        "date_to": "to",
        "mean_ndvi": "Mean NDVI",
        "scenes": "Satellite scenes",
        "valid_pixels": "Valid pixels",
        "total_pixels": "Total sampled pixels",
        "coverage": "Valid-pixel coverage",
        "warnings": "Valid-Pixel Coverage Warnings",
        "coverage_explanation": (
            "Coverage is the percentage of sampled AOI pixels with usable "
            "NDVI after masking. Coverage alone does not guarantee imagery "
            "quality."
        ),
        "results": "Analysis Results",
        "change": "NDVI change",
        "classification": "Classification",
        "not_available": "Not available",
        "decreased": "NDVI decreased",
        "increased": "NDVI increased",
        "no_change": "NDVI showed no relative change",
        "could_not_calculate": "NDVI change could not be calculated",
        "interpretation": "Interpretation",
        "provisional": "Treat this comparison as provisional.",
        "caution": (
            "Seasonal variation, cloud masking, and image availability may "
            "affect the result. NDVI change alone does not confirm vegetation "
            "damage or flooding."
        ),
        "before_warning": "Before period",
        "after_warning": "After period",
        "provisional_classification": "Provisional result",
        "low_coverage": "low valid-pixel coverage",
        "ndvi_decreased_by": "NDVI decreased by",
        "ndvi_increased_by": "NDVI increased by",
        "coverage_sentence_low": "However, only",
        "coverage_sentence_low_end": (
            "of sampled pixels were usable in the after period, "
            "below the threshold."
        ),
        "coverage_sentence_ok": (
            "Valid-pixel coverage in the after period was"
        ),
        "coverage_unavailable": "After-period coverage was not available.",
        "threshold": "Threshold",
        "water_extent": "Estimated Water Extent (MNDWI)",
        "water_area": "Water-like area (km\u00b2)",
        "valid_area": "Valid analysed area (km\u00b2)",
        "water_percent": "Water % of valid area",
        "water_not_available": "Not available",
    }




def translate_classification(value, labels: dict):
    """Translate known classification values using the selected language."""

    hindi = is_hindi_labels(labels)

    if value is None:
        return labels["not_available"]

    # Preserve the original classification in English mode.
    if not hindi:
        return str(value)

    normalized = str(value).strip().lower().replace("_", " ").replace("-", " ")

    if normalized in {
        "vegetation gain",
        "vegetation increased",
        "vegetation increase",
        "increase",
        "increased",
        "gain",
    }:
        return labels["increased"]

    if normalized in {
        "vegetation loss",
        "vegetation decreased",
        "vegetation decrease",
        "decrease",
        "decreased",
        "loss",
    }:
        return labels["decreased"]

    if normalized in {
        "no significant change",
        "no change",
        "unchanged",
    }:
        return labels["no_change"]

    # Preserve unexpected classifications rather than mislabeling them.
    return str(value)



def translate_coverage_warning(value, request, period_label):
    """Translate known coverage-warning text while preserving unknown details."""

    if not value:
        return None

    if not is_hindi(request):
        return f"{period_label}: {value}"

    text = str(value).strip()
    normalized = text.lower()

    # Common coverage warnings used by the NDVI pipeline.
    if "coverage" in normalized and (
        "low" in normalized or "below" in normalized
    ):
        return (
            f"{period_label}: उपयोग योग्य पिक्सेल कवरेज कम है। "
            "इस अवधि के परिणाम को सावधानी से देखें।"
        )

    if "no usable" in normalized or "no valid" in normalized:
        return (
            f"{period_label}: उपयोग योग्य NDVI डेटा उपलब्ध नहीं था।"
        )

    # Preserve the source message if its exact meaning is unknown.
    return f"{period_label}: {text}"


def run_analysis(request: ChangeDetectionRequest) -> dict:
    """Run the shared analysis used by both API endpoints."""

    print(
        f"DRISHTI DEBUG: language={request.language!r}, "
        f"hindi={is_hindi(request)}, "
        f"file={Path(__file__).resolve()}"
    )

    if request.west >= request.east:
        raise ValueError(
            "West longitude must be less than east longitude."
        )

    if request.south >= request.north:
        raise ValueError(
            "South latitude must be less than north latitude."
        )

    if not (
        -180 <= request.west <= 180
        and -180 <= request.east <= 180
        and -90 <= request.south <= 90
        and -90 <= request.north <= 90
    ):
        raise ValueError(
            "The supplied coordinates are outside valid ranges."
        )

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
        raise ValueError(
            "Before-period start date must not follow its end date."
        )

    if parsed_dates["after_start"] > parsed_dates["after_end"]:
        raise ValueError(
            "After-period start date must not follow its end date."
        )

    key_path = Path(SERVICE_ACCOUNT_KEY).expanduser()

    if not key_path.is_file():
        raise RuntimeError(
            "Google Earth Engine service-account key was not found. "
            "Set EARTH_ENGINE_KEY_PATH to the path of your local key "
            "JSON file. Keep that file private and outside shared ZIP files."
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
    labels = report_labels(request)
    hindi = is_hindi(request)

    coverage_warnings = []

    before_warning = translate_coverage_warning(
        before["coverage_warning"],
        request,
        labels["before_warning"],
    )
    after_warning = translate_coverage_warning(
        after["coverage_warning"],
        request,
        labels["after_warning"],
    )

    if before_warning:
        coverage_warnings.append(before_warning)

    if after_warning:
        coverage_warnings.append(after_warning)

    low_coverage = (
        before["coverage_percent"] < COVERAGE_WARNING_THRESHOLD
        or after["coverage_percent"] < COVERAGE_WARNING_THRESHOLD
    )

    original_classification = change_result["direction"]
    translated_classification = (
        translate_classification(
    original_classification,
    labels,
)
        if hindi
        else original_classification
    )

    if low_coverage and original_classification:
        if hindi:
            classification = (
                f"{labels['provisional_classification']} — "
                f"{translated_classification}; {labels['low_coverage']}"
            )
            note = (
                "कम से कम एक अवधि में उपयोग योग्य पिक्सेल कवरेज कम है। "
                "NDVI परिवर्तन को प्रारंभिक परिणाम मानें। मौसम, बादलों को "
                "हटाने की प्रक्रिया और चित्रों की उपलब्धता परिणाम को प्रभावित "
                "कर सकते हैं। केवल NDVI परिवर्तन से बाढ़ या वनस्पति क्षति की "
                "पुष्टि नहीं होती।"
            )
        else:
            classification = (
                f"Provisional result — {original_classification}; "
                "low valid-pixel coverage"
            )
            note = (
                "This comparison has low valid-pixel coverage in at least "
                "one period. Treat the apparent NDVI change as provisional. "
                "Seasonal variation, cloud masking, and image availability "
                "may affect the result. NDVI change alone does not confirm "
                "vegetation damage or flooding."
            )
    else:
        classification = (
            translated_classification if hindi else original_classification
        )

        if hindi:
            note = (
                "NDVI में बदलाव वनस्पति की हरियाली में बदलाव का संकेत देता है। "
                "यह अपने आप में बाढ़, वनस्पति क्षति या किसी क्षेत्र की सुरक्षा "
                "की पुष्टि नहीं करता। मौसम और उपग्रह चित्रों की उपलब्धता "
                "परिणाम को प्रभावित कर सकती है।"
            )
        else:
            note = (
                "NDVI change indicates a change in vegetation greenness. It "
                "does not independently confirm flooding, vegetation damage, "
                "or whether an area is safe. Seasonal differences and image "
                "availability can affect the results."
            )

    # ---------------------------------------------------------------------------
    # Water extent estimation (MNDWI) — independent of NDVI; failures are
    # caught silently so that the NDVI report always succeeds.
    # ---------------------------------------------------------------------------

    def _estimate_water(composite_image, period_label: str):
        """Return water-extent dict, or None if estimation fails."""
        try:
            return get_mndwi_water_extent(composite_image, aoi)
        except Exception as water_error:
            # Log the reason but do not expose credentials or stack details.
            print(
                f"PRAKRITI water-analysis warning ({period_label}): "
                f"{type(water_error).__name__}: {water_error}"
            )
            return None

    water_before = _estimate_water(
        before_result["composite_image"], "before period"
    )
    water_after = _estimate_water(
        after_result["composite_image"], "after period"
    )

    water_note = (
        "MNDWI-based water extent is an estimate. Seasonal conditions, "
        "shadows, and image quality can affect the result. It does not "
        "independently confirm flooding, drought, or encroachment."
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
        "water": {
            "before": water_before,
            "after": water_after,
            "note": water_note,
        },
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

    print(f"PRAKRITI analysis error: {error}")

    raise HTTPException(
        status_code=500,
        detail="Analysis failed. Check the backend logs for details.",
    ) from error


@router.post("/analyze/change")
def analyze_change(request: ChangeDetectionRequest):
    """Return NDVI change analysis as JSON."""

    try:
        return run_analysis(request)
    except Exception as error:
        handle_analysis_error(error)


@router.post("/analyze/change/pdf")
def analyze_change_pdf(request: ChangeDetectionRequest):
    """Generate and download a PDF report in the selected language."""

    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            KeepTogether,
        )

        result = run_analysis(request)
        labels = report_labels(request)
        hindi = is_hindi(request)
        buffer = BytesIO()

        font_name = "Helvetica"

        if hindi:
            font_path = (
                Path(__file__).resolve().parents[1]
                / "assets"
                / "NotoSansDevanagari-Regular.ttf"
            )

            if not font_path.is_file():
                raise RuntimeError(
                    "Hindi font not found at "
                    "app/assets/NotoSansDevanagari-Regular.ttf"
                )

            if "PrakritiHindi" not in pdfmetrics.getRegisteredFontNames():
                pdfmetrics.registerFont(
                    TTFont(
                        "PrakritiHindi",
                        str(font_path),
                        shapable=True,
                    )
                )

            font_name = "PrakritiHindi"

        document = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=0.55 * inch,
            leftMargin=0.55 * inch,
            topMargin=0.5 * inch,
            bottomMargin=0.5 * inch,
            title=labels["title"],
            author="PRAKRITI",
        )

        styles = getSampleStyleSheet()

        if hindi:
            for style_name in ("Normal", "Italic", "Title", "Heading2"):
                styles[style_name].fontName = font_name

        styles["Title"].textColor = colors.HexColor("#172554")
        styles["Title"].alignment = TA_CENTER
        styles["Title"].spaceAfter = 5

        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName=font_name,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=16,
        )

        warning_style = ParagraphStyle(
            "CoverageWarning",
            parent=styles["Normal"],
            fontName=font_name,
            textColor=colors.HexColor("#92400e"),
            leading=14,
        )

        interpretation_style = ParagraphStyle(
            "Interpretation",
            parent=styles["Italic"],
            fontName=font_name,
            leading=14,
            spaceBefore=3,
        )

        story = [
            Paragraph(escape(labels["title"]), styles["Title"]),
            Paragraph(escape(labels["subtitle"]), subtitle_style),
        ]

        header_cell_style = ParagraphStyle(
            "TableHeaderCell",
            parent=styles["Normal"],
            fontName=font_name,
            textColor=colors.white,
        )

        def safe_cell(value, style=None):
            """Escape text before putting it into a ReportLab Paragraph."""
            return Paragraph(
                escape(str(value)),
                style or styles["Normal"],
            )

        def add_table(data, widths):
            # Paragraph cells allow long Hindi labels to wrap correctly.
            paragraph_data = [
                [
                    safe_cell(cell, header_cell_style if row_index == 0 else None)
                    for cell in row
                ]
                for row_index, row in enumerate(data)
            ]

            table = Table(
                paragraph_data,
                colWidths=widths,
                repeatRows=1,
            )

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
                        [
                            colors.white,
                            colors.HexColor("#f8fafc"),
                        ],
                    ),
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("PADDING", (0, 0), (-1, -1), 6),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ])
            )

            return table

        # Area of interest.
        story.append(
            Paragraph(escape(labels["aoi"]), styles["Heading2"])
        )

        aoi_data = [
            [labels["coordinate"], labels["value"]],
            [labels["west"], f"{request.west:.6f}"],
            [labels["south"], f"{request.south:.6f}"],
            [labels["east"], f"{request.east:.6f}"],
            [labels["north"], f"{request.north:.6f}"],
        ]

        story.append(add_table(aoi_data, [2.2 * inch, 3.5 * inch]))
        story.append(Spacer(1, 16))

        # Comparison periods.
        before = result["before"]
        after = result["after"]

        story.append(
            Paragraph(escape(labels["periods"]), styles["Heading2"])
        )

        period_data = [
            [labels["metric"], labels["before"], labels["after"]],
            [
                labels["date_range"],
                f"{request.before_start} {labels['date_to']} {request.before_end}",
                f"{request.after_start} {labels['date_to']} {request.after_end}",
            ],
            [
                labels["mean_ndvi"],
                (
                    f"{before['mean_ndvi']:.6f}"
                    if before["mean_ndvi"] is not None
                    else labels["not_available"]
                ),
                (
                    f"{after['mean_ndvi']:.6f}"
                    if after["mean_ndvi"] is not None
                    else labels["not_available"]
                ),
            ],
            [
                labels["scenes"],
                str(before["scenes"]),
                str(after["scenes"]),
            ],
            [
                labels["valid_pixels"],
                f"{before['valid_pixel_count']:,}",
                f"{after['valid_pixel_count']:,}",
            ],
            [
                labels["total_pixels"],
                f"{before['total_pixel_count']:,}",
                f"{after['total_pixel_count']:,}",
            ],
            [
                labels["coverage"],
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

        # Water extent section.
        water_result = result.get("water", {}) or {}
        water_before_data = water_result.get("before")
        water_after_data = water_result.get("after")

        def _fmt_water_val(data, key):
            """Format a water value, showing N/A if data is missing."""
            if data is None:
                return labels["water_not_available"]
            val = data.get(key)
            if val is None:
                return labels["water_not_available"]
            return f"{float(val):.4f}"

        def _fmt_water_pct(data):
            """Format water percentage, showing N/A if data is missing."""
            if data is None:
                return labels["water_not_available"]
            val = data.get("water_percent_valid")
            if val is None:
                return labels["water_not_available"]
            return f"{float(val):.2f}%"

        story.append(
            Paragraph(escape(labels["water_extent"]), styles["Heading2"])
        )

        water_data = [
            [labels["metric"], labels["before"], labels["after"]],
            [
                labels["water_area"],
                _fmt_water_val(water_before_data, "water_area_km2"),
                _fmt_water_val(water_after_data, "water_area_km2"),
            ],
            [
                labels["valid_area"],
                _fmt_water_val(water_before_data, "valid_area_km2"),
                _fmt_water_val(water_after_data, "valid_area_km2"),
            ],
            [
                labels["water_percent"],
                _fmt_water_pct(water_before_data),
                _fmt_water_pct(water_after_data),
            ],
        ]

        story.append(
            add_table(
                water_data,
                [1.75 * inch, 2.0 * inch, 2.0 * inch],
            )
        )
        story.append(Spacer(1, 14))

        # Coverage warnings.
        if result["coverage_warnings"]:
            warning_items = [
                Paragraph(
                    escape(labels["warnings"]),
                    styles["Heading2"],
                )
            ]

            for warning in result["coverage_warnings"]:
                warning_items.append(
                    Paragraph(escape(warning), warning_style)
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
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("LEFTPADDING", (0, 0), (-1, -1), 9),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ])
            )

            story.append(
                KeepTogether([
                    warning_table,
                    Spacer(1, 10),
                ])
            )

        coverage_explanation = (
    f"{labels['coverage_explanation']}: "
    f"{COVERAGE_WARNING_THRESHOLD:.0f}%"
    if hindi
    else (
        f"{labels['coverage_explanation']}: "
        f"{COVERAGE_WARNING_THRESHOLD:.0f}%."
    )
)

        story.append(
            Paragraph(
                escape(coverage_explanation),
                styles["Italic"],
            )
        )
        story.append(Spacer(1, 14))

        # Analysis results.
        story.append(
            Paragraph(escape(labels["results"]), styles["Heading2"])
        )

        change = result["change_percent"]

        change_text = (
            f"{change:+.1f}%"
            if isinstance(change, (int, float))
            else labels["not_available"]
        )

        story.append(
            Paragraph(
                escape(f"{labels['change']}: {change_text}"),
                styles["Normal"],
            )
        )

        story.append(
            Paragraph(
                escape(
                    f"{labels['classification']}: "
                    f"{result['classification']}"
                ),
                styles["Normal"],
            )
        )
        story.append(Spacer(1, 8))

        # Interpretation.
        after_coverage = after["coverage_percent"]
        coverage_threshold = result["coverage_warning_threshold"]

        if isinstance(change, (int, float)):
            if change < 0:
                change_description = (
                    f"{labels['ndvi_decreased_by']} {abs(change):.1f}%"
                )
            elif change > 0:
                change_description = (
                    f"{labels['ndvi_increased_by']} {change:.1f}%"
                )
            else:
                change_description = labels["no_change"]
        else:
            change_description = labels["could_not_calculate"]

        if isinstance(after_coverage, (int, float)):
            if after_coverage < coverage_threshold:
                if hindi:
                    coverage_sentence = (
                        f"{labels['coverage_sentence_low']} "
                        f"{after_coverage:.2f}% "
                        f"{labels['coverage_sentence_low_end']} "
                        f"{labels['threshold']}: "
                        f"{coverage_threshold:.0f}%. "
                    )
                else:
                    coverage_sentence = (
                        f"{labels['coverage_sentence_low']} "
                        f"{after_coverage:.2f}% "
                        f"{labels['coverage_sentence_low_end']} "
                        f"Threshold: {coverage_threshold:.0f}%. "
                    )
            else:
                coverage_sentence = (
                    f"{labels['coverage_sentence_ok']} "
                    f"{after_coverage:.2f}%. "
                )
        else:
            coverage_sentence = labels["coverage_unavailable"] + " "

        if result["coverage_warnings"]:
            interpretation = (
                f"{labels['interpretation']}: "
                f"{change_description}. "
                f"{coverage_sentence}"
                f"{labels['provisional']} "
                f"{labels['caution']}"
            )
        else:
            interpretation = (
                f"{labels['interpretation']}: "
                f"{change_description}. "
                f"{coverage_sentence}"
                f"{labels['caution']}"
            )

        story.append(
            Paragraph(
                escape(interpretation),
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
                    'attachment; filename="prakriti-ndvi-report.pdf"'
                )
            },
        )

    except Exception as error:
        handle_analysis_error(error)
