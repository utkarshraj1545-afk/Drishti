"""English / Hindi tests for the JSON analysis and the PDF report.

Earth Engine is replaced with fixed numbers, so these tests need no
credentials and no internet. A fake key file is created in a temp folder.
"""
import re

import pytest

import app.routes.change_detection as cd

DEVANAGARI = re.compile("[\u0900-\u097f]")


def has_hindi(text: str) -> bool:
    return bool(DEVANAGARI.search(text))


def english_words(text: str) -> list:
    """English words left in a string. 'NDVI' is an acronym, so it is allowed."""
    return [w for w in re.findall(r"[A-Za-z]{3,}", text) if w != "NDVI"]


def make_period(mean, coverage=95.0, warning=None):
    return {
        "mean_ndvi": mean,
        "scene_count": 5,
        "valid_pixel_count": 9500,
        "total_pixel_count": 10000,
        "coverage_percent": coverage,
        "coverage_warning": warning,
    }


LOW_COVERAGE_WARNING = (
    "Low valid-pixel coverage: 41.50% of sampled AOI pixels were usable. "
    "Interpret this period's NDVI cautiously."
)


@pytest.fixture
def fake_ee(monkeypatch, tmp_path):
    """Replace Earth Engine with fixed values. Returns a settings dict."""
    settings = {
        "before": make_period(0.30),
        "after": make_period(0.40),  # +33% -> vegetation gain
    }

    key_file = tmp_path / "fake-key.json"
    key_file.write_text("{}")  # only has to exist; it is never read

    def fake_period(aoi, start_date, end_date, max_cloud_pct=80):
        return settings["before"] if start_date.startswith("2023") else settings["after"]

    monkeypatch.setattr(cd, "SERVICE_ACCOUNT_KEY", str(key_file))
    monkeypatch.setattr(cd, "init_earth_engine", lambda *a, **k: None)
    monkeypatch.setattr(cd, "resolve_aoi", lambda *a, **k: "fake-aoi")
    monkeypatch.setattr(cd, "get_ndvi_for_period", fake_period)
    return settings


def body(language="en"):
    return {
        "west": 77.20, "south": 28.60, "east": 77.30, "north": 28.70,
        "before_start": "2023-10-01", "before_end": "2023-11-30",
        "after_start": "2024-10-01", "after_end": "2024-11-30",
        "language": language,
        "max_cloud_pct": 80,
    }


def analyse(language="en"):
    return cd.run_analysis(cd.ChangeDetectionRequest(**body(language)))


# ---------------------------------------------------------------- JSON: Hindi

def test_json_hindi_gain(fake_ee):
    result = analyse("hi")
    assert result["classification"] == "NDVI में वृद्धि"
    assert has_hindi(result["note"])
    assert english_words(result["classification"]) == []
    assert english_words(result["note"]) == []


def test_json_hindi_loss(fake_ee):
    fake_ee["before"], fake_ee["after"] = make_period(0.50), make_period(0.30)
    result = analyse("hi")
    assert result["classification"] == "NDVI में कमी"


def test_json_hindi_low_coverage(fake_ee):
    fake_ee["after"] = make_period(0.40, 41.5, LOW_COVERAGE_WARNING)
    result = analyse("hi")
    assert result["classification"].startswith("प्रारंभिक परिणाम")
    assert has_hindi(result["note"])
    assert len(result["coverage_warnings"]) == 1
    assert has_hindi(result["coverage_warnings"][0])
    assert english_words(result["classification"]) == []
    assert english_words(result["note"]) == []
    assert english_words(result["coverage_warnings"][0]) == []


@pytest.mark.parametrize(
    "language", ["hi", "HI", " hi ", "hindi", "Hindi", "hi-IN", "hi_IN"]
)
def test_json_hindi_language_spellings(fake_ee, language):
    assert has_hindi(analyse(language)["classification"])


# -------------------------------------------------------------- JSON: English

def test_json_english_is_unchanged(fake_ee):
    result = analyse("en")
    assert result["classification"] == "vegetation gain"
    assert result["note"].startswith("NDVI change indicates a change in vegetation")
    assert not has_hindi(result["note"])


def test_json_english_low_coverage_is_unchanged(fake_ee):
    fake_ee["after"] = make_period(0.40, 41.5, LOW_COVERAGE_WARNING)
    result = analyse("en")
    assert result["classification"] == (
        "Provisional result — vegetation gain; low valid-pixel coverage"
    )
    assert result["coverage_warnings"] == [f"After period: {LOW_COVERAGE_WARNING}"]


def test_json_unknown_language_falls_back_to_english(fake_ee):
    assert analyse("fr")["classification"] == "vegetation gain"


def test_json_numbers_identical_in_both_languages(fake_ee):
    hindi, english = analyse("hi"), analyse("en")
    assert hindi["change_percent"] == english["change_percent"]
    assert hindi["before"] == english["before"]
    assert hindi["after"] == english["after"]


# ------------------------------------------------- HTTP + PDF (use TestClient)

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def test_http_hindi_response_is_utf8_hindi(fake_ee, client):
    response = client.post("/analyze/change", json=body("hi"))
    assert response.status_code == 200
    assert response.headers["content-type"].lower().startswith("application/json")
    assert response.json()["classification"] == "NDVI में वृद्धि"


def test_http_english_response(fake_ee, client):
    response = client.post("/analyze/change", json=body("en"))
    assert response.status_code == 200
    assert response.json()["classification"] == "vegetation gain"


def test_pdf_english_downloads_without_hindi_font(fake_ee, client):
    response = client.post("/analyze/change/pdf", json=body("en"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert b"Devanagari" not in response.content


def test_pdf_hindi_downloads_with_devanagari_font(fake_ee, client):
    from reportlab.pdfbase import pdfmetrics

    response = client.post("/analyze/change/pdf", json=body("hi"))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
    assert "PrakritiHindi" in pdfmetrics.getRegisteredFontNames()
    assert b"NotoSansDevanagari" in response.content  # font was embedded


def test_pdf_hindi_low_coverage_builds(fake_ee, client):
    fake_ee["after"] = make_period(0.40, 41.5, LOW_COVERAGE_WARNING)
    response = client.post("/analyze/change/pdf", json=body("hi"))
    assert response.status_code == 200
    assert response.content.startswith(b"%PDF")
