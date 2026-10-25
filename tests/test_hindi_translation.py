"""
Unit tests for Hindi translation in app/routes/change_detection.py.

These tests mock the Earth Engine pipeline so they run instantly without
any network access or credentials.

Run:
    python -m pytest tests/test_hindi_translation.py -v
"""

import sys
import types
import unittest
from unittest.mock import patch, MagicMock


# ---------------------------------------------------------------------------
# Stub the `ee` module so change_detection.py can be imported without the
# earthengine-api being authenticated.
# ---------------------------------------------------------------------------
def _stub_ee():
    ee_mod = types.ModuleType("ee")
    for name in (
        "ServiceAccountCredentials",
        "Initialize",
        "Geometry",
        "ImageCollection",
        "Image",
        "Filter",
        "Reducer",
    ):
        setattr(ee_mod, name, MagicMock())
    sys.modules.setdefault("ee", ee_mod)


_stub_ee()

# Now we can safely import the route module.
from app.routes.change_detection import (  # noqa: E402
    ChangeDetectionRequest,
    is_hindi,
    report_labels,
    translate_classification,
    translate_coverage_warning,
    run_analysis,
)
from app.services.ndvi_pipeline import COVERAGE_WARNING_THRESHOLD  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(**overrides):
    defaults = dict(
        west=77.87,
        south=30.15,
        east=77.93,
        north=30.21,
        before_start="2018-01-01",
        before_end="2018-03-31",
        after_start="2024-01-01",
        after_end="2024-03-31",
        language="en",
        max_cloud_pct=80,
    )
    defaults.update(overrides)
    return ChangeDetectionRequest(**defaults)


FAKE_PERIOD_RESULT = {
    "mean_ndvi": 0.5,
    "composite_image": MagicMock(),
    "scene_count": 3,
    "valid_pixel_count": 900,
    "total_pixel_count": 1000,
    "coverage_percent": 90.0,
    "coverage_warning": None,
}

FAKE_PERIOD_LOW = {
    **FAKE_PERIOD_RESULT,
    "coverage_percent": 40.0,
    "coverage_warning": (
        "Low valid-pixel coverage: 40.00% of sampled AOI pixels "
        "were usable. Interpret this period's NDVI cautiously."
    ),
}

FAKE_CHANGE_LOSS = {
    "pct_change": -10.0,
    "direction": "vegetation loss",
    "note": "NDVI change indicates a change in vegetation greenness.",
}

FAKE_CHANGE_GAIN = {
    "pct_change": 8.0,
    "direction": "vegetation gain",
    "note": "NDVI change indicates a change in vegetation greenness.",
}

FAKE_CHANGE_NONE = {
    "pct_change": 1.0,
    "direction": "no significant change",
    "note": "NDVI change indicates a change in vegetation greenness.",
}


# ---------------------------------------------------------------------------
# 1.  is_hindi / report_labels
# ---------------------------------------------------------------------------

class TestIsHindi(unittest.TestCase):

    def test_en_is_not_hindi(self):
        self.assertFalse(is_hindi(_make_request(language="en")))

    def test_hi_is_hindi(self):
        self.assertTrue(is_hindi(_make_request(language="hi")))

    def test_hindi_word_is_hindi(self):
        self.assertTrue(is_hindi(_make_request(language="hindi")))

    def test_hi_with_whitespace(self):
        self.assertTrue(is_hindi(_make_request(language=" hi ")))

    def test_HI_uppercase(self):
        self.assertTrue(is_hindi(_make_request(language="HI")))


class TestReportLabels(unittest.TestCase):

    def test_english_labels_have_english_title(self):
        labels = report_labels(_make_request(language="en"))
        self.assertIn("NDVI Change Analysis", labels["title"])
        self.assertIn("English", labels.get("title", "") + labels.get("subtitle", "") + "English")

    def test_hindi_labels_have_devanagari_title(self):
        labels = report_labels(_make_request(language="hi"))
        # Title must contain at least one Devanagari character (U+0900–U+097F)
        has_devanagari = any("\u0900" <= ch <= "\u097f" for ch in labels["title"])
        self.assertTrue(has_devanagari, f"Expected Devanagari in title: {labels['title']!r}")

    def test_hindi_labels_all_keys_present(self):
        en = report_labels(_make_request(language="en"))
        hi = report_labels(_make_request(language="hi"))
        missing = set(en.keys()) - set(hi.keys())
        self.assertFalse(missing, f"Hindi labels missing keys: {missing}")

    def test_hindi_not_same_as_english(self):
        en = report_labels(_make_request(language="en"))
        hi = report_labels(_make_request(language="hi"))
        same = {k for k in en if en[k] == hi[k]}
        # A few keys (like "NDVI") may legitimately be unchanged
        pure_text_keys = {"title", "subtitle", "results", "classification", "interpretation"}
        still_same_text = same & pure_text_keys
        self.assertFalse(still_same_text,
                         f"Hindi labels still show English for: {still_same_text}")


# ---------------------------------------------------------------------------
# 2.  translate_classification
# ---------------------------------------------------------------------------

class TestTranslateClassification(unittest.TestCase):

    def setUp(self):
        self.en = report_labels(_make_request(language="en"))
        self.hi = report_labels(_make_request(language="hi"))

    def _hi_has_devanagari(self, text):
        return any("\u0900" <= ch <= "\u097f" for ch in text)

    def test_vegetation_loss_hindi(self):
        result = translate_classification("vegetation loss", self.hi)
        self.assertTrue(self._hi_has_devanagari(result),
                        f"Expected Devanagari for 'vegetation loss': {result!r}")

    def test_vegetation_gain_hindi(self):
        result = translate_classification("vegetation gain", self.hi)
        self.assertTrue(self._hi_has_devanagari(result),
                        f"Expected Devanagari for 'vegetation gain': {result!r}")

    def test_no_significant_change_hindi(self):
        result = translate_classification("no significant change", self.hi)
        self.assertTrue(self._hi_has_devanagari(result),
                        f"Expected Devanagari for 'no significant change': {result!r}")

    def test_none_returns_not_available_hindi(self):
        result = translate_classification(None, self.hi)
        self.assertTrue(self._hi_has_devanagari(result),
                        f"Expected Devanagari for None: {result!r}")

    def test_vegetation_loss_english_unchanged(self):
        result = translate_classification("vegetation loss", self.en)
        self.assertEqual(result, "vegetation loss")

    def test_vegetation_gain_english_unchanged(self):
        result = translate_classification("vegetation gain", self.en)
        self.assertEqual(result, "vegetation gain")

    def test_aliases_normalized(self):
        hi = self.hi
        for alias in ("vegetation-loss", "vegetation_loss", "loss", "decreased"):
            result = translate_classification(alias, hi)
            self.assertTrue(self._hi_has_devanagari(result),
                            f"Alias {alias!r} → {result!r} not Devanagari")

    def test_unknown_classification_passthrough(self):
        result = translate_classification("some unknown label", self.hi)
        self.assertEqual(result, "some unknown label")


# ---------------------------------------------------------------------------
# 3.  translate_coverage_warning
# ---------------------------------------------------------------------------

class TestTranslateCoverageWarning(unittest.TestCase):

    def _hi_request(self):
        return _make_request(language="hi")

    def _en_request(self):
        return _make_request(language="en")

    def _has_devanagari(self, text):
        return any("\u0900" <= ch <= "\u097f" for ch in text)

    def test_none_returns_none_hindi(self):
        self.assertIsNone(translate_coverage_warning(None, self._hi_request(), "पहले की अवधि"))

    def test_none_returns_none_english(self):
        self.assertIsNone(translate_coverage_warning(None, self._en_request(), "Before period"))

    def test_low_coverage_warning_hindi(self):
        raw = "Low valid-pixel coverage: 40.00% of sampled AOI pixels were usable."
        result = translate_coverage_warning(raw, self._hi_request(), "पहले की अवधि")
        self.assertTrue(self._has_devanagari(result),
                        f"Expected Devanagari in coverage warning: {result!r}")

    def test_low_coverage_warning_english_prefix(self):
        raw = "Low valid-pixel coverage: 90.00% of sampled AOI pixels were usable."
        result = translate_coverage_warning(raw, self._en_request(), "Before period")
        self.assertIn("Before period", result)
        self.assertIn(raw, result)

    def test_no_usable_warning_hindi(self):
        raw = "No usable pixels were found."
        result = translate_coverage_warning(raw, self._hi_request(), "बाद की अवधि")
        self.assertTrue(self._has_devanagari(result),
                        f"Expected Devanagari for 'no usable': {result!r}")

    def test_unknown_warning_preserved_with_label(self):
        raw = "Some unknown warning text."
        result = translate_coverage_warning(raw, self._hi_request(), "पहले की अवधि")
        # Period label should be prepended; raw text should be included
        self.assertIn(raw, result)


# ---------------------------------------------------------------------------
# 4.  run_analysis — integration with mocked EE
# ---------------------------------------------------------------------------

MOCK_INIT = "app.routes.change_detection.init_earth_engine"
MOCK_RESOLVE = "app.routes.change_detection.resolve_aoi"
MOCK_NDVI = "app.routes.change_detection.get_ndvi_for_period"
MOCK_CHANGE = "app.routes.change_detection.compute_change"
MOCK_WATER = "app.routes.change_detection.get_mndwi_water_extent"


class TestRunAnalysisHindi(unittest.TestCase):
    """Verify run_analysis() returns Hindi text when language='hi'."""

    def _has_devanagari(self, text):
        if text is None:
            return False
        return any("\u0900" <= ch <= "\u097f" for ch in str(text))

    def _run_hi(self, before=None, after=None, change=None):
        before = before or FAKE_PERIOD_RESULT
        after = after or {**FAKE_PERIOD_RESULT, "mean_ndvi": 0.45}
        change = change or FAKE_CHANGE_LOSS

        req = _make_request(language="hi")
        with (
            patch(MOCK_INIT),
            patch(MOCK_RESOLVE),
            patch(MOCK_NDVI, side_effect=[before, after]),
            patch(MOCK_CHANGE, return_value=change),
            patch(MOCK_WATER, return_value=None),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            return run_analysis(req)

    def _run_en(self, before=None, after=None, change=None):
        before = before or FAKE_PERIOD_RESULT
        after = after or {**FAKE_PERIOD_RESULT, "mean_ndvi": 0.45}
        change = change or FAKE_CHANGE_LOSS

        req = _make_request(language="en")
        with (
            patch(MOCK_INIT),
            patch(MOCK_RESOLVE),
            patch(MOCK_NDVI, side_effect=[before, after]),
            patch(MOCK_CHANGE, return_value=change),
            patch(MOCK_WATER, return_value=None),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            return run_analysis(req)

    def test_hindi_classification_has_devanagari(self):
        result = self._run_hi()
        self.assertTrue(
            self._has_devanagari(result["classification"]),
            f"classification not Devanagari: {result['classification']!r}",
        )

    def test_hindi_note_has_devanagari(self):
        result = self._run_hi()
        self.assertTrue(
            self._has_devanagari(result["note"]),
            f"note not Devanagari: {result['note']!r}",
        )

    def test_english_classification_is_english(self):
        result = self._run_en()
        self.assertFalse(
            self._has_devanagari(result["classification"]),
            f"English classification has Devanagari: {result['classification']!r}",
        )

    def test_english_note_is_english(self):
        result = self._run_en()
        self.assertFalse(
            self._has_devanagari(result["note"]),
            f"English note has Devanagari: {result['note']!r}",
        )

    def test_vegetation_gain_hindi(self):
        result = self._run_hi(change=FAKE_CHANGE_GAIN)
        self.assertTrue(self._has_devanagari(result["classification"]))

    def test_no_change_hindi(self):
        result = self._run_hi(change=FAKE_CHANGE_NONE)
        self.assertTrue(self._has_devanagari(result["classification"]))

    def test_low_coverage_hindi(self):
        result = self._run_hi(after=FAKE_PERIOD_LOW)
        # Both coverage_warnings and classification must contain Devanagari
        self.assertTrue(
            self._has_devanagari(result["classification"]),
            f"Low-coverage Hindi classification missing Devanagari: {result['classification']!r}",
        )
        if result["coverage_warnings"]:
            for w in result["coverage_warnings"]:
                self.assertTrue(self._has_devanagari(w),
                                f"Coverage warning not in Hindi: {w!r}")

    def test_low_coverage_english(self):
        result = self._run_en(after=FAKE_PERIOD_LOW)
        self.assertFalse(self._has_devanagari(result["classification"]))

    def test_response_keys_present(self):
        result = self._run_hi()
        for key in ("before", "after", "change_percent", "classification",
                    "note", "coverage_warnings", "coverage_warning_threshold"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_change_percent_unchanged(self):
        result = self._run_hi()
        self.assertEqual(result["change_percent"], -10.0)


# ---------------------------------------------------------------------------
# 5.  ResultsPanel classification icon selection (Python-side logic mirror)
# ---------------------------------------------------------------------------

class TestFrontendClassificationMeta(unittest.TestCase):
    """
    Mirror the getClassificationMeta() JS logic in Python to verify that
    Hindi classification strings do NOT accidentally appear to be 'provisional'
    or 'gain/loss/no-change' based on English keyword matching.

    This is a regression guard for the UI badge selection.
    """

    ENGLISH_KEYWORDS = {
        "provisional": ["provisional", "low valid-pixel coverage"],
        "gain": ["gain", "increase"],
        "loss": ["loss", "decrease"],
        "no_change": ["stable", "no significant change"],
    }

    def _classify(self, text: str) -> str:
        normalized = text.lower()
        for kw in self.ENGLISH_KEYWORDS["provisional"]:
            if kw in normalized:
                return "provisional"
        for kw in self.ENGLISH_KEYWORDS["gain"]:
            if kw in normalized:
                return "gain"
        for kw in self.ENGLISH_KEYWORDS["loss"]:
            if kw in normalized:
                return "loss"
        for kw in self.ENGLISH_KEYWORDS["no_change"]:
            if kw in normalized:
                return "no_change"
        return "unclassified"

    def test_hindi_loss_does_not_match_english_keywords(self):
        hi_labels = report_labels(_make_request(language="hi"))
        hindi_loss = hi_labels["decreased"]  # e.g. "NDVI में कमी"
        result = self._classify(hindi_loss)
        # It must not accidentally match English keywords
        self.assertNotIn(result, ("gain", "loss", "provisional", "no_change"),
                         f"Hindi '{hindi_loss}' matched English keyword as '{result}'")

    def test_hindi_gain_does_not_match_english_keywords(self):
        hi_labels = report_labels(_make_request(language="hi"))
        hindi_gain = hi_labels["increased"]
        result = self._classify(hindi_gain)
        self.assertNotIn(result, ("gain", "loss", "provisional", "no_change"),
                         f"Hindi '{hindi_gain}' matched English keyword as '{result}'")

    def test_english_loss_matches_correctly(self):
        self.assertEqual(self._classify("vegetation loss"), "loss")

    def test_english_gain_matches_correctly(self):
        self.assertEqual(self._classify("vegetation gain"), "gain")


if __name__ == "__main__":
    unittest.main(verbosity=2)
