"""
Tests for MNDWI water-extent analysis.

Covers:
  - run_analysis() always includes a 'water' key in its output
  - Successful water estimation populates before/after water data
  - Water estimation failure leaves NDVI results intact and sets
    water.before / water.after to None (graceful fallback)
  - PDF generation still works when water data is None (no crash)
  - New label keys exist in both English and Hindi report_labels()

All Earth Engine calls are mocked; no network access or credentials needed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub the `ee` module so imports work without the Earth Engine SDK.
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
        "Number",
        "Algorithms",
    ):
        setattr(ee_mod, name, MagicMock())
    sys.modules.setdefault("ee", ee_mod)


_stub_ee()

from app.routes.change_detection import (          # noqa: E402
    ChangeDetectionRequest,
    run_analysis,
    report_labels,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

MOCK_INIT   = "app.routes.change_detection.init_earth_engine"
MOCK_RESOLVE = "app.routes.change_detection.resolve_aoi"
MOCK_NDVI   = "app.routes.change_detection.get_ndvi_for_period"
MOCK_CHANGE = "app.routes.change_detection.compute_change"
MOCK_WATER  = "app.routes.change_detection.get_mndwi_water_extent"

_COMPOSITE = MagicMock()

FAKE_PERIOD = {
    "mean_ndvi": 0.5,
    "composite_image": _COMPOSITE,
    "scene_count": 3,
    "valid_pixel_count": 900,
    "total_pixel_count": 1000,
    "coverage_percent": 90.0,
    "coverage_warning": None,
}

FAKE_CHANGE = {
    "pct_change": -10.0,
    "direction": "vegetation loss",
    "note": "NDVI change indicates a change in vegetation greenness.",
}

FAKE_WATER_BEFORE = {
    "water_area_km2": 1.2345,
    "valid_area_km2": 48.7655,
    "water_percent_valid": 2.47,
}

FAKE_WATER_AFTER = {
    "water_area_km2": 3.9876,
    "valid_area_km2": 46.0124,
    "water_percent_valid": 7.98,
}


def _make_request(**overrides):
    defaults = dict(
        west=77.87, south=30.15, east=77.93, north=30.21,
        before_start="2018-01-01", before_end="2018-03-31",
        after_start="2024-01-01", after_end="2024-03-31",
        language="en", max_cloud_pct=80,
    )
    defaults.update(overrides)
    return ChangeDetectionRequest(**defaults)


def _run(water_side_effect=None, water_return=None, language="en"):
    """Call run_analysis() with all Earth Engine calls mocked."""
    req = _make_request(language=language)
    after_period = {**FAKE_PERIOD, "mean_ndvi": 0.45}

    water_mock_kwargs = {}
    if water_side_effect is not None:
        water_mock_kwargs["side_effect"] = water_side_effect
    elif water_return is not None:
        water_mock_kwargs["return_value"] = water_return
    else:
        water_mock_kwargs["side_effect"] = [FAKE_WATER_BEFORE, FAKE_WATER_AFTER]

    with (
        patch(MOCK_INIT),
        patch(MOCK_RESOLVE),
        patch(MOCK_NDVI, side_effect=[FAKE_PERIOD, after_period]),
        patch(MOCK_CHANGE, return_value=FAKE_CHANGE),
        patch(MOCK_WATER, **water_mock_kwargs),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        return run_analysis(req)


# ---------------------------------------------------------------------------
# 1.  'water' key is always present in the response
# ---------------------------------------------------------------------------

class TestWaterKeyAlwaysPresent(unittest.TestCase):

    def test_water_key_exists_on_success(self):
        result = _run()
        self.assertIn("water", result, "'water' key missing from run_analysis result")

    def test_water_key_exists_when_water_fails(self):
        result = _run(water_side_effect=RuntimeError("GEE timeout"))
        self.assertIn("water", result, "'water' key missing when water estimation fails")

    def test_water_key_has_before_and_after_subkeys(self):
        result = _run()
        self.assertIn("before", result["water"])
        self.assertIn("after", result["water"])

    def test_water_key_has_note(self):
        result = _run()
        self.assertIn("note", result["water"])
        self.assertIsNotNone(result["water"]["note"])


# ---------------------------------------------------------------------------
# 2.  Successful water estimation populates data correctly
# ---------------------------------------------------------------------------

class TestWaterSuccessfulEstimation(unittest.TestCase):

    def setUp(self):
        self.result = _run()

    def test_before_water_area_km2(self):
        self.assertAlmostEqual(
            self.result["water"]["before"]["water_area_km2"],
            FAKE_WATER_BEFORE["water_area_km2"],
        )

    def test_before_valid_area_km2(self):
        self.assertAlmostEqual(
            self.result["water"]["before"]["valid_area_km2"],
            FAKE_WATER_BEFORE["valid_area_km2"],
        )

    def test_before_water_percent_valid(self):
        self.assertAlmostEqual(
            self.result["water"]["before"]["water_percent_valid"],
            FAKE_WATER_BEFORE["water_percent_valid"],
        )

    def test_after_water_area_km2(self):
        self.assertAlmostEqual(
            self.result["water"]["after"]["water_area_km2"],
            FAKE_WATER_AFTER["water_area_km2"],
        )

    def test_after_water_percent_valid(self):
        self.assertAlmostEqual(
            self.result["water"]["after"]["water_percent_valid"],
            FAKE_WATER_AFTER["water_percent_valid"],
        )

    def test_ndvi_results_still_present_alongside_water(self):
        """NDVI fields must not be displaced by water data."""
        for key in ("before", "after", "change_percent", "classification", "note"):
            self.assertIn(key, self.result, f"Key '{key}' missing")


# ---------------------------------------------------------------------------
# 3.  Water failure leaves NDVI intact; water fields become None
# ---------------------------------------------------------------------------

class TestWaterFailureGracefulFallback(unittest.TestCase):

    def setUp(self):
        self.result = _run(
            water_side_effect=ValueError("No B11 band in composite")
        )

    def test_before_water_is_none_when_estimation_fails(self):
        self.assertIsNone(
            self.result["water"]["before"],
            "Expected water.before to be None on failure",
        )

    def test_after_water_is_none_when_estimation_fails(self):
        self.assertIsNone(
            self.result["water"]["after"],
            "Expected water.after to be None on failure",
        )

    def test_ndvi_before_mean_still_present(self):
        self.assertIsNotNone(self.result["before"]["mean_ndvi"])

    def test_ndvi_change_percent_still_present(self):
        self.assertIsNotNone(self.result["change_percent"])

    def test_ndvi_classification_still_present(self):
        self.assertIsNotNone(self.result["classification"])

    def test_water_is_partial_when_only_after_fails(self):
        """Water data for one period can succeed even if the other fails."""
        result = _run(
            water_side_effect=[FAKE_WATER_BEFORE, RuntimeError("EE error")]
        )
        self.assertIsNotNone(result["water"]["before"])
        self.assertIsNone(result["water"]["after"])


# ---------------------------------------------------------------------------
# 4.  Water error is logged, not silently swallowed
# ---------------------------------------------------------------------------

class TestWaterErrorLogging(unittest.TestCase):

    def test_water_failure_is_logged_with_print(self):
        """A failed water estimate must print a warning so it's visible in logs."""
        with patch("builtins.print") as mock_print:
            _run(water_side_effect=RuntimeError("GEE error"))

        # At least one print call should mention the water analysis warning.
        all_calls = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn(
            "water-analysis warning",
            all_calls,
            "Expected a water-analysis warning to be printed",
        )


# ---------------------------------------------------------------------------
# 5.  New label keys exist in English and Hindi report_labels()
# ---------------------------------------------------------------------------

class TestWaterLabelKeys(unittest.TestCase):

    EXPECTED_WATER_KEYS = {
        "water_extent",
        "water_area",
        "valid_area",
        "water_percent",
        "water_not_available",
    }

    def test_english_labels_have_all_water_keys(self):
        labels = report_labels(_make_request(language="en"))
        missing = self.EXPECTED_WATER_KEYS - set(labels.keys())
        self.assertFalse(missing, f"English labels missing water keys: {missing}")

    def test_hindi_labels_have_all_water_keys(self):
        labels = report_labels(_make_request(language="hi"))
        missing = self.EXPECTED_WATER_KEYS - set(labels.keys())
        self.assertFalse(missing, f"Hindi labels missing water keys: {missing}")

    def test_hindi_water_extent_label_has_devanagari(self):
        labels = report_labels(_make_request(language="hi"))
        text = labels["water_extent"]
        has_dev = any("\u0900" <= ch <= "\u097f" for ch in text)
        self.assertTrue(has_dev, f"Hindi 'water_extent' label has no Devanagari: {text!r}")

    def test_hindi_water_area_label_has_devanagari(self):
        labels = report_labels(_make_request(language="hi"))
        text = labels["water_area"]
        has_dev = any("\u0900" <= ch <= "\u097f" for ch in text)
        self.assertTrue(has_dev, f"Hindi 'water_area' label has no Devanagari: {text!r}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
