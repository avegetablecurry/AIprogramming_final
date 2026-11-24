import tempfile
import unittest
from pathlib import Path

from predict import ROOT, YEAR, load, run, jeongsi_forecast


class AdmissionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as folder:
            cls.rows, cls.scores = run(ROOT / "data", Path(folder), False)

    def test_all_outputs(self):
        self.assertEqual(len(self.rows), 52)
        self.assertEqual(len({(r["track"], r["scope"], r["target"]) for r in self.rows}), 52)

    def test_rounding_and_cut_types(self):
        past, _, _ = load(ROOT / "data")
        r = next(r for r in past if r["track"] == "일반전형" and r["year"] == 2026)
        self.assertEqual(r["competition"], 4.46)
        cut = next(r for r in self.rows if r["track"] == "일반전형" and r["target"] == "final_cut70")
        avg = next(r for r in self.rows if r["track"] == "일반전형" and r["target"] == "final_avg")
        self.assertAlmostEqual(cut["value"], avg["value"] + (0.20 + 0.23)/2)
        self.assertEqual((cut["n"], cut["검증건수"]), (2, 1))

    def test_absent_cut(self):
        cut = next(r for r in self.rows if "Ⅱ" in r["track"] and r["target"] == "final_cut70")
        self.assertIsNone(cut["value"])

    def test_regular_scope_and_consistency(self):
        groups = {}
        for r in self.rows:
            if r["mode"] == "정시":
                groups.setdefault((r["scope"],r["track"]), {})[r["target"]] = r["value"]
        self.assertEqual(len(groups), 4)
        for g in groups.values():
            self.assertEqual(g["competition"], g["applicants"]/g["quota"])

    def test_year_boundary(self):
        _, regular, _ = load(ROOT / "data")
        series = regular["AI융합학과", "Normal"]
        reference = jeongsi_forecast(series, YEAR, 3)
        self.assertEqual(reference, jeongsi_forecast(series + [dict(year=YEAR, quota=99, competition=99)], YEAR, 3))
        self.assertTrue(all(r["year"] < YEAR for r in self.scores))


if __name__ == "__main__":
    unittest.main()
