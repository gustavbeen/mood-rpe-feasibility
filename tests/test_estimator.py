"""Small extraction checks against the unchanged NB6 function; synthetic only."""
import ast
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "notebooks"))
from estimator import fit_intervals

PREDICTORS = [
    "mood_within", "CR", "EV", "RPE", "within_x_RPE", "between_x_RPE",
]


def nb6_functions():
    # Read only function definitions: never execute notebook batches or load data.
    notebook = json.loads((ROOT / "notebooks/06_mood_rpe_simulation.ipynb").read_text())
    wanted = {"generate_reward_intervals", "simulate_play", "simulate_dataset",
              "fit_simulated_with_intervals"}
    namespace = {"np": np, "pd": pd, "sm": sm, "t": t,
                 "predictor_names": list(PREDICTORS),
                 "generating_values": pd.Series(0.0, index=PREDICTORS)}
    found = set()
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for node in ast.parse("".join(cell["source"])).body:
            if isinstance(node, ast.FunctionDef) and node.name in wanted:
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             "NB6", "exec"), namespace)
                found.add(node.name)
    assert found == wanted
    return namespace


class ExtractionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb6 = nb6_functions()

    def check_equivalence(self, data):
        before = data.copy(deep=True)
        original = self.nb6["fit_simulated_with_intervals"](data)
        result = fit_intervals(data, PREDICTORS)
        np.testing.assert_allclose(result["summary"].loc[list(PREDICTORS)],
                                   original, rtol=1e-10, atol=1e-12)
        self.assertEqual(tuple(result["summary"].index), tuple(result["covariance"].index))
        self.assertEqual(tuple(result["summary"].index), tuple(result["covariance"].columns))
        np.testing.assert_allclose(np.diag(result["covariance"]),
                                   result["summary"].SE**2, rtol=1e-10, atol=1e-12)
        pd.testing.assert_frame_equal(data, before)
        return result

    def test_null_and_nonzero(self):
        for theta in [0.0, 0.001]:
            with self.subTest(theta=theta):
                self.check_equivalence(self.nb6["simulate_dataset"](
                    2026112001, n_plays=40, theta_W=theta))

    def test_missing_interior_rating_unbalanced_panel(self):
        data = self.nb6["simulate_dataset"](2026112002, n_plays=40, theta_W=0.001)
        # Example policy solely for this extraction test: losing rating H_5
        # invalidates interval 5's outcome and interval 6's prior rating.
        # Remove both rows without merging their rewards or renumbering positions.
        data = data.loc[~((data.play_id == 1) & data.interval.isin([5, 6]))].copy()
        mean = data.groupby("play_id").prior_happiness.transform("mean")
        data["mood_within"] = data.prior_happiness - mean
        data["mood_between"] = mean - 50
        data["within_x_RPE"] = data.mood_within * data.RPE
        data["between_x_RPE"] = data.mood_between * data.RPE
        result = self.check_equivalence(data)
        self.assertEqual(len(data), 438)
        self.assertEqual(data.play_id.nunique(), 40)
        # Row order/nonconsecutive index should not change estimates or covariance.
        reordered = fit_intervals(data.sample(frac=1, random_state=7), PREDICTORS)
        np.testing.assert_allclose(result["summary"], reordered["summary"],
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result["covariance"], reordered["covariance"],
                                   rtol=1e-10, atol=1e-12)

    def test_covariance_against_explicit_fixed_effects(self):
        data = self.nb6["simulate_dataset"](2026112003, n_plays=30, theta_W=0.001)
        result = fit_intervals(data, PREDICTORS)
        positions = pd.get_dummies(data.interval, prefix="position", drop_first=True, dtype=float)
        design = pd.concat([data[list(PREDICTORS)], positions], axis=1)
        intercepts = pd.get_dummies(data.play_id, prefix="person", dtype=float)
        fit = sm.OLS(data.next_happiness, pd.concat([design, intercepts], axis=1)).fit(
            cov_type="cluster", cov_kwds={"groups": data.play_id, "use_correction": False})
        names = list(result["summary"].index)
        n, n_plays = len(data), data.play_id.nunique()
        rank = len(names)
        correction = n_plays / (n_plays - 1) * (n - 1) / (n - rank)
        np.testing.assert_allclose(result["summary"].estimate, fit.params.loc[names],
                                   rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(result["covariance"],
            fit.cov_params().loc[names, names] * correction,
            rtol=1e-10, atol=1e-12)

    def test_missing_values_are_not_silently_dropped(self):
        data = self.nb6["simulate_dataset"](2026112004, n_plays=10)
        data.loc[0, "next_happiness"] = np.nan
        with self.assertRaisesRegex(AssertionError, "Required values are missing"):
            fit_intervals(data, PREDICTORS)


if __name__ == "__main__":
    unittest.main()
