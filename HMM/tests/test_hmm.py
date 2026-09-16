"""Correctness checks for the beginner-friendly HMM implementation."""

from __future__ import annotations

import itertools
import math
import sys
import unittest
from pathlib import Path


# The explained Python model is kept beside its offline JavaScript twin. Add the
# dist directory to Python's import path for normal desktop correctness tests.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dist"))

from hmm import STATES, analyse, make_model  # noqa: E402


def path_probability(path: tuple[str, ...], observations: list[str], model: dict) -> float:
    """Calculate one path directly; useful as a simple independent reference."""

    probability = model["initial"][path[0]] * model["emissions"][path[0]][observations[0]]
    for index in range(1, len(observations)):
        probability *= model["transitions"][path[index - 1]][path[index]]
        probability *= model["emissions"][path[index]][observations[index]]
    return probability


class HiddenCoinTests(unittest.TestCase):
    def test_probability_rows_sum_to_one(self) -> None:
        model = make_model()
        self.assertAlmostEqual(sum(model["initial"].values()), 1.0)
        for row in model["transitions"].values():
            self.assertAlmostEqual(sum(row.values()), 1.0)
        for row in model["emissions"].values():
            self.assertAlmostEqual(sum(row.values()), 1.0)

    def test_viterbi_matches_exhaustive_search(self) -> None:
        """For nine tosses we can enumerate all 2^9 paths as a reference."""

        result = analyse("H H H H H T H T T")
        observations = result["observations"]
        all_paths = itertools.product(STATES, repeat=len(observations))
        expected_path = max(
            all_paths,
            key=lambda candidate: path_probability(candidate, observations, result["model"]),
        )
        expected_probability = path_probability(expected_path, observations, result["model"])

        self.assertEqual(result["path"], list(expected_path))
        self.assertTrue(math.isclose(result["path_probability"], expected_probability, rel_tol=1e-12))

    def test_forward_probability_matches_sum_of_all_paths(self) -> None:
        result = analyse("H T H T")
        observations = result["observations"]
        expected_probability = sum(
            path_probability(path, observations, result["model"])
            for path in itertools.product(STATES, repeat=len(observations))
        )
        self.assertTrue(
            math.isclose(result["sequence_probability"], expected_probability, rel_tol=1e-12)
        )

    def test_each_posterior_column_is_a_probability_distribution(self) -> None:
        result = analyse("HHTHTT")
        for column in result["posterior"]:
            self.assertAlmostEqual(sum(column.values()), 1.0)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in column.values()))

    def test_invalid_observation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "H .* or T"):
            analyse("H X T")


if __name__ == "__main__":
    unittest.main(verbosity=2)
