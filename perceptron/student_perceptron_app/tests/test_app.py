"""Regression tests for the perceptron algorithm and Flask JSON contract."""

from __future__ import annotations

import unittest

import numpy as np

from app import app
from perceptron import Perceptron


SEPARABLE_POINTS = [
    {"p1": 1.0, "p2": 2.0, "target": 0},
    {"p1": 2.0, "p2": 1.0, "target": 0},
    {"p1": 3.0, "p2": 3.0, "target": 0},
    {"p1": 6.0, "p2": 6.0, "target": 1},
    {"p1": 8.0, "p2": 7.0, "target": 1},
]

XOR_POINTS = [
    {"p1": 2.0, "p2": 2.0, "target": 0},
    {"p1": 8.0, "p2": 8.0, "target": 0},
    {"p1": 2.0, "p2": 8.0, "target": 1},
    {"p1": 8.0, "p2": 2.0, "target": 1},
]


class PerceptronTests(unittest.TestCase):
    """Check the mathematical behavior independently of Flask."""

    def test_hardlim_includes_zero_in_class_one(self) -> None:
        np.testing.assert_array_equal(
            Perceptron.hardlim([-2.0, -0.01, 0.0, 3.0]),
            np.asarray([0, 0, 1, 1]),
        )

    def test_train_step_applies_unified_rule(self) -> None:
        model = Perceptron()
        diagnostic = model.train_step([2.0, 3.0], 0)

        self.assertEqual(diagnostic["a"], 1)
        self.assertEqual(diagnostic["e"], -1)
        self.assertTrue(diagnostic["updated"])
        np.testing.assert_allclose(model.weights, [-2.0, -3.0])
        self.assertEqual(model.bias, -1.0)

    def test_predict_is_vectorized(self) -> None:
        model = Perceptron()
        model.weights = np.asarray([1.0, 1.0])
        model.bias = -5.0

        predictions = model.predict([[1.0, 1.0], [4.0, 3.0], [2.5, 2.5]])
        np.testing.assert_array_equal(predictions, np.asarray([0, 1, 1]))

    def test_fit_converges_for_separable_points(self) -> None:
        model = Perceptron()
        samples = [[point["p1"], point["p2"]] for point in SEPARABLE_POINTS]
        targets = [point["target"] for point in SEPARABLE_POINTS]

        result = model.fit(samples, targets, max_epochs=100)

        self.assertTrue(result["converged"])
        self.assertEqual(result["final"]["accuracy"], 1.0)
        self.assertGreater(result["total_updates"], 0)
        self.assertEqual(len(result["history"]), result["total_steps"])

    def test_fit_stops_for_xor(self) -> None:
        model = Perceptron()
        samples = [[point["p1"], point["p2"]] for point in XOR_POINTS]
        targets = [point["target"] for point in XOR_POINTS]

        result = model.fit(samples, targets, max_epochs=12)

        self.assertFalse(result["converged"])
        self.assertEqual(result["total_epochs"], 12)


class ApiTests(unittest.TestCase):
    """Exercise the public routes through Flask's in-process test client."""

    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_index_and_health_are_available(self) -> None:
        index_response = self.client.get("/")
        health_response = self.client.get("/api/health")

        self.assertEqual(index_response.status_code, 200)
        self.assertIn(b"Student Perceptron Lab", index_response.data)
        self.assertEqual(health_response.get_json()["service"], "student-perceptron")

    def test_train_rejects_an_empty_dataset(self) -> None:
        response = self.client.post("/api/train", json={"points": []})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "empty_dataset")

    def test_train_rejects_a_single_class(self) -> None:
        response = self.client.post(
            "/api/train",
            json={"points": [{"p1": 7, "p2": 8, "target": 1}]},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "single_class_dataset")

    def test_train_rejects_conflicting_labels(self) -> None:
        response = self.client.post(
            "/api/train",
            json={
                "points": [
                    {"p1": 5, "p2": 5, "target": 1},
                    {"p1": 5, "p2": 5, "target": 0},
                ]
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.get_json()["error"]["code"], "conflicting_labels")

    def test_train_returns_animation_history_and_boundary(self) -> None:
        response = self.client.post("/api/train", json={"points": SEPARABLE_POINTS})
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(body["converged"])
        self.assertEqual(body["status"], "converged")
        self.assertEqual(body["final"]["accuracy"], 1.0)
        self.assertIn("initial_state", body)
        self.assertIn("history", body)
        self.assertIn("epoch_history", body)
        self.assertIn(body["boundary"]["type"], {"vertical", "slope_intercept"})

    def test_xor_returns_a_bounded_nonconvergence_result(self) -> None:
        response = self.client.post(
            "/api/train",
            json={"points": XOR_POINTS, "max_epochs": 8},
        )
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertFalse(body["converged"])
        self.assertEqual(body["status"], "not_converged")
        self.assertEqual(body["total_epochs"], 8)

    def test_predict_accepts_explicit_parameters(self) -> None:
        response = self.client.post(
            "/api/predict",
            json={"p1": 8, "p2": 7, "weights": [1, 1], "bias": -10},
        )
        body = response.get_json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["prediction"], 1)
        self.assertEqual(body["label"], "Pass")
        self.assertAlmostEqual(body["net_input"], 5.0)


if __name__ == "__main__":
    unittest.main()
