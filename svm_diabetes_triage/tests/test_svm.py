"""Verify binary decisions, dual constraints, supports, and SVM geometry."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from svm_model import SimpleLinearSVM, TrainingExample, load_dataset, LOWER_LABEL, HIGHER_LABEL

ROOT = Path(__file__).resolve().parents[1]


class SVMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = load_dataset(ROOT / "data/diabetes_triage.csv")
        cls.model = SimpleLinearSVM().fit(cls.rows)

    def test_fit_converges_and_classifies_training_data(self):
        self.assertEqual(len(self.rows), 40)
        self.assertEqual(self.model.training_accuracy(), 1)
        self.assertTrue(self.model.converged)
        self.assertLessEqual(self.model.kkt_violation, self.model.tolerance*2)

    def test_binary_decisions_inside_margin(self):
        self.assertEqual(self.model.predict(85,24)["numeric_label"], -1)
        self.assertEqual(self.model.predict(150,33)["numeric_label"], 1)
        bmi = 29
        z2 = (bmi-self.model.means[1])/self.model.scales[1]
        for target in (-0.5,0.5):
            glucose = self.model.means[0] + self.model.scales[0]*(target-self.model.bias-self.model.weights[1]*z2)/self.model.weights[0]
            result = self.model.predict(glucose,bmi)
            self.assertTrue(result["inside_margin"])
            self.assertEqual(result["numeric_label"], 1 if target > 0 else -1)
        self.assertIn("+1", self.model.zone_from_score(0))

    def test_dual_constraints_and_actual_support_vectors(self):
        self.assertAlmostEqual(sum(a*row.numeric_label for a,row in zip(self.model.alphas,self.rows)), 0, places=8)
        self.assertTrue(all(-1e-8 <= a <= self.model.C+1e-8 for a in self.model.alphas))
        supports = self.model.support_indices()
        self.assertTrue(supports)
        self.assertEqual(supports, {i for i,a in enumerate(self.model.alphas) if a > 1e-8})
        for i in supports:
            if self.model.alphas[i] < self.model.C-1e-8:
                self.assertAlmostEqual(self.rows[i].numeric_label*self.model.decision_score(*self.rows[i].features), 1, delta=0.0002)

    def test_known_maximum_margin_solution(self):
        # Two parallel groups: centered/scaled x = +/-1, so w=[1,0], b=0.
        rows = [TrainingExample(1,1,LOWER_LABEL), TrainingExample(1,3,LOWER_LABEL),
                TrainingExample(3,1,HIGHER_LABEL), TrainingExample(3,3,HIGHER_LABEL)]
        model = SimpleLinearSVM(C=10).fit(rows)
        self.assertAlmostEqual(model.weights[0], 1, places=4)
        self.assertAlmostEqual(model.weights[1], 0, places=4)
        self.assertAlmostEqual(model.bias, 0, places=4)
        self.assertAlmostEqual(2/math.hypot(*model.weights), 2, places=4)

    def test_C_changes_margin(self):
        loose = SimpleLinearSVM(C=0.01).fit(self.rows)
        self.assertLess(math.hypot(*loose.weights), math.hypot(*self.model.weights))

    def test_invalid_inputs_and_parameters(self):
        for glucose,bmi in [(0,25),(float("nan"),25),(100,float("inf"))]:
            with self.assertRaises(ValueError):
                self.model.predict(glucose,bmi)
        for params in [dict(C=0),dict(tolerance=float("nan")),dict(max_passes=0),dict(max_iterations=1.5)]:
            with self.assertRaises(ValueError):
                SimpleLinearSVM(**params)
        with self.assertRaises(ValueError):
            SimpleLinearSVM().fit([TrainingExample(1,2,LOWER_LABEL)]*4)

    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"model.json"
            self.model.export_json(p)
            package=json.loads(p.read_text())
        self.assertEqual(package["kernel"], "linear")
        self.assertEqual(package["solver"], "SMO")
        self.assertEqual(sum(row["is_support_example"] for row in package["training_examples"]),len(self.model.support_indices()))
        self.assertAlmostEqual(package["margin_width"], 2/math.hypot(*self.model.weights))


if __name__ == "__main__":
    unittest.main()
