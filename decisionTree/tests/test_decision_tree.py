"""Tests for the pure-Python Decision Tree."""

import json
import tempfile
import unittest
from pathlib import Path

from decision_tree_model import HIGH_LABEL, LOW_LABEL, SimpleDecisionTree, load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]


class DecisionTreeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_dataset(PROJECT_DIR / "data" / "cardiac_triage.csv")
        cls.tree = SimpleDecisionTree(max_depth=2).fit(cls.examples)

    def test_dataset_and_training(self) -> None:
        self.assertEqual(len(self.examples), 48)
        self.assertEqual(self.tree.training_accuracy(), 1.0)

    def test_learned_axis_aligned_questions(self) -> None:
        self.assertEqual(self.tree.root.feature, "systolic_bp_mm_hg")
        self.assertEqual(self.tree.root.threshold, 140)
        self.assertEqual(self.tree.root.right.feature, "max_heart_rate_bpm")
        self.assertEqual(self.tree.root.right.threshold, 115)

    def test_three_paths(self) -> None:
        left_leaf = self.tree.predict(130, 150)
        lower_right_leaf = self.tree.predict(155, 100)
        upper_right_leaf = self.tree.predict(155, 130)

        self.assertEqual(left_leaf["label"], LOW_LABEL)
        self.assertEqual(lower_right_leaf["label"], LOW_LABEL)
        self.assertEqual(upper_right_leaf["label"], HIGH_LABEL)
        self.assertEqual(len(left_leaf["path"]), 1)
        self.assertEqual(len(upper_right_leaf["path"]), 2)

    def test_bad_measurements_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.tree.predict(0, 100)

    def test_browser_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.json"
            self.tree.export_json(output_path)
            model = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(model["model_type"], "decision_tree_classifier")
        self.assertEqual(model["tree"]["feature"], "systolic_bp_mm_hg")
        self.assertEqual(len(model["training_examples"]), 48)
        self.assertIn("Never use", model["warning"])


if __name__ == "__main__":
    unittest.main()
