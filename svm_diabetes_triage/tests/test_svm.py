"""Small tests for the Python SVM. Run with unittest; no packages needed."""

import json
import tempfile
import unittest
from pathlib import Path

from svm_model import HIGHER_ZONE, LOWER_ZONE, MARGIN_ZONE, SimpleLinearSVM, load_dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]


class SimpleSvmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_dataset(PROJECT_DIR / "data" / "diabetes_triage.csv")
        cls.model = SimpleLinearSVM().fit(cls.examples)

    def test_dataset_and_training(self) -> None:
        self.assertEqual(len(self.examples), 40)
        self.assertGreaterEqual(self.model.training_accuracy(), 0.95)

    def test_three_teaching_zones(self) -> None:
        self.assertEqual(self.model.predict(85, 24)["zone"], LOWER_ZONE)
        self.assertEqual(self.model.predict(112, 29)["zone"], MARGIN_ZONE)
        self.assertEqual(self.model.predict(150, 33)["zone"], HIGHER_ZONE)

    def test_bad_measurements_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            self.model.predict(0, 25)

    def test_browser_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "model.json"
            self.model.export_json(output_path)
            exported = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(exported["model_type"], "linear_svm")
        self.assertEqual(len(exported["weights"]), 2)
        self.assertEqual(len(exported["training_examples"]), 40)
        self.assertEqual(
            sum(row["is_support_example"] for row in exported["training_examples"]),
            6,
        )


if __name__ == "__main__":
    unittest.main()
