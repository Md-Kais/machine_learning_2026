"""Small tests that use only Python's built-in unittest module."""

import json
import tempfile
import unittest
from pathlib import Path

from knn_model import SimpleKNN, calculate_bmi, load_dataset


PROJECT_FOLDER = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_FOLDER / "data" / "bmi_data.csv"


class BmiKnnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = load_dataset(DATA_FILE)
        cls.model = SimpleKNN(k=5)
        cls.model.fit(cls.examples)

    def test_bmi_formula(self) -> None:
        self.assertAlmostEqual(calculate_bmi(170, 65), 22.49, places=2)

    def test_example_categories(self) -> None:
        cases = [
            (170, 50, "Underweight"),
            (170, 65, "Normal"),
            (170, 80, "Overweight"),
            (170, 100, "Obese"),
        ]

        for height, weight, expected in cases:
            with self.subTest(height=height, weight=weight):
                self.assertEqual(self.model.predict(height, weight)["category"], expected)

    def test_bad_measurements_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            calculate_bmi(0, 65)

    def test_browser_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_folder:
            output = Path(temp_folder) / "model.json"
            self.model.export_for_browser(output)
            self.assertTrue(output.exists())
            exported = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exported["k"], 5)
            self.assertEqual(len(exported["training_examples"]), len(self.examples))
            first_example = exported["training_examples"][0]
            self.assertIn("height_cm", first_example)
            self.assertIn("weight_kg", first_example)


if __name__ == "__main__":
    unittest.main()
