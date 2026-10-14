"""Validate the lecture dataset, ID3 calculations, paths, and stopping rules."""
import json
import tempfile
import unittest
from pathlib import Path
from decision_tree_model import ID3DecisionTree, entropy, information_gain, load_dataset

ROOT = Path(__file__).resolve().parents[1]


class ID3Tests(unittest.TestCase):
    def setUp(self):
        self.rows = load_dataset(ROOT / "data/play_tennis.csv")
        self.model = ID3DecisionTree().fit(self.rows)

    def test_lecture_entropy_and_gains(self):
        self.assertEqual(len(self.rows), 14)
        self.assertAlmostEqual(entropy(row["Play"] for row in self.rows), 0.9402859587)
        expected = {"Outlook": 0.2467498198, "Temp": 0.0292225657,
                    "Humidity": 0.1518355014, "Wind": 0.0481270304}
        for feature, gain in expected.items():
            self.assertAlmostEqual(information_gain(self.rows, feature), gain)

    def test_pdf_tree_and_question(self):
        root = self.model.tree
        self.assertEqual(root["feature"], "Outlook")
        self.assertEqual(set(root["children"]), {"Sunny", "Overcast", "Rain"})
        self.assertEqual(root["children"]["Sunny"]["feature"], "Humidity")
        self.assertEqual(root["children"]["Rain"]["feature"], "Wind")
        self.assertEqual(root["children"]["Overcast"]["prediction"], "Yes")
        question = dict(Outlook="Sunny", Temp="Mild", Humidity="High", Wind="Strong")
        result = self.model.predict(question)
        self.assertEqual(result["label"], "No")
        self.assertEqual([(s["feature"], s["value"]) for s in result["path"]], [("Outlook", "Sunny"), ("Humidity", "High")])
        self.assertEqual(self.model.training_accuracy(), 1.0)

    def test_parameter_changes_alter_tree(self):
        stump = ID3DecisionTree(max_depth=0).fit(self.rows)
        self.assertTrue(stump.tree["is_leaf"])
        self.assertEqual(stump.tree["prediction"], "Yes")
        self.assertTrue(ID3DecisionTree(min_gain=1).fit(self.rows).tree["is_leaf"])
        self.assertTrue(ID3DecisionTree(min_samples_split=15).fit(self.rows).tree["is_leaf"])
        one_level = ID3DecisionTree(max_depth=1).fit(self.rows)
        self.assertTrue(all(child["is_leaf"] for child in one_level.tree["children"].values()))

    def test_unseen_valid_branch_uses_node_majority(self):
        limited = ID3DecisionTree().fit([row for row in self.rows if row["Outlook"] != "Overcast"])
        result = limited.predict(dict(Outlook="Overcast", Temp="Hot", Humidity="High", Wind="Weak"))
        self.assertTrue(result["fallback"])
        self.assertEqual(result["label"], limited.tree["prediction"])
        with self.assertRaises(ValueError):
            self.model.predict(dict(Outlook="Snow", Temp="Mild", Humidity="High", Wind="Strong"))

    def test_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "model.json"
            self.model.export_json(p)
            package = json.loads(p.read_text())
        self.assertEqual(package["model_type"], "id3")
        self.assertEqual(package["criterion"], "entropy_information_gain")
        self.assertEqual(len(package["training_examples"]), 14)

    def test_invalid_parameters(self):
        for params in [dict(max_depth=-1), dict(min_samples_split=1), dict(min_gain=float("nan"))]:
            with self.assertRaises(ValueError):
                ID3DecisionTree(**params)


if __name__ == "__main__":
    unittest.main()
