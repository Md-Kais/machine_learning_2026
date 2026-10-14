"""Load the PDF's tennis problem, train ID3, and export its browser model."""
import argparse
from pathlib import Path
from decision_tree_model import ID3DecisionTree, load_dataset

PROJECT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Train categorical ID3 on Play Tennis.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_DIR / "data/play_tennis.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "dist/model.json")
    parser.add_argument("--max-depth", type=int, default=None, help="Root depth is 0; omit for full ID3.")
    parser.add_argument("--min-samples-split", type=int, default=2)
    parser.add_argument("--min-gain", type=float, default=0.0, help="Minimum entropy reduction in bits.")
    args = parser.parse_args()
    model = ID3DecisionTree(args.max_depth, args.min_samples_split, args.min_gain).fit(load_dataset(args.dataset))
    model.export_json(args.output)
    print(f"Loaded {len(model.examples)} tennis records. Training accuracy: {model.training_accuracy():.1%}")
    print("Root information gains:", model.tree["gains"])
    example = {"Outlook": "Sunny", "Temp": "Mild", "Humidity": "High", "Wind": "Strong"}
    print("PDF question:", example, "->", model.predict(example))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
