"""Grow the tiny Decision Tree and pack its questions for the browser.

Think of the CSV as a pile of pretend patient cards. The tree learns a short
yes/no guessing game that sorts those cards into two toy-colored baskets. The
labels and measurements are synthetic: this code is never a real triage tool.
"""

# Path creates dependable locations relative to this file.
from pathlib import Path

# Import our hand-written tree and its checked CSV loader.
from decision_tree_model import SimpleDecisionTree, load_dataset


# Find this project's folder even when the command starts somewhere else.
PROJECT_DIR = Path(__file__).resolve().parent
# This table contains pretend blood-pressure, heart-rate, and answer rows.
DATASET_PATH = PROJECT_DIR / "data" / "cardiac_triage.csv"
# This JSON file will carry the learned question boxes to JavaScript.
MODEL_PATH = PROJECT_DIR / "dist" / "model.json"


def main() -> None:
    """Run the complete read → grow → export → explain training recipe."""

    # Turn each CSV row into a checked TrainingExample card.
    examples = load_dataset(DATASET_PATH)
    # max_depth=2 permits at most two questions from root to answer. fit()
    # searches for the cleanest cuts and returns the trained tree itself.
    tree = SimpleDecisionTree(max_depth=2).fit(examples)
    # Save the nested questions, leaves, labels, and graph points as JSON.
    tree.export_json(MODEL_PATH)

    # Print useful, inspectable facts instead of hiding training behind a spinner.
    print(f"Loaded {len(examples)} synthetic teaching examples.")
    print(f"Training accuracy: {tree.training_accuracy() * 100:.1f}%")
    print("\nLearned tree:\n")
    # Convert nested nodes into ordinary IF / ELSE sentences.
    print(tree.rules_as_text())
    print(f"\nSaved browser model to: {MODEL_PATH}")
    # Repeat the safety boundary wherever a learner runs the model.
    print("Warning: programming lesson only; never use for real ER decisions.")


# This guard runs main only for ``python train_model.py``, not during imports.
if __name__ == "__main__":
    main()
