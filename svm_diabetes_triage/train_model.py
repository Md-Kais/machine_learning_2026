"""Train the tiny straight-line SVM and pack its learned numbers for the site.

Imagine two teams of dots on a playground. The SVM learns how to draw a wide,
straight road between them. This script loads the pretend teaching dots, nudges
the road many times, checks which side they occupy, and saves the final tilt and
position to JSON. It teaches programming; it does not diagnose diabetes.
"""

# Path builds locations relative to this project instead of the current terminal.
from pathlib import Path

# Import our visible, from-scratch SVM and CSV checker.
from svm_model import SimpleLinearSVM, load_dataset


# Locate the project, its input table, and its browser-ready output.
PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "diabetes_triage.csv"
MODEL_PATH = PROJECT_DIR / "dist" / "model.json"


def main() -> None:
    """Run the read → learn → check → export recipe."""

    # Read fasting-glucose, BMI, and known outside-group labels from the CSV.
    examples = load_dataset(DATASET_PATH)
    # Start with an untrained road, nudge it using every dot, and keep the result.
    model = SimpleLinearSVM().fit(examples)
    # Save means, scales, weights, bias, margins, and graph points for JavaScript.
    model.export_json(MODEL_PATH)

    # Print enough facts for a learner to inspect what happened.
    print(f"Loaded {len(examples)} teaching examples.")
    print(f"Training-side accuracy: {model.training_accuracy() * 100:.1f}%")
    # A weight describes how strongly one scaled measurement tilts the road.
    print(f"Learned weights: glucose={model.weights[0]:.3f}, BMI={model.weights[1]:.3f}")
    print(f"Saved browser model to: {MODEL_PATH}")
    # The third, middle UI label is made from the SVM road itself, not a third
    # group of training examples.
    print("The middle SVM margin is used as the follow-up teaching zone.")


# Run only when a learner enters ``python train_model.py``.
if __name__ == "__main__":
    main()
