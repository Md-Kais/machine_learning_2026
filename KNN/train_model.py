"""Train, check, and pack the tiny BMI KNN model for the web page.

Imagine a class notebook filled with example people. This script first hides
some pages so the computer cannot peek at the answers during its practice test.
After the test, it lets KNN remember every page and packs the notebook into
``dist/model.json`` so JavaScript can use it in the browser.

Run this file whenever ``data/bmi_data.csv`` changes:

    python train_model.py
"""

# Path helps us build file locations that work on Windows, macOS, and Linux.
from pathlib import Path

# These are the example-row shape, the KNN brain, and the CSV reader that we
# wrote ourselves in knn_model.py. No hidden machine-learning library is used.
from knn_model import BmiExample, SimpleKNN, load_dataset


# __file__ means "this Python file"; .parent means its containing KNN folder.
PROJECT_FOLDER = Path(__file__).parent
# The input is a table of height, weight, and known category examples.
DATA_FILE = PROJECT_FOLDER / "data" / "bmi_data.csv"
# The output is a JSON notebook that a browser can read.
MODEL_FILE = PROJECT_FOLDER / "dist" / "model.json"


def split_for_testing(
    examples: list[BmiExample],
) -> tuple[list[BmiExample], list[BmiExample]]:
    """Place every fifth row in a test pile and the other rows in a study pile.

    Input: a list such as 40 labeled example people.
    Output: two lists—32 study examples and 8 unseen test examples in that case.
    This is like covering every fifth flash card before checking what we learned.
    """

    # Start with two empty baskets.
    training: list[BmiExample] = []
    testing: list[BmiExample] = []

    # enumerate gives us both a row number (index) and the example on that row.
    for index, example in enumerate(examples):
        # A remainder of 0 means row 0, 5, 10, 15, and so on: every fifth row.
        if index % 5 == 0:
            # Hide this example in the test basket.
            testing.append(example)
        else:
            # Let KNN study this example.
            training.append(example)

    # Give both baskets back to the caller in a fixed order.
    return training, testing


def measure_accuracy(training: list[BmiExample], testing: list[BmiExample]) -> float:
    """Return the share of hidden cards that KNN labels correctly.

    Input: a study pile and a hidden-answer test pile.
    Output: a number from 0 to 1; for example, 0.875 means 87.5% correct.
    """

    # k=5 tells KNN to ask the five closest BMI neighbors to vote.
    test_model = SimpleKNN(k=5)
    # "Training" KNN simply means remembering the labeled study cards.
    test_model.fit(training)
    # Each True answer counts as 1 and each False answer counts as 0.
    correct = sum(
        # Predict from the BMI number, then compare with the covered answer.
        test_model.predict_bmi(example.bmi) == example.category
        for example in testing
    )
    # correct / total changes a count such as 7 of 8 into 0.875.
    return correct / len(testing)


def main() -> None:
    """Run the complete load → test → retrain → export recipe."""

    # Turn every CSV row into a checked BmiExample object.
    examples = load_dataset(DATA_FILE)
    # Make separate study and test piles.
    training, testing = split_for_testing(examples)
    # Ask how well five-neighbor voting works on the hidden pile.
    accuracy = measure_accuracy(training, testing)

    # The test is finished, so build the final model using every available card.
    final_model = SimpleKNN(k=5)
    # KNN stores the examples; it does not squeeze them into a magic formula.
    final_model.fit(examples)
    # Save k and all remembered examples as numbers and words in JSON.
    final_model.export_for_browser(MODEL_FILE)

    # These messages tell a learner what went in, how the check went, and where
    # the browser-ready output went.
    print(f"Loaded {len(examples)} dataset rows.")
    print(f"Test accuracy: {accuracy:.0%} ({len(testing)} test rows)")
    print(f"Browser model saved to: {MODEL_FILE}")


# Python sets __name__ to "__main__" only when this file is run directly.
if __name__ == "__main__":
    # This guard prevents training from starting merely because another file
    # imports one of the helper functions above.
    main()
