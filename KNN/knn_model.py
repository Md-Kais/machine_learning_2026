"""A tiny K-Nearest Neighbors (KNN) model that shows all of its work.

Think of BMI values as dots on a long strip of graph paper. To label a new dot,
KNN measures the gap to every labeled dot with a ruler, invites the five closest
dots to vote, and uses the most popular label. The inputs are height in
centimetres and weight in kilograms; the output is one BMI category.
"""

# Postponing type-hint evaluation lets classes refer to types defined later.
from __future__ import annotations

# csv reads the table, json packs the browser model, and Counter counts votes.
import csv
import json
from collections import Counter
# dataclass makes one tidy, named container for each example row.
from dataclasses import dataclass
from pathlib import Path


# Only these four answer labels are allowed in the teaching dataset.
VALID_CATEGORIES = ("Underweight", "Normal", "Overweight", "Obese")


@dataclass(frozen=True)
class BmiExample:
    """One unchangeable flash card from the CSV table.

    A card receives height (cm), weight (kg), calculated BMI, and the known word
    label. For example: 170, 65, about 22.5, and ``Normal``.
    """

    height_cm: float
    weight_kg: float
    bmi: float
    category: str


def calculate_bmi(height_cm: float, weight_kg: float) -> float:
    """Turn centimetres and kilograms into one BMI number.

    The formula is ``weight / height²``. Squaring height is like covering a
    floor tile whose sides are both the person's height. A 170 cm, 65 kg input
    becomes ``65 / (1.70 × 1.70) = 22.49``.
    """

    # Zero or negative body measurements cannot describe a person and would
    # also make the division unsafe.
    if height_cm <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must both be greater than zero.")

    # The BMI recipe uses metres, so 170 cm becomes 1.70 m.
    height_m = height_cm / 100
    # Multiply height by itself, then divide weight by that square.
    return weight_kg / (height_m * height_m)


def load_dataset(csv_path: str | Path) -> list[BmiExample]:
    """Read, check, and convert all CSV rows into BmiExample cards."""

    # This basket will collect the clean example cards.
    examples: list[BmiExample] = []

    # Open text as UTF-8; newline="" lets the csv module handle row endings.
    with Path(csv_path).open(encoding="utf-8", newline="") as csv_file:
        # DictReader names each cell using the header at the top of its column.
        for row in csv.DictReader(csv_file):
            # Convert numeric text such as "170" into a number the model can use.
            height = float(row["height_cm"])
            weight = float(row["weight_kg"])
            # strip removes accidental spaces around the answer word.
            category = row["category"].strip()

            # Reject a misspelled or invented answer instead of learning it.
            if category not in VALID_CATEGORIES:
                raise ValueError(f"Unknown category in dataset: {category}")

            # Build one card. BMI is calculated from its height and weight so
            # the saved number always agrees with the visible measurements.
            examples.append(
                BmiExample(
                    height_cm=height,
                    weight_kg=weight,
                    bmi=calculate_bmi(height, weight),
                    category=category,
                )
            )

    # An empty notebook gives the neighbors nothing to vote about.
    if not examples:
        raise ValueError("The dataset is empty.")

    # Return the complete checked notebook.
    return examples


class SimpleKNN:
    """A from-scratch KNN classifier that measures distance along the BMI line."""

    def __init__(self, k: int = 5) -> None:
        # An odd k reduces tied votes; k must also be at least 1.
        if k <= 0 or k % 2 == 0:
            raise ValueError("k must be a positive odd number, such as 3 or 5.")
        # Remember how many neighbors get a ballot.
        self.k = k
        # Before fit(), the model's notebook is empty.
        self.examples: list[BmiExample] = []

    def fit(self, examples: list[BmiExample]) -> None:
        """Train by copying the labeled examples into the model's memory."""

        # We need at least k cards to invite k different neighbors.
        if len(examples) < self.k:
            raise ValueError("The dataset must contain at least k examples.")
        # list(...) makes our own basket rather than borrowing the caller's one.
        self.examples = list(examples)

    def nearest_neighbors(self, bmi: float) -> list[tuple[float, BmiExample]]:
        """Return ``(ruler gap, card)`` pairs for the k closest BMI dots."""

        # Predicting before studying any cards would be a guessing game.
        if not self.examples:
            raise RuntimeError("Call fit() before predict().")

        # abs removes the minus sign: the gap from 22 to 25 is 3 whichever dot
        # is written first. This is one-dimensional Euclidean distance—a ruler
        # measurement between two dots on a straight number line.
        distances = [(abs(example.bmi - bmi), example) for example in self.examples]
        # Put the smallest ruler gap first.
        distances.sort(key=lambda item: item[0])
        # Keep exactly k pairs; with k=5, items 0 through 4 get to vote.
        return distances[: self.k]

    def predict_bmi(self, bmi: float) -> str:
        """Let the nearest cards vote and return the winning category word."""

        # Ask the ruler helper for the k nearest cards.
        neighbors = self.nearest_neighbors(bmi)
        # Counter turns labels such as Normal, Normal, Overweight into vote totals.
        vote_counts = Counter(example.category for _, example in neighbors)

        # Keep a second scoreboard for total distance. It breaks a rare tie by
        # choosing the tied team whose voters stand closest to the new dot.
        distance_totals: dict[str, float] = {}
        for distance, example in neighbors:
            distance_totals[example.category] = (
                # Start a category at zero, then add this voter's ruler gap.
                distance_totals.get(example.category, 0.0) + distance
            )

        # min chooses the smallest comparison key. A negative vote count makes
        # 3 votes (-3) beat 2 votes (-2); total distance settles equal counts.
        return min(
            vote_counts,
            key=lambda category: (-vote_counts[category], distance_totals[category]),
        )

    def predict(self, height_cm: float, weight_kg: float) -> dict[str, object]:
        """Return the BMI, winning word, and five inspectable neighbor cards."""

        # Two measurements go in; one BMI coordinate comes out.
        bmi = calculate_bmi(height_cm, weight_kg)
        # Save the same neighbors that will explain the result on the web page.
        neighbors = self.nearest_neighbors(bmi)
        # Run the vote.
        category = self.predict_bmi(bmi)

        # A dictionary is convenient for printing or turning into JSON.
        return {
            # One decimal is friendly to read: 22.491... becomes 22.5.
            "bmi": round(bmi, 1),
            "category": category,
            "neighbors": [
                {
                    "height_cm": example.height_cm,
                    "weight_kg": example.weight_kg,
                    "bmi": round(example.bmi, 1),
                    "category": example.category,
                    "distance": round(distance, 2),
                }
                for distance, example in neighbors
            ],
        }

    def export_for_browser(self, output_path: str | Path) -> None:
        """Pack k and every remembered example into a browser-readable JSON file."""

        # An empty model has learned nothing worth exporting.
        if not self.examples:
            raise RuntimeError("Call fit() before export_for_browser().")

        # KNN's learned "model" really is its labeled notebook plus k.
        browser_model = {
            "name": "Simple BMI KNN",
            "k": self.k,
            "feature": "bmi",
            "training_examples": [
                {
                    "height_cm": example.height_cm,
                    "weight_kg": example.weight_kg,
                    "bmi": round(example.bmi, 3),
                    "category": example.category,
                }
                for example in self.examples
            ],
        }

        # Create the dist folder if it is missing.
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        # indent=2 makes the JSON look like a tidy notebook instead of one line.
        output.write_text(json.dumps(browser_model, indent=2), encoding="utf-8")
