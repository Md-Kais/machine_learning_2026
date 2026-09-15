"""A tiny, from-scratch linear SVM for a two-feature teaching project.

The model learns from two outside groups:
  - lower_reference = fasting glucose at or below 99 mg/dL
  - higher_reference = fasting glucose at or above 126 mg/dL

The empty space between those groups becomes the SVM margin. The website uses
that margin as its "Pre-diabetic / Further Diagnostic Needed" teaching zone.
This is an educational screening demo, not a medical diagnosis.

Kid-sized picture: blue players and red players stand on graph paper. The SVM
tries to lay the widest straight road between the teams. ``weights`` control the
road's tilt, ``bias`` slides it, and scores -1/+1 mark the two road edges.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOWER_LABEL = "lower_reference"
HIGHER_LABEL = "higher_reference"
LOWER_ZONE = "Lower screening signal"
MARGIN_ZONE = "Pre-diabetic / Further Diagnostic Needed"
HIGHER_ZONE = "Higher screening signal"


@dataclass(frozen=True)
class TrainingExample:
    """One labeled row from the CSV dataset."""

    fasting_glucose_mg_dl: float
    bmi: float
    label: str

    @property
    def numeric_label(self) -> int:
        """SVMs use -1 and +1 instead of long category names."""

        return -1 if self.label == LOWER_LABEL else 1

    @property
    def features(self) -> tuple[float, float]:
        return (self.fasting_glucose_mg_dl, self.bmi)


def load_dataset(csv_path: str | Path) -> list[TrainingExample]:
    """Read and validate the beginner-friendly CSV file."""

    examples: list[TrainingExample] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
            try:
                glucose = float(row["fasting_glucose_mg_dl"])
                bmi = float(row["bmi"])
                label = row["label"].strip()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Bad dataset value on row {row_number}.") from error

            if glucose <= 0 or bmi <= 0:
                raise ValueError(f"Measurements must be positive on row {row_number}.")
            if label not in {LOWER_LABEL, HIGHER_LABEL}:
                raise ValueError(f"Unknown label '{label}' on row {row_number}.")

            examples.append(TrainingExample(glucose, bmi, label))

    if len(examples) < 4:
        raise ValueError("The dataset needs at least four examples.")
    if {example.label for example in examples} != {LOWER_LABEL, HIGHER_LABEL}:
        raise ValueError("The dataset needs examples from both outside groups.")
    return examples


class SimpleLinearSVM:
    """A soft-margin linear SVM trained with small gradient steps.

    Think of ``weights`` as instructions for tilting a straight line. During
    training, every mistake pushes the line in a better direction. The model
    stops after repeating that tiny correction many times.
    """

    def __init__(
        self,
        learning_rate: float = 0.08,
        regularization: float = 0.02,
        epochs: int = 6000,
    ) -> None:
        if learning_rate <= 0 or regularization <= 0 or epochs <= 0:
            raise ValueError("Learning settings must be greater than zero.")
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.means = [0.0, 0.0]
        self.scales = [1.0, 1.0]
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        self.training_examples: list[TrainingExample] = []

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        """Balance all numbers on one seesaw: total divided by item count."""
        return sum(values) / len(values)

    def _standardize(self, features: Sequence[float]) -> tuple[float, float]:
        """Change different rulers into comparable map steps.

        Subtracting the mean puts the group middle at zero. Dividing by the
        scale says how many usual-size steps the value sits from that middle.
        """
        return (
            (features[0] - self.means[0]) / self.scales[0],
            (features[1] - self.means[1]) / self.scales[1],
        )

    def fit(self, examples: Iterable[TrainingExample]) -> "SimpleLinearSVM":
        """Learn one straight separator from the labeled examples."""

        self.training_examples = list(examples)
        if len(self.training_examples) < 4:
            raise ValueError("Fit needs at least four examples.")

        # Turn rows [(glucose, BMI), ...] into two columns so each ruler can be
        # centered and scaled separately.
        columns = list(zip(*(example.features for example in self.training_examples)))
        # The mean is the balance point of each column's seesaw.
        self.means = [self._mean(column) for column in columns]
        self.scales = []
        for column, mean in zip(columns, self.means):
            # Variance averages the squared gaps from the middle; squaring makes
            # left and right gaps both positive and makes big gaps count more.
            variance = self._mean([(value - mean) ** 2 for value in column])
            # Square root returns to normal-sized ruler units. 1e-9 prevents a
            # zero ruler if every value in a column happens to match.
            self.scales.append(max(math.sqrt(variance), 1e-9))

        # Store each equal-ruler point beside its team sign: -1 or +1.
        standardized = [
            (self._standardize(example.features), example.numeric_label)
            for example in self.training_examples
        ]
        # Begin with a flat, unsteered score; training will nudge these numbers.
        self.weights = [0.0, 0.0]
        self.bias = 0.0
        count = len(standardized)

        # An epoch is one practice round through all dots.
        for epoch in range(self.epochs):
            # Regularization is a gentle rubber band pulling huge weights toward
            # zero so the road stays simple instead of tilting wildly.
            weight_gradient = [
                2 * self.regularization * self.weights[0],
                2 * self.regularization * self.weights[1],
            ]
            bias_gradient = 0.0

            # Let every training dot suggest a correction.
            for features, label in standardized:
                # The dot product multiplies each map step by its steering
                # weight, adds both pushes, then slides by the bias.
                score = self.weights[0] * features[0] + self.weights[1] * features[1] + self.bias
                # label * score >= 1 means the dot is safely beyond its road
                # edge. A smaller value means it is wrong or inside the road.
                if label * score < 1:
                    weight_gradient[0] -= label * features[0] / count
                    weight_gradient[1] -= label * features[1] / count
                    bias_gradient -= label / count

            # Take smaller footsteps in later rounds so the road settles rather
            # than wobbling past a good place.
            step = self.learning_rate / (1 + epoch * 0.0005)
            # Move opposite each gradient—the direction that lowers the mistake.
            self.weights[0] -= step * weight_gradient[0]
            self.weights[1] -= step * weight_gradient[1]
            self.bias -= step * bias_gradient

        return self

    def decision_score(self, fasting_glucose_mg_dl: float, bmi: float) -> float:
        """Return position relative to the separator; this is not probability."""

        if not self.training_examples:
            raise ValueError("Train the model with fit() before making a prediction.")
        if fasting_glucose_mg_dl <= 0 or bmi <= 0:
            raise ValueError("Glucose and BMI must be greater than zero.")
        # Put the new dot on the same equal-step rulers used during training.
        glucose_scaled, bmi_scaled = self._standardize((fasting_glucose_mg_dl, bmi))
        # This signed road position is not a percent chance.
        return self.weights[0] * glucose_scaled + self.weights[1] * bmi_scaled + self.bias

    @staticmethod
    def zone_from_score(score: float) -> str:
        if score <= -1:
            return LOWER_ZONE
        if score >= 1:
            return HIGHER_ZONE
        return MARGIN_ZONE

    def predict(self, fasting_glucose_mg_dl: float, bmi: float) -> dict[str, float | str]:
        score = self.decision_score(fasting_glucose_mg_dl, bmi)
        return {"zone": self.zone_from_score(score), "score": score}

    def training_accuracy(self) -> float:
        """Check whether each outside example is on the correct side."""

        correct = 0
        for example in self.training_examples:
            score = self.decision_score(*example.features)
            predicted_label = 1 if score >= 0 else -1
            correct += predicted_label == example.numeric_label
        return correct / len(self.training_examples)

    def support_indices(self, per_side: int = 3) -> set[int]:
        """Pick examples closest to their SVM margin line for the graph."""

        selected: set[int] = set()
        for numeric_label in (-1, 1):
            candidates = [
                (
                    abs(self.decision_score(*example.features) - numeric_label),
                    index,
                )
                for index, example in enumerate(self.training_examples)
                if example.numeric_label == numeric_label
            ]
            selected.update(index for _, index in sorted(candidates)[:per_side])
        return selected

    def export_json(self, output_path: str | Path) -> None:
        """Save the learned line and graph points for JavaScript."""

        support_indices = self.support_indices()
        payload = {
            "model_type": "linear_svm",
            "feature_names": ["Fasting Blood Glucose (mg/dL)", "Body Mass Index (BMI)"],
            "means": self.means,
            "scales": self.scales,
            "weights": self.weights,
            "bias": self.bias,
            "margin": {"lower": -1, "center": 0, "upper": 1},
            "zones": {
                "lower": LOWER_ZONE,
                "margin": MARGIN_ZONE,
                "higher": HIGHER_ZONE,
            },
            "clinical_glucose_reference": {
                "normal_max": 99,
                "prediabetes_min": 100,
                "prediabetes_max": 125,
                "diabetes_min": 126,
            },
            "training_examples": [
                {
                    "fasting_glucose_mg_dl": example.fasting_glucose_mg_dl,
                    "bmi": example.bmi,
                    "label": example.label,
                    "score": self.decision_score(*example.features),
                    "is_support_example": index in support_indices,
                }
                for index, example in enumerate(self.training_examples)
            ],
        }
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
