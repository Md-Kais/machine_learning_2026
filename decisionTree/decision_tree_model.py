"""A tiny Decision Tree written from scratch for programming education.

The dataset and labels are synthetic. The names "Low Risk / Discharge" and
"High Risk / Admit to ICU" are toy target labels requested for the lesson.
Never use this program for a real patient or a real medical decision.

Imagine sorting pretend cards with yes/no gates. At each gate, a number on or
below the fence goes left and a larger number goes right. Training tries every
sensible fence and keeps the one that separates the two card colors most cleanly.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


LOW_LABEL = "low_risk_discharge"
HIGH_LABEL = "high_risk_icu"

FEATURES = ("systolic_bp_mm_hg", "max_heart_rate_bpm")
FEATURE_TITLES = {
    "systolic_bp_mm_hg": "Systolic blood pressure",
    "max_heart_rate_bpm": "Maximum heart rate",
}
FEATURE_UNITS = {
    "systolic_bp_mm_hg": "mmHg",
    "max_heart_rate_bpm": "bpm",
}
LABEL_TITLES = {
    LOW_LABEL: "Low Risk / Discharge",
    HIGH_LABEL: "High Risk / Admit to ICU",
}


@dataclass(frozen=True)
class TrainingExample:
    """One row from the CSV table."""

    systolic_bp_mm_hg: float
    max_heart_rate_bpm: float
    label: str

    def value(self, feature: str) -> float:
        return float(getattr(self, feature))


@dataclass
class TreeNode:
    """One question box or one final answer in the tree."""

    node_id: str
    depth: int
    sample_count: int
    class_counts: dict[str, int]
    prediction: str
    impurity: float
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> dict:
        result = {
            "id": self.node_id,
            "depth": self.depth,
            "samples": self.sample_count,
            "class_counts": self.class_counts,
            "prediction": self.prediction,
            "impurity": self.impurity,
            "is_leaf": self.is_leaf,
        }
        if not self.is_leaf:
            result.update(
                {
                    "feature": self.feature,
                    "feature_title": FEATURE_TITLES[self.feature],
                    "unit": FEATURE_UNITS[self.feature],
                    "threshold": self.threshold,
                    "left": self.left.to_dict(),
                    "right": self.right.to_dict(),
                }
            )
        return result


def load_dataset(csv_path: str | Path) -> list[TrainingExample]:
    """Read the CSV and reject confusing or broken rows."""

    examples: list[TrainingExample] = []
    with Path(csv_path).open(newline="", encoding="utf-8") as csv_file:
        for row_number, row in enumerate(csv.DictReader(csv_file), start=2):
            try:
                systolic_bp = float(row["systolic_bp_mm_hg"])
                max_heart_rate = float(row["max_heart_rate_bpm"])
                label = row["label"].strip()
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Bad dataset value on row {row_number}.") from error

            if systolic_bp <= 0 or max_heart_rate <= 0:
                raise ValueError(f"Measurements must be positive on row {row_number}.")
            if label not in {LOW_LABEL, HIGH_LABEL}:
                raise ValueError(f"Unknown label '{label}' on row {row_number}.")
            examples.append(TrainingExample(systolic_bp, max_heart_rate, label))

    if len(examples) < 4:
        raise ValueError("The dataset needs at least four examples.")
    if {example.label for example in examples} != {LOW_LABEL, HIGH_LABEL}:
        raise ValueError("The dataset needs both target labels.")
    return examples


def gini_impurity(examples: Sequence[TrainingExample]) -> float:
    """Return how mixed the label colors are: 0 means one perfectly sorted color.

    If a bag is 3/4 blue and 1/4 red, Gini is
    ``1 - (3/4)² - (1/4)² = 0.375``. Squaring rewards a bag dominated by one
    color, like noticing that grabbing two blue marbles in a row is quite likely.
    """

    if not examples:
        return 0.0
    # Count how many cards wear each of the two label colors.
    counts = Counter(example.label for example in examples)
    # total is the number of cards in this bag.
    total = len(examples)
    # Turn each count into a fraction, square it, add the squares, then subtract
    # from 1. A pure bag gives 1 - 1² = 0 messiness.
    return 1.0 - sum((count / total) ** 2 for count in counts.values())


def majority_label(examples: Sequence[TrainingExample]) -> str:
    """Choose the most common answer, preferring the low toy label on a tie."""

    counts = Counter(example.label for example in examples)
    return max((LOW_LABEL, HIGH_LABEL), key=lambda label: (counts[label], label == LOW_LABEL))


class SimpleDecisionTree:
    """A small CART-style classifier using only yes/no number questions."""

    def __init__(self, max_depth: int = 2, min_samples_split: int = 2) -> None:
        if max_depth < 1 or min_samples_split < 2:
            raise ValueError("max_depth must be at least 1 and min_samples_split at least 2.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root: TreeNode | None = None
        self.training_examples: list[TrainingExample] = []
        self._next_node_number = 1

    def _new_node_id(self) -> str:
        node_id = f"node-{self._next_node_number}"
        self._next_node_number += 1
        return node_id

    def _best_split(
        self, examples: Sequence[TrainingExample]
    ) -> tuple[str, float, list[TrainingExample], list[TrainingExample]] | None:
        """Try every halfway cut and keep the one making the purest groups."""

        # Measure the messiness before placing any new fence.
        parent_impurity = gini_impurity(examples)
        best_gain = 0.0
        best_result = None

        # Try both number columns, one at a time.
        for feature in FEATURES:
            # Repeated values need only one position on our number line.
            values = sorted({example.value(feature) for example in examples})
            # A safe fence sits halfway between neighboring values, never on a card.
            thresholds = [
                (left_value + right_value) / 2
                for left_value, right_value in zip(values, values[1:])
            ]

            # Test every possible fence for this feature.
            for threshold in thresholds:
                # Cards on/below the fence enter the left basket; larger cards
                # enter the right basket.
                left = [example for example in examples if example.value(feature) <= threshold]
                right = [example for example in examples if example.value(feature) > threshold]
                if not left or not right:
                    continue

                # A large basket must count more than a tiny basket, so multiply
                # each messiness score by that basket's share of all cards.
                weighted_impurity = (
                    len(left) / len(examples) * gini_impurity(left)
                    + len(right) / len(examples) * gini_impurity(right)
                )
                # Gain is how much mess the fence removed; larger is better.
                gain = parent_impurity - weighted_impurity

                # Strictly greater means an exact tie keeps the first feature.
                # That makes the tiny example deterministic and easy to follow.
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_result = (feature, threshold, left, right)

        return best_result

    def _build(self, examples: Sequence[TrainingExample], depth: int) -> TreeNode:
        # Count label colors and create the question/answer box for this bag.
        counts = Counter(example.label for example in examples)
        node = TreeNode(
            node_id=self._new_node_id(),
            depth=depth,
            sample_count=len(examples),
            class_counts={label: counts[label] for label in (LOW_LABEL, HIGH_LABEL)},
            prediction=majority_label(examples),
            impurity=gini_impurity(examples),
        )

        # Stop when the tree is tall enough, the bag is too small, or every card
        # already has the same color (impurity 0).
        should_stop = (
            depth >= self.max_depth
            or len(examples) < self.min_samples_split
            or node.impurity == 0
        )
        if should_stop:
            return node

        # Ask the fence contest for the best available question.
        split = self._best_split(examples)
        if split is None:
            return node

        # Unpack the winning question and its two new card baskets.
        feature, threshold, left_examples, right_examples = split
        node.feature = feature
        node.threshold = threshold
        # Grow the left and right child boxes by repeating the same recipe.
        node.left = self._build(left_examples, depth + 1)
        node.right = self._build(right_examples, depth + 1)
        return node

    def fit(self, examples: Iterable[TrainingExample]) -> "SimpleDecisionTree":
        """Grow the yes/no tree from the training examples."""

        self.training_examples = list(examples)
        if len(self.training_examples) < 4:
            raise ValueError("Fit needs at least four examples.")
        self._next_node_number = 1
        self.root = self._build(self.training_examples, depth=0)
        return self

    def predict(self, systolic_bp_mm_hg: float, max_heart_rate_bpm: float) -> dict:
        """Follow yes/no branches and return the toy label plus the full path."""

        if self.root is None:
            raise ValueError("Train the tree with fit() before predicting.")
        if systolic_bp_mm_hg <= 0 or max_heart_rate_bpm <= 0:
            raise ValueError("Blood pressure and heart rate must be greater than zero.")

        values = {
            "systolic_bp_mm_hg": float(systolic_bp_mm_hg),
            "max_heart_rate_bpm": float(max_heart_rate_bpm),
        }
        path = []
        node = self.root

        while not node.is_leaf:
            # Read the input number named by this particular question box.
            value = values[node.feature]
            # On or below the fence is Yes/left; above is No/right.
            go_left = value <= node.threshold
            path.append(
                {
                    "node_id": node.node_id,
                    "feature": node.feature,
                    "feature_title": FEATURE_TITLES[node.feature],
                    "unit": FEATURE_UNITS[node.feature],
                    "value": value,
                    "threshold": node.threshold,
                    "operator": "<=",
                    "answer": "Yes" if go_left else "No",
                    "branch": "left" if go_left else "right",
                }
            )
            # Walk to the chosen child and ask again until a leaf is reached.
            node = node.left if go_left else node.right

        return {"label": node.prediction, "leaf_id": node.node_id, "path": path}

    def training_accuracy(self) -> float:
        correct = sum(
            self.predict(example.systolic_bp_mm_hg, example.max_heart_rate_bpm)["label"]
            == example.label
            for example in self.training_examples
        )
        return correct / len(self.training_examples)

    def rules_as_text(self) -> str:
        """Make an indented, human-readable version of the learned tree."""

        if self.root is None:
            raise ValueError("Train the tree before printing rules.")
        lines: list[str] = []

        def visit(node: TreeNode, indent: str) -> None:
            if node.is_leaf:
                lines.append(f"{indent}RETURN {LABEL_TITLES[node.prediction]}")
                return
            title = FEATURE_TITLES[node.feature]
            unit = FEATURE_UNITS[node.feature]
            lines.append(f"{indent}IF {title} <= {node.threshold:g} {unit}:")
            visit(node.left, indent + "  ")
            lines.append(f"{indent}ELSE:")
            visit(node.right, indent + "  ")

        visit(self.root, "")
        return "\n".join(lines)

    def export_json(self, output_path: str | Path) -> None:
        """Save the learned questions and graph points for the website."""

        if self.root is None:
            raise ValueError("Train the tree before exporting it.")
        payload = {
            "model_type": "decision_tree_classifier",
            "algorithm": "CART-style binary tree using Gini impurity",
            "max_depth": self.max_depth,
            "features": {
                feature: {"title": FEATURE_TITLES[feature], "unit": FEATURE_UNITS[feature]}
                for feature in FEATURES
            },
            "labels": {
                LOW_LABEL: LABEL_TITLES[LOW_LABEL],
                HIGH_LABEL: LABEL_TITLES[HIGH_LABEL],
            },
            "tree": self.root.to_dict(),
            "training_examples": [
                {
                    "systolic_bp_mm_hg": example.systolic_bp_mm_hg,
                    "max_heart_rate_bpm": example.max_heart_rate_bpm,
                    "label": example.label,
                }
                for example in self.training_examples
            ],
            "warning": (
                "Synthetic programming lesson only. Never use two vital signs "
                "to discharge, admit, diagnose, or treat a real patient."
            ),
        }
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
