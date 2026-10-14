"""Categorical ID3: entropy, information gain, and one branch per value.

Reference: 15.+Decision+tree+_+ID3.pdf, Tennis Playing example.
Training and prediction live here; train_model.py and the UI are separate.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

FEATURES = ("Outlook", "Temp", "Humidity", "Wind")
CATEGORIES = {
    "Outlook": ("Sunny", "Overcast", "Rain"),
    "Temp": ("Hot", "Mild", "Cool"),
    "Humidity": ("High", "Normal"),
    "Wind": ("Weak", "Strong"),
}


def load_dataset(csv_path):
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not set((*FEATURES, "Play")).issubset(reader.fieldnames or []):
            raise ValueError("CSV needs Outlook, Temp, Humidity, Wind, Play columns.")
        rows = []
        for number, row in enumerate(reader, 2):
            clean = {key: (row.get(key) or "").strip() for key in (*FEATURES, "Play")}
            for key in FEATURES:
                if clean[key] not in CATEGORIES[key]:
                    raise ValueError(f"Unknown {key} on CSV row {number}: {clean[key]!r}")
            if clean["Play"] not in ("Yes", "No"):
                raise ValueError(f"Play must be Yes or No on CSV row {number}.")
            rows.append(clean)
    if not rows:
        raise ValueError("Dataset cannot be empty.")
    return rows


def entropy(labels):
    """H(S) = -sum(p * log2(p)); a pure or empty set has entropy zero."""
    counts = Counter(labels)
    total = sum(counts.values())
    return max(0.0, -sum((n / total) * math.log2(n / total) for n in counts.values())) if total else 0.0


def information_gain(rows, feature):
    """Parent entropy minus the weighted entropy of categorical partitions."""
    if not rows:
        return 0.0
    groups = {}
    for row in rows:
        groups.setdefault(row[feature], []).append(row["Play"])
    return entropy(row["Play"] for row in rows) - sum(
        len(labels) / len(rows) * entropy(labels) for labels in groups.values()
    )


class ID3DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_gain=0.0):
        if max_depth is not None and (isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a nonnegative integer or None.")
        if isinstance(min_samples_split, bool) or not isinstance(min_samples_split, int) or min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer at least 2.")
        if not math.isfinite(min_gain) or min_gain < 0:
            raise ValueError("min_gain must be finite and nonnegative.")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_gain = min_gain
        self.tree = None
        self.examples = []

    def fit(self, rows):
        self.examples = [dict(row) for row in rows]
        if not self.examples:
            raise ValueError("Fit needs at least one example.")
        for row in self.examples:
            if any(row.get(key) not in CATEGORIES[key] for key in FEATURES) or row.get("Play") not in ("Yes", "No"):
                raise ValueError("Training rows must use the Play Tennis categories.")
        self._next_id = 0
        self.tree = self._build(self.examples, list(FEATURES), 0)
        return self

    def _build(self, rows, features, depth):
        counts = Counter(row["Play"] for row in rows)
        # A tied majority uses No; tied gains use the original feature order.
        prediction = max(("No", "Yes"), key=lambda label: counts[label])
        node = {"id": f"n{self._next_id}", "depth": depth, "sample_count": len(rows),
                "class_counts": {label: counts[label] for label in ("Yes", "No")},
                "prediction": prediction, "entropy": entropy(row["Play"] for row in rows),
                "is_leaf": True, "gains": {}, "children": {}}
        self._next_id += 1
        if len(counts) == 1 or not features or len(rows) < self.min_samples_split or (
            self.max_depth is not None and depth >= self.max_depth
        ):
            return node
        node["gains"] = {feature: information_gain(rows, feature) for feature in features}
        best = max(features, key=lambda feature: node["gains"][feature])
        if node["gains"][best] < self.min_gain:
            return node
        # Zero-gain splits are allowed: later attributes may explain XOR-like data.
        if len({row[best] for row in rows}) < 2:
            remaining = [feature for feature in features if feature != best]
            if not remaining:
                return node
            # Skip a constant attribute without creating a meaningless single branch.
            replacement = self._build(rows, remaining, depth)
            return replacement
        node.update(is_leaf=False, feature=best, gain=node["gains"][best])
        remaining = [feature for feature in features if feature != best]
        for value in CATEGORIES[best]:
            subset = [row for row in rows if row[best] == value]
            if subset:
                node["children"][value] = self._build(subset, remaining, depth + 1)
        return node

    def predict(self, values):
        if self.tree is None:
            raise ValueError("Train with fit() before predicting.")
        for key in FEATURES:
            if values.get(key) not in CATEGORIES[key]:
                raise ValueError(f"Choose a valid {key}: {', '.join(CATEGORIES[key])}.")
        node, steps = self.tree, []
        while not node["is_leaf"]:
            feature, value = node["feature"], values[node["feature"]]
            child = node["children"].get(value)
            steps.append({"node_id": node["id"], "feature": feature, "value": value,
                          "gain": node["gain"], "fallback": child is None})
            if child is None:
                break  # Valid but unseen category: use this node's training majority.
            node = child
        return {"label": node["prediction"], "leaf_id": node["id"], "path": steps,
                "fallback": bool(steps and steps[-1]["fallback"])}

    def training_accuracy(self):
        if not self.examples:
            raise ValueError("Train with fit() first.")
        return sum(self.predict(row)["label"] == row["Play"] for row in self.examples) / len(self.examples)

    def export_json(self, output_path):
        if self.tree is None:
            raise ValueError("Train with fit() before exporting.")
        payload = {"model_type": "id3", "criterion": "entropy_information_gain",
                   "features": list(FEATURES), "categories": CATEGORIES,
                   "parameters": {"max_depth": self.max_depth, "min_samples_split": self.min_samples_split,
                                  "min_gain": self.min_gain}, "tree": self.tree,
                   "training_examples": self.examples, "training_accuracy": self.training_accuracy(),
                   "reference": "15.+Decision+tree+_+ID3.pdf (pages 2-3, 19-26)"}
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
