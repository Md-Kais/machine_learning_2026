"""Binary linear soft-margin SVM, solved with sequential minimal optimization.

Based on svm15.pdf: score = w dot z + b; margins are score = +/-1.
The PDF starts with hard margins and later introduces slack. A finite C here
penalizes slack. No nonlinear kernel and no third class are used.
"""
from __future__ import annotations
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

LOWER_LABEL = "lower_reference"
HIGHER_LABEL = "higher_reference"
LOWER_ZONE = "Lower reference (-1)"
HIGHER_ZONE = "Higher reference (+1)"


@dataclass(frozen=True)
class TrainingExample:
    fasting_glucose_mg_dl: float
    bmi: float
    label: str

    @property
    def features(self):
        return (self.fasting_glucose_mg_dl, self.bmi)

    @property
    def numeric_label(self):
        if self.label not in (LOWER_LABEL, HIGHER_LABEL):
            raise ValueError("Unknown training label.")
        return -1 if self.label == LOWER_LABEL else 1


def load_dataset(csv_path):
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as source:
        rows = []
        for number, row in enumerate(csv.DictReader(source), 2):
            try:
                example = TrainingExample(float(row["fasting_glucose_mg_dl"]), float(row["bmi"]), row["label"].strip())
                validate_features(*example.features)
                example.numeric_label
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Bad dataset value on row {number}: {error}") from error
            rows.append(example)
    validate_examples(rows)
    return rows


def validate_features(glucose, bmi):
    if not all(math.isfinite(value) and value > 0 for value in (glucose, bmi)):
        raise ValueError("Glucose and BMI must be finite numbers greater than zero.")


def validate_examples(examples):
    if len(examples) < 4:
        raise ValueError("Fit needs at least four examples.")
    for example in examples:
        validate_features(*example.features)
        example.numeric_label
    if {example.label for example in examples} != {LOWER_LABEL, HIGHER_LABEL}:
        raise ValueError("Dataset needs both reference classes.")


class SimpleLinearSVM:
    def __init__(self, C=10.0, tolerance=0.0001, max_passes=20, max_iterations=10000, random_seed=42):
        if not math.isfinite(C) or C <= 0 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("C and tolerance must be finite and positive.")
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (max_passes, max_iterations)):
            raise ValueError("max_passes and max_iterations must be positive integers.")
        self.C, self.tolerance = C, tolerance
        self.max_passes, self.max_iterations, self.random_seed = max_passes, max_iterations, random_seed
        self.weights, self.bias = [0.0, 0.0], 0.0
        self.means, self.scales = [0.0, 0.0], [1.0, 1.0]
        self.training_examples = []
        self.alphas = []

    def _standardize(self, values):
        return tuple((value - mean) / scale for value, mean, scale in zip(values, self.means, self.scales))

    def fit(self, examples):
        self.training_examples = list(examples)
        validate_examples(self.training_examples)
        columns = list(zip(*(row.features for row in self.training_examples)))
        self.means = [sum(col) / len(col) for col in columns]
        self.scales = [math.sqrt(sum((v - mean)**2 for v in col) / len(col)) or 1.0
                       for col, mean in zip(columns, self.means)]
        X = [self._standardize(row.features) for row in self.training_examples]
        y = [row.numeric_label for row in self.training_examples]
        n = len(X)
        K = [[sum(a*b for a,b in zip(xi,xj)) for xj in X] for xi in X]
        self.alphas, self.bias = [0.0]*n, 0.0
        rng = random.Random(self.random_seed)
        passes = iterations = 0
        eps = 1e-8

        def score(i):
            return sum(a*yi*k for a,yi,k in zip(self.alphas,y,K[i])) + self.bias

        # Dual: maximize sum(alpha) - 0.5*sum(ai*aj*yi*yj*xi dot xj),
        # subject to 0 <= alpha <= C and sum(alpha*y) = 0.
        # Update two multipliers together to preserve the equality constraint.
        while passes < self.max_passes and iterations < self.max_iterations:
            changed = 0
            for i in range(n):
                Ei = score(i) - y[i]
                ai = self.alphas[i]
                if not ((y[i]*Ei < -self.tolerance and ai < self.C-eps) or
                        (y[i]*Ei > self.tolerance and ai > eps)):
                    continue
                candidates = [j for j in range(n) if j != i]
                rng.shuffle(candidates)
                candidates.sort(key=lambda j: abs(Ei - (score(j)-y[j])), reverse=True)
                for j in candidates:
                    Ej = score(j) - y[j]
                    aj = self.alphas[j]
                    if y[i] != y[j]:
                        L, H = max(0.0, aj-ai), min(self.C, self.C+aj-ai)
                    else:
                        L, H = max(0.0, ai+aj-self.C), min(self.C, ai+aj)
                    eta = K[i][i] + K[j][j] - 2*K[i][j]
                    if H-L < eps:
                        continue
                    if eta > eps:
                        new_j = min(H, max(L, aj + y[j]*(Ei-Ej)/eta))
                    else:
                        # Duplicate points: compare the dual at both endpoints.
                        def objective(candidate):
                            trial = self.alphas.copy()
                            trial[j] = candidate
                            trial[i] = ai + y[i]*y[j]*(aj-candidate)
                            return sum(trial) - 0.5*sum(trial[u]*trial[v]*y[u]*y[v]*K[u][v]
                                                      for u in range(n) for v in range(n))
                        new_j = max((L,H), key=objective)
                    if abs(new_j-aj) < eps:
                        continue
                    new_i = ai + y[i]*y[j]*(aj-new_j)
                    b1 = self.bias-Ei-y[i]*(new_i-ai)*K[i][i]-y[j]*(new_j-aj)*K[i][j]
                    b2 = self.bias-Ej-y[i]*(new_i-ai)*K[i][j]-y[j]*(new_j-aj)*K[j][j]
                    self.bias = b1 if eps < new_i < self.C-eps else b2 if eps < new_j < self.C-eps else (b1+b2)/2
                    self.alphas[i], self.alphas[j] = new_i, new_j
                    changed += 1
                    break
            iterations += 1
            passes = passes + 1 if changed == 0 else 0
        self.iterations = iterations
        self.weights = [sum(a*yi*x[k] for a,yi,x in zip(self.alphas,y,X)) for k in (0,1)]
        residuals = []
        for a,yi,x in zip(self.alphas,y,X):
            margin = yi*(sum(w*v for w,v in zip(self.weights,x))+self.bias)
            residuals.append(max(0.0,1-margin) if a <= eps else max(0.0,margin-1) if a >= self.C-eps else abs(margin-1))
        self.kkt_violation = max(residuals)
        self.converged = self.kkt_violation <= self.tolerance*2
        return self

    def decision_score(self, fasting_glucose_mg_dl, bmi):
        if not self.training_examples:
            raise ValueError("Train with fit() before predicting.")
        validate_features(fasting_glucose_mg_dl, bmi)
        z = self._standardize((fasting_glucose_mg_dl,bmi))
        return sum(w*v for w,v in zip(self.weights,z)) + self.bias

    @staticmethod
    def zone_from_score(score):
        if not math.isfinite(score):
            raise ValueError("Score must be finite.")
        return HIGHER_ZONE if score >= 0 else LOWER_ZONE

    def predict(self, fasting_glucose_mg_dl, bmi):
        score = self.decision_score(fasting_glucose_mg_dl,bmi)
        norm = math.hypot(*self.weights)
        return {"zone": self.zone_from_score(score), "label": HIGHER_LABEL if score >= 0 else LOWER_LABEL,
                "numeric_label": 1 if score >= 0 else -1, "score": score,
                "inside_margin": abs(score) < 1, "signed_distance": score/norm if norm else None}

    def training_accuracy(self):
        if not self.training_examples:
            raise ValueError("Train with fit() first.")
        return sum(self.predict(*row.features)["numeric_label"] == row.numeric_label
                   for row in self.training_examples) / len(self.training_examples)

    def support_indices(self):
        """Actual dual support vectors: alpha > 1e-8, including slack vectors."""
        return {i for i,a in enumerate(self.alphas) if a > 1e-8}

    def export_json(self, output_path):
        if not self.training_examples:
            raise ValueError("Train with fit() before exporting.")
        support = self.support_indices()
        norm = math.hypot(*self.weights)
        payload = {"model_type": "linear_svm", "kernel": "linear", "solver": "SMO",
                   "means": self.means, "scales": self.scales, "weights": self.weights, "bias": self.bias,
                   "margin": {"lower": -1, "center": 0, "upper": 1}, "margin_width": 2/norm if norm else None,
                   "zones": {"lower": LOWER_ZONE, "higher": HIGHER_ZONE},
                   "parameters": {"C": self.C, "tolerance": self.tolerance, "max_passes": self.max_passes,
                                  "max_iterations": self.max_iterations, "random_seed": self.random_seed},
                   "training_accuracy": self.training_accuracy(), "converged": self.converged,
                   "iterations": self.iterations, "kkt_violation": self.kkt_violation,
                   "reference": "svm15.pdf, Andrew W. Moore: linear classifiers, margins, dual QP and slack",
                   "training_examples": [{"fasting_glucose_mg_dl": row.fasting_glucose_mg_dl,
                                          "bmi": row.bmi, "label": row.label, "numeric_label": row.numeric_label,
                                          "z": self._standardize(row.features), "alpha": self.alphas[i],
                                          "score": self.decision_score(*row.features), "is_support_example": i in support}
                                         for i,row in enumerate(self.training_examples)]}
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
