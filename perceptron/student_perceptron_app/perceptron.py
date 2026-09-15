"""Core implementation of Hagan's single-neuron perceptron learning rule.

The web application intentionally keeps the learning algorithm in this small,
framework-independent module.  It can therefore be imported from Flask, a
notebook, a test suite, or a command-line program without starting a server.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Perceptron:
    """A binary hard-limit perceptron trained one sample at a time.

    Parameters
    ----------
    input_dim:
        Number of input features.  The student demo uses two: study hours and
        normalized attendance.
    learning_rate:
        Multiplier applied to every correction.  Hagan's unified rule is
        recovered exactly when this value is ``1.0``.

    Notes
    -----
    ``weights`` is stored as a one-dimensional NumPy array with shape
    ``(input_dim,)``.  For one sample ``p`` with the same shape, ``weights @ p``
    is the scalar dot product :math:`w^T p`.  Zero initialization makes every
    animation deterministic and keeps the first model state easy to explain.
    """

    def __init__(self, input_dim: int = 2, learning_rate: float = 1.0) -> None:
        """Create a zero-initialized perceptron with validated hyperparameters."""

        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer.")
        if not np.isfinite(learning_rate) or float(learning_rate) <= 0:
            raise ValueError("learning_rate must be a finite positive number.")

        self.input_dim = input_dim
        self.learning_rate = float(learning_rate)

        # W has one coefficient for each input feature.  Zeros provide a
        # reproducible starting boundary for a teaching visualization.
        self.weights: NDArray[np.float64] = np.zeros(input_dim, dtype=np.float64)

        # The scalar bias shifts the decision boundary away from the origin.
        self.bias = 0.0

    @staticmethod
    def hardlim(net: ArrayLike) -> int | NDArray[np.int64]:
        """Apply Hagan's hard-limit transfer function element by element.

        Values on the boundary (``n == 0``) belong to Class 1, exactly matching
        ``hardlim(n) = 1`` for ``n >= 0`` and ``0`` otherwise.
        """

        net_array = np.asarray(net, dtype=np.float64)
        activations = (net_array >= 0.0).astype(np.int64)
        return int(activations) if activations.ndim == 0 else activations

    def _coerce_inputs(self, p: ArrayLike) -> tuple[NDArray[np.float64], bool]:
        """Validate input data and report whether it represents one sample."""

        inputs = np.asarray(p, dtype=np.float64)
        is_single_sample = inputs.ndim == 1

        if is_single_sample:
            if inputs.shape != (self.input_dim,):
                raise ValueError(
                    f"A sample must have shape ({self.input_dim},), got {inputs.shape}."
                )
        elif inputs.ndim == 2:
            if inputs.shape[1] != self.input_dim:
                raise ValueError(
                    f"Each sample must contain {self.input_dim} features; "
                    f"received shape {inputs.shape}."
                )
        else:
            raise ValueError("Inputs must be one sample or a two-dimensional sample matrix.")

        if not np.all(np.isfinite(inputs)):
            raise ValueError("Inputs must contain only finite numbers.")

        return inputs, is_single_sample

    def _coerce_dataset(
        self, P: ArrayLike, T: Sequence[int] | NDArray[np.integer[Any]]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """Convert a complete training set to aligned NumPy arrays."""

        samples, is_single_sample = self._coerce_inputs(P)
        if is_single_sample:
            samples = samples.reshape(1, self.input_dim)

        targets_raw = np.asarray(T)
        if targets_raw.ndim != 1 or len(targets_raw) != len(samples):
            raise ValueError("T must be a one-dimensional target vector aligned with P.")
        if len(samples) == 0:
            raise ValueError("The training set must contain at least one sample.")
        if not np.all(np.isin(targets_raw, (0, 1))):
            raise ValueError("Every target must be either 0 or 1.")

        return samples, targets_raw.astype(np.int64)

    def net_input(self, p: ArrayLike) -> float | NDArray[np.float64]:
        """Compute :math:`n = Wp + b` for one sample or a batch.

        For one sample, NumPy evaluates ``inputs @ weights`` as a scalar dot
        product.  For a matrix ``P`` with shape ``(sample_count, input_dim)``,
        the same expression performs all row-by-row dot products at once and
        returns one net input per sample.  Adding the scalar bias relies on
        NumPy broadcasting, so the same ``b`` is added to every sample.
        """

        inputs, is_single_sample = self._coerce_inputs(p)
        nets = inputs @ self.weights + self.bias
        return float(nets) if is_single_sample else np.asarray(nets, dtype=np.float64)

    def predict(self, p: ArrayLike) -> int | NDArray[np.int64]:
        """Return hard-limit class predictions for one sample or a batch."""

        # The matrix operation occurs in net_input; hardlim then thresholds
        # each scalar net value without an explicit Python loop.
        return self.hardlim(self.net_input(p))

    def train_step(self, p: ArrayLike, t: int) -> dict[str, Any]:
        """Process one example and apply the unified perceptron update.

        The calculation follows Chapter 4 directly:

        ``n = w^T p + b``
        ``a = hardlim(n)``
        ``e = t - a``
        ``w_new = w_old + learning_rate * e * p``
        ``b_new = b_old + learning_rate * e``

        A serializable diagnostics dictionary describes both the state used for
        the prediction and the state after the possible correction.
        """

        sample, is_single_sample = self._coerce_inputs(p)
        if not is_single_sample:
            raise ValueError("train_step accepts exactly one input sample.")
        if isinstance(t, bool) or t not in (0, 1):
            raise ValueError("The target t must be the integer 0 or 1.")

        # Save the old parameters so the UI can explain the exact transition.
        weights_before = self.weights.copy()
        bias_before = self.bias

        # One dot product produces the neuron's scalar net input.
        net = float(self.weights @ sample + self.bias)
        activation = int(self.hardlim(net))
        error = int(t - activation)

        # Multiplying the vector p by scalar e gives a vector correction with
        # one element per weight.  If e == 0, both corrections are exactly zero.
        if error != 0:
            self.weights = self.weights + self.learning_rate * error * sample
            self.bias = self.bias + self.learning_rate * error

        return {
            "p": sample.tolist(),
            "t": int(t),
            "n": net,
            "a": activation,
            "e": error,
            "W_before": weights_before.tolist(),
            "b_before": float(bias_before),
            "W": self.weights.tolist(),
            "b": float(self.bias),
            "updated": error != 0,
        }

    def train_epoch(self, P: ArrayLike, T: Sequence[int]) -> list[dict[str, Any]]:
        """Visit every training pair once, in the supplied sequence."""

        samples, targets = self._coerce_dataset(P, T)
        history: list[dict[str, Any]] = []

        # Online perceptron learning updates the parameters immediately, so the
        # next sample sees the weights produced by the previous sample.
        for sample_index, (sample, target) in enumerate(zip(samples, targets)):
            diagnostics = self.train_step(sample, int(target))
            diagnostics["sample_index"] = sample_index
            history.append(diagnostics)

        return history

    def _accuracy(self, samples: NDArray[np.float64], targets: NDArray[np.int64]) -> float:
        """Return the fraction of training samples currently classified correctly."""

        predictions = np.asarray(self.predict(samples), dtype=np.int64)
        return float(np.mean(predictions == targets))

    def fit(
        self,
        P: ArrayLike,
        T: Sequence[int],
        max_epochs: int = 100,
    ) -> dict[str, Any]:
        """Train until an entire data set is correct or the epoch limit is met.

        Every sample visit is recorded for boundary animation.  A second,
        compact epoch history records the number of corrections and the final
        accuracy after each full pass through the data.

        Returns
        -------
        dict
            JSON-friendly training history, convergence metadata, and final
            model parameters.  The perceptron's own ``weights`` and ``bias``
            also remain at that final state.
        """

        samples, targets = self._coerce_dataset(P, T)
        if isinstance(max_epochs, bool) or not isinstance(max_epochs, int) or max_epochs < 1:
            raise ValueError("max_epochs must be a positive integer.")

        initial_accuracy = self._accuracy(samples, targets)
        initial_state = {
            "W": self.weights.tolist(),
            "b": float(self.bias),
            "accuracy": initial_accuracy,
            "epoch": 0,
            "global_step": 0,
        }

        step_history: list[dict[str, Any]] = []
        epoch_history: list[dict[str, Any]] = []
        converged = initial_accuracy == 1.0
        epochs_completed = 0
        global_step = 0

        for epoch_number in range(1, max_epochs + 1):
            if converged:
                break

            updates_this_epoch = 0
            for sample_index, (sample, target) in enumerate(zip(samples, targets)):
                global_step += 1
                diagnostics = self.train_step(sample, int(target))
                updates_this_epoch += int(diagnostics["updated"])

                # Re-evaluating all rows after the correction produces the live
                # accuracy that accompanies this exact boundary snapshot.
                diagnostics.update(
                    {
                        "epoch": epoch_number,
                        "sample_index": sample_index,
                        "global_step": global_step,
                        "accuracy": self._accuracy(samples, targets),
                    }
                )
                step_history.append(diagnostics)

            epochs_completed = epoch_number
            epoch_accuracy = self._accuracy(samples, targets)
            converged = epoch_accuracy == 1.0
            epoch_history.append(
                {
                    "epoch": epoch_number,
                    "updates": updates_this_epoch,
                    "accuracy": epoch_accuracy,
                    "W": self.weights.tolist(),
                    "b": float(self.bias),
                    "converged": converged,
                }
            )

        final_accuracy = self._accuracy(samples, targets)
        return {
            "initial_state": initial_state,
            "history": step_history,
            "epoch_history": epoch_history,
            "converged": converged,
            "total_epochs": epochs_completed,
            "total_steps": global_step,
            "total_updates": sum(int(step["updated"]) for step in step_history),
            "final": {
                "W": self.weights.tolist(),
                "b": float(self.bias),
                "accuracy": final_accuracy,
            },
        }
