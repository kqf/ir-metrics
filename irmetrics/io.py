import numpy as np
from functools import wraps


def to_scalar(x):
    if not x.shape:
        return x.item()
    return x


def ensure_inputs(y_true, y_pred, k=20):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)

    # Take at most k labels
    y_true = y_true.T[:, :k]
    y_pred = y_pred[:, :k]
    return y_true, y_pred


def _ensure_io(f):
    @wraps(f)
    def wrapper(y_true, y_pred, k=20):
        # Ensure (n_samples, n_labels) shapes for the inputs
        y_true_, y_pred_ = ensure_inputs(y_true, y_pred, k)

        # Calculate the measure
        raw_outputs = f(y_true_, y_pred_, k)

        # Remove unwanted dimensions if any
        return to_scalar(np.squeeze(raw_outputs))

    return wrapper
