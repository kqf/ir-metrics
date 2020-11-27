import numpy as np
from functools import wraps


def _ensure_io(f):
    @wraps(f)
    def wrapper(y_true, y_pred, k=20):
        # Ensure (n_samples, n_labels) shapes for the inputs
        y_true, y_pred = np.atleast_2d(y_true, y_pred)

        # Take at most k labels
        y_true = y_true.T[:, :k]
        y_pred = y_pred[:, :k]

        # Calculate the measure
        raw_outputs = f(y_true, y_pred, k)

        # Remove unwanted dimensions if any
        outputs = np.squeeze(raw_outputs)
        if not outputs.shape:
            return outputs.item()
        return outputs

    return wrapper
