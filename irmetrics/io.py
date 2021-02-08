import warnings
import numpy as np

from functools import wraps
from irmetrics.relevance import multilabel


def to_scalar(x):
    if not x.shape:
        return x.item()
    return x


def ensure_inputs(y_true, y_pred, k=None):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)

    # `np.atleast_2d` adds a new axis as a batch dimension
    # thus y_pred[n_samples] is converted to y_pred[1, n_samples]
    # In other words, this condition allows passing y_true of [n_samples,]
    if y_true.shape[0] == 1 and y_true.shape[1] == y_pred.shape[0]:
        y_true = y_true.T

    # Take at most k labels
    y_true = y_true[:, :k]
    y_pred = y_pred[:, :k]
    return y_true, y_pred


def _validate_unique(f):
    @wraps(f)
    def wrapper(y_true, y_pred, k=None, relevance=multilabel, **kwargs):
        repeat_count = (y_pred[:, :, None] == y_pred[:, None]).sum(axis=-1)
        if np.any(repeat_count > 1):
            message = (
                "Repeated predictions detected. "
                "This is an error unless the predictions are padded. "
                "Use np.nan as a padding token to suppress the warning for "
                "integer labels."
            )
            warnings.warn(message, RuntimeWarning)
        return f(y_true, y_pred, k, relevance=relevance, **kwargs)

    return wrapper


def _ensure_io(f):
    @wraps(f)
    def wrapper(y_true, y_pred, k=None, relevance=multilabel, **kwargs):
        # Ensure (n_samples, n_labels) shapes for the inputs
        y_true_, y_pred_ = ensure_inputs(y_true, y_pred, k)

        # Calculate the measure
        raw_outputs = f(y_true_, y_pred_, k, relevance=relevance, **kwargs)

        # Remove unwanted dimensions if any
        return to_scalar(np.squeeze(raw_outputs))

    return wrapper
