import numpy as np


def rr(y_true, y_pred, k=20):
    y_true, y_pred = np.atleast_2d(y_true, y_pred)
    y_true = y_true.T[:, :k]
    y_pred = y_pred[:, :k]

    relevant = y_true == y_pred
    index = relevant.argmax(-1)
    rrs = np.squeeze(relevant.any(-1) / (index + 1))
    if not rrs.shape:
        return rrs.item()
    return rrs
