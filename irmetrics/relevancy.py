

def unilabel(y_true, y_pred):
    return y_true == y_pred


def multilabel(y_true, y_pred):
    return (y_pred[:, :, None] == y_true[:, None]).any(axis=-1)
