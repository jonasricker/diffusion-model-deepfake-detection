import numpy as np
from sklearn.metrics import roc_curve


def pd_at(y_true: np.ndarray, y_score: np.ndarray, x: float) -> float:
    """
    Compute probability of detection (true positive rate) given a certain false positive/alarm rate.
    Corresponds to the y-coordinate on a ROC curve given a certain x-coordinate.
    """
    fpr, tpr, _ = roc_curve(y_true=y_true, y_score=y_score, drop_intermediate=False)
    idx = np.argmin(abs(fpr - x))  # get index where fpr is closest to x

    return tpr[idx]
