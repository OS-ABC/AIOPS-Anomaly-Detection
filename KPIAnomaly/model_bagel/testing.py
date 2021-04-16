import model_bagel
import numpy as np

from sklearn.metrics import precision_recall_curve
from typing import Sequence, Tuple, Dict, Optional


def _adjust_scores(labels: np.ndarray,
                   scores: np.ndarray,
                   delay: Optional[int] = None,
                   inplace: bool = False) -> np.ndarray:
    if np.shape(scores) != np.shape(labels):
        raise ValueError('`labels` and `scores` must have same shape')
    if delay is None:
        delay = len(scores)
    splits = np.where(labels[1:] != labels[:-1])[0] + 1
    is_anomaly = labels[0] == 1
    adjusted_scores = np.copy(scores) if not inplace else scores
    pos = 0
    for part in splits:
        if is_anomaly:
            ptr = min(pos + delay + 1, part)
            adjusted_scores[pos: ptr] = np.max(adjusted_scores[pos: ptr])
            adjusted_scores[ptr: part] = np.maximum(adjusted_scores[ptr: part], adjusted_scores[pos])
        is_anomaly = not is_anomaly
        pos = part
    part = len(labels)
    if is_anomaly:
        ptr = min(pos + delay + 1, part)
        adjusted_scores[pos: part] = np.max(adjusted_scores[pos: ptr])
    return adjusted_scores


def _ignore_missing(series_list: Sequence, missing: np.ndarray) -> Tuple[np.ndarray, ...]:
    ret = []
    for series in series_list:
        series = np.copy(series)
        ret.append(series[missing != 1])
    return tuple(ret)


def _best_f1score(labels: np.ndarray, scores: np.ndarray) -> Tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true=labels, probas_pred=scores)
    f1score = 2 * precision * recall / np.clip(precision + recall, a_min=1e-8, a_max=None)

    best_threshold = thresholds[np.argmax(f1score)]
    best_precision = precision[np.argmax(f1score)]
    best_recall = recall[np.argmax(f1score)]

    return best_threshold, best_precision, best_recall, np.max(f1score)


def get_test_results(labels: np.ndarray,
                     scores: np.ndarray,
                     missing: np.ndarray,
                     window_size: int,
                     delay: Optional[int] = None) -> Dict:
    labels = labels[window_size - 1:]
    scores = scores[window_size - 1:]
    missing = missing[window_size - 1:]
    adjusted_scores = _adjust_scores(labels=labels, scores=scores, delay=delay)
    adjusted_labels, adjusted_scores = _ignore_missing([labels, adjusted_scores], missing=missing)
    threshold, precision, recall, f1score = _best_f1score(labels=adjusted_labels, scores=adjusted_scores)
    return {'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1score': f1score}


class KPIStats:

    def __init__(self, kpi: model_bagel.data.KPI):
        self.num_points = len(kpi.values)
        self.num_missing = len(kpi.missing[kpi.missing == 1])
        self.num_anomaly = len(kpi.labels[kpi.labels == 1])
        self.missing_rate = self.num_missing / self.num_points
        self.anomaly_rate = self.num_anomaly / self.num_points


def get_kpi_stats(*kpis: model_bagel.data.KPI) -> Tuple[KPIStats, ...]:
    ret = []
    for kpi in kpis:
        ret.append(KPIStats(kpi))
    return tuple(ret)
