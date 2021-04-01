import uuid
import numpy as np
import tensorflow as tf

from typing import Sequence, Tuple, Optional


class KPI:

    def __init__(self,
                 timestamps: Sequence,
                 values: Sequence,
                 labels: Optional[Sequence] = None,
                 missing: Optional[Sequence] = None,
                 name: Optional[str] = None):
        self.timestamps = np.asarray(timestamps, dtype=np.int)
        self.values = np.asarray(values, dtype=np.float32)

        if labels is None:
            self.labels = np.zeros(np.shape(values), dtype=np.int)
        else:
            self.labels = np.asarray(labels, dtype=np.int)

        if missing is None:
            self.missing = np.zeros(np.shape(values), dtype=np.int)
        else:
            self.missing = np.asarray(missing, dtype=np.int)

        if name is None:
            self.name = str(uuid.uuid4())
        else:
            self.name = name

        self.labels[self.missing == 1] = 0

    @property
    def abnormal(self) -> np.ndarray:
        return np.logical_or(self.missing, self.labels).astype(np.int)

    def complete_timestamp(self):
        src_idx = np.argsort(self.timestamps)
        timestamp_sorted = self.timestamps[src_idx]
        intervals = np.unique(np.diff(timestamp_sorted))
        interval = np.min(intervals)
        if interval == 0:
            raise ValueError('Duplicated values in `timestamp`')
        for itv in intervals:
            if itv % interval != 0:
                raise ValueError('Not all intervals in `timestamp` are multiples of the minimum interval')

        length = (timestamp_sorted[-1] - timestamp_sorted[0]) // interval + 1
        new_timestamps = np.arange(timestamp_sorted[0], timestamp_sorted[-1] + interval, interval, dtype=np.int)
        new_values = np.zeros([length], dtype=self.values.dtype)
        new_labels = np.zeros([length], dtype=self.labels.dtype)
        new_missing = np.ones([length], dtype=self.missing.dtype)

        dst_idx = np.asarray((timestamp_sorted - timestamp_sorted[0]) // interval, dtype=np.int)
        new_values[dst_idx] = self.values[src_idx]
        new_labels[dst_idx] = self.labels[src_idx]
        new_missing[dst_idx] = self.missing[src_idx]

        self.timestamps = new_timestamps
        self.values = new_values
        self.labels = new_labels
        self.missing = new_missing

    def split(self, ratios: Sequence) -> Tuple['KPI', ...]:
        if abs(1.0 - sum(ratios)) > 1e-4:
            raise ValueError('The sum of `ratios` must be 1')
        partition = np.asarray(np.cumsum(np.asarray(ratios, dtype=np.float32)) * len(self.values), dtype=np.int)
        partition[-1] = len(self.values)
        partition = np.concatenate(([0], partition))
        ret = []
        for low, high in zip(partition[:-1], partition[1:]):
            ret.append(KPI(timestamps=self.timestamps[low:high],
                           values=self.values[low:high],
                           labels=self.labels[low:high],
                           missing=self.missing[low:high],
                           name=self.name))
        return tuple(ret)

    def standardize(self, mean: Optional[float] = None, std: Optional[float] = None) -> Tuple['KPI', float, float]:
        if (mean is None) != (std is None):
            raise ValueError('`mean` and `std` must be both None or not None')
        if mean is None:
            mean = self.values.mean()
            std = self.values.std()
        values = (self.values - mean) / std
        kpi = KPI(timestamps=self.timestamps, values=values, labels=self.labels, missing=self.missing, name=self.name)
        return kpi, mean, std

    def use_labels(self, rate: float = 1.) -> 'KPI':
        if not 0. <= rate <= 1.:
            raise ValueError('`rate` must be in [0, 1]')
        if rate == 0.:
            return KPI(timestamps=self.timestamps, values=self.values, labels=None, missing=self.missing,
                       name=self.name)
        if rate == 1.:
            return self
        labels = np.copy(self.labels)
        anomaly_idx = labels.nonzero()[0]
        drop_idx = np.random.choice(anomaly_idx, round((1 - rate) * len(anomaly_idx)), replace=False)
        labels[drop_idx] = 0
        return KPI(timestamps=self.timestamps, values=self.values, labels=labels, missing=self.missing, name=self.name)

    def no_labels(self) -> 'KPI':
        return self.use_labels(0.)


class KPIDataset:

    def __init__(self, kpi: KPI, window_size: int, missing_injection_rate: float = 0.):
        self._window_size = window_size
        self._missing_injection_rate = missing_injection_rate

        self._one_hot_minute = self._one_hot(self._ts2minute(kpi.timestamps), depth=60)
        self._one_hot_hour = self._one_hot(self._ts2hour(kpi.timestamps), depth=24)
        self._one_hot_weekday = self._one_hot(self._ts2weekday(kpi.timestamps), depth=7)

        self._value_windows = self._to_windows(kpi.values)
        self._label_windows = self._to_windows(kpi.labels)
        self._normal_windows = self._to_windows(1 - kpi.abnormal)

        self._time_code = []
        self._values = []
        self._normal = []
        for i in range(len(self._value_windows)):
            values = np.copy(self._value_windows[i]).astype(np.float32)
            labels = np.copy(self._label_windows[i]).astype(np.int)
            normal = np.copy(self._normal_windows[i]).astype(np.int)

            injected_missing = np.random.binomial(1, self._missing_injection_rate, np.shape(values[normal == 1]))
            normal[normal == 1] = 1 - injected_missing
            values[np.logical_and(normal == 0, labels == 0)] = 0.

            time_index = i + self._window_size - 1
            time_code = np.concatenate(
                [self._one_hot_minute[time_index], self._one_hot_hour[time_index], self._one_hot_weekday[time_index]],
                axis=-1
            )

            self._time_code.append(time_code)
            self._values.append(values)
            self._normal.append(normal)

    def _to_windows(self, series: np.ndarray) -> np.ndarray:
        return np.lib.stride_tricks.as_strided(
            series,
            shape=(np.size(series, 0) - self._window_size + 1, self._window_size),
            strides=(series.strides[-1], series.strides[-1])
        )

    @staticmethod
    def _ts2hour(ts: np.ndarray) -> np.ndarray:
        return (ts % 86400) // 3600

    @staticmethod
    def _ts2minute(ts: np.ndarray) -> np.ndarray:
        return ((ts % 86400) % 3600) // 60

    @staticmethod
    def _ts2weekday(ts: np.ndarray) -> np.ndarray:
        return ((ts // 86400) + 4) % 7

    @staticmethod
    def _one_hot(indices: Sequence, depth: int) -> np.ndarray:
        return np.eye(depth)[indices]

    @property
    def time_code(self) -> np.ndarray:
        return np.asarray(self._time_code, dtype=np.float32)

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._values, dtype=np.float32)

    @property
    def normal(self) -> np.ndarray:
        return np.asarray(self._normal, dtype=np.float32)

    def to_tensorflow(self) -> tf.data.Dataset:
        return tf.data.Dataset.from_tensor_slices((self.values, self.time_code, self.normal))
