"""Different types of data channels."""

import bisect
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial

import numpy as np
from beartype.typing import Any, Callable, Optional, Sequence, Union, cast
from jaxtyping import Num, UInt
from deepradar._compat.roverd import Dataset

from . import transforms

#: Any type which can be used as an index
Index = Union[np.integer, int]


class Channel(ABC):
    """Sensor stream.

    Channels must implement `_index`, which indexes into the channel by its
    native index; index alignments and data transformations are handled by the
    base class, and data transformations.

    Data can also be loaded as a time window relative to the target index with
    a `(past, future)` offset, where `past` and `future` indicate additional
    samples to include relative to the specified index::

        transform(data[idx - left, idx + right + 1])

    Note that this results in `past + future + 1` samples being read.

    NOTE: random accesses (e.g. `idx` in arbitrary order) must be supported.

    Args:
        dataset: path to dataset.
        indices: index transformation from trace to channel index; if `None`,
            no transformation is applied.
        transform: list of :py:class:`Transform` to apply to the data.
        window: past and future samples to include as a `(past, future)` offset
            relative to the current index; if `None`, single samples are
            loaded with the time dimension collapsed.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]] = None,
        transform: list[Callable[[str], transforms.Transform]] = [],
        window: Optional[Sequence[int]] = None
    ) -> None:
        self._transforms = [tf(dataset) for tf in transform]
        self._indices = indices

        if window is None:
            self._window = (0, 0)
            self._squeeze = True
        else:
            assert len(window) == 2
            self._window = cast(tuple[int, int], tuple(window))
            self._squeeze = False

    @abstractmethod
    def _index(self, idx: Index) -> Any:
        pass

    def index(self, idx: Index, aug: dict[str, Any] = {}) -> Any:
        """Index into sensor stream.

        Args:
            idx: trace index. The caller should guarantee that it is in bounds
                for this channel.
            aug: data augmentations to apply; is passsed to each
                :py:class:`Transform`, which is responsible for picking out
                relevant keys.

        Returns:
            Loaded and transformed data corresponding to the global `idx`.
        """
        if self._indices is None:
            data = self._index(idx)
        else:
            data = self._index(self._indices[idx])

        for tf in self._transforms:
            data = tf(data, aug=aug, idx=int(idx))

        return data[0] if self._squeeze else data

    @classmethod
    def from_config(
        cls, dataset: str, indices: Optional[Num[np.ndarray, "N"]],
        transform: list[dict] = [], **kwargs
    ) -> "Channel":
        """Create channel from config.

        Args:
            dataset: path to dataset.
            indices: trace to sensor index conversion.
            transform: list of transformations to apply.
            kwargs: passthrough for channel type-specific configuration.
        """
        return cls(
            dataset=dataset, indices=indices, transform=[
                partial(getattr(transforms, tf["name"]), **tf["args"])
                for tf in transform],
            **kwargs)


class RawChannel(Channel):
    """Generic N-d time series stream in `red-rover` format.

    Args:
        dataset: dataset path.
        sensor: sensor name.
        channel: channel within sensor.
        transform: list of :class:`Transform` to apply to the data.
        window: past and future samples to include as a `(past, future)` offset
            relative to the current index; see :py:class:`.Channel`.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]],
        sensor: str, channel: str,
        transform: list[Callable[[str], transforms.Transform]] = [],
        window: Optional[Sequence[int]] = None
    ) -> None:
        super().__init__(
            dataset=dataset, indices=indices,
            transform=transform, window=window)
        self.channel = Dataset(dataset)[sensor][channel]

    def _index(self, idx: Index) -> Num[np.ndarray, "T ..."]:
        try:
            past, future = self._window
            return self.channel.read(
                int(idx) - past, samples=future + past + 1)
        except IndexError as e:
            print(self.channel, idx, self._window)
            raise(e)


class NPChannel(Channel):
    """N-d time series stream stored in a numpy array.

    NOTE: the numpy array is fully loaded into (main) memory by this loader.

    Args:
        dataset: dataset path.
        path: path of file within dataset.
        keys: keys to load from the `.npz` archive. If `keys` is a str, single
            arrays are yielded instead of dicts of arrays.
        transform: list of :class:`Transform` to apply to the data.
        window: past and future samples to include as a `(past, future)` offset
            relative to the current index; see :py:class:`.Channel`.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]],
        path: str, keys: list[str] | str = [],
        transform: list[Callable[[str], transforms.Transform]] = [],
        window: Optional[tuple[int, int]] = None
    ) -> None:
        super().__init__(
            dataset=dataset, indices=indices,
            transform=transform, window=window)
        npz = np.load(os.path.join(dataset, path))

        if isinstance(keys, str):
            self.arr = npz[keys]
        else:
            self.arr = {k: npz[k] for k in keys}

        self.index_map: Optional[UInt[np.ndarray, "N2"]] = None
        if "mask" in npz:
            mask = npz["mask"]
            self.index_map = np.zeros(mask.shape, dtype=np.uint32)
            self.index_map[mask] = np.arange(np.sum(mask), dtype=np.uint32)

    def _index(self, idx: Index) -> Any:
        if self.index_map is not None:
            idx = self.index_map[idx]

        past, future = self._window
        ii = slice(idx - past, idx + future + 1, None)
        if isinstance(self.arr, dict):
            return {k: v[ii] for k, v in self.arr.items()}
        else:
            return self.arr[ii]


class PrecomputedRadarChannel(Channel):
    """Radar channel backed by per-trace precomputed numpy shards.

    The cache is keyed by trace-local sample position. The public channel API
    still receives the aligned sensor index from ``RoverTrace``, so this class
    rebuilds the trace's radar-index-to-cache-position map from
    ``_fusion/indices.npz`` before reading the corresponding shard.
    """

    def __init__(
        self, dataset: str, indices: Optional[UInt[np.ndarray, "N"]],
        cache_root: str, sensor: str = "radar",
        index_file: str = "_fusion/indices.npz", lru_size: int = 8,
        transform: list[Callable[[str], transforms.Transform]] = [],
        window: Optional[Sequence[int]] = None
    ) -> None:
        if window is not None:
            raise ValueError("PrecomputedRadarChannel does not support windows.")

        super().__init__(
            dataset=dataset, indices=indices,
            transform=transform, window=None)

        self.cache_root = os.path.abspath(cache_root)
        self.dataset = os.path.abspath(dataset)
        self.lru_size = max(1, int(lru_size))
        self._shard_cache: OrderedDict[str, np.ndarray] = OrderedDict()

        root_manifest_path = os.path.join(self.cache_root, "manifest.json")
        with open(root_manifest_path, "r", encoding="utf-8") as f:
            root_manifest = json.load(f)
        if root_manifest.get("complete") is not True:
            raise ValueError(
                f"Incomplete precomputed cache root: {root_manifest_path}")

        full_trace_counts = root_manifest.get("full_trace_counts")
        processed_samples = root_manifest.get("processed_samples")
        if not isinstance(full_trace_counts, dict) or processed_samples is None:
            raise ValueError(
                "Precomputed cache root cannot prove full-data completion: "
                f"{root_manifest_path}")
        expected_samples = sum(int(v) for v in full_trace_counts.values())
        if int(processed_samples) != expected_samples:
            raise ValueError(
                "Limited precomputed cache cannot be used for normal "
                f"training: processed={processed_samples}, "
                f"expected={expected_samples}, manifest={root_manifest_path}")

        data_root = root_manifest.get("data_root")
        if data_root is not None:
            data_root = os.path.abspath(data_root)
            try:
                trace = os.path.relpath(self.dataset, data_root)
            except ValueError:
                trace = os.path.join(
                    os.path.basename(os.path.dirname(self.dataset)),
                    os.path.basename(self.dataset))
        else:
            trace = os.path.join(
                os.path.basename(os.path.dirname(self.dataset)),
                os.path.basename(self.dataset))
        self.trace = trace.replace(os.sep, "/")

        manifest_path = os.path.join(
            self.cache_root, "traces", *self.trace.split("/"),
            "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)
        if self.manifest.get("complete") is not True:
            raise ValueError(f"Incomplete precomputed trace cache: {manifest_path}")
        processed = int(self.manifest.get("processed", -1))
        count = int(self.manifest.get("count", -1))
        limit = int(self.manifest.get("limit", -1))
        if processed != count or limit != count:
            raise ValueError(
                "Limited precomputed trace cache cannot be used for normal "
                f"training: trace={self.trace}, processed={processed}, "
                f"limit={limit}, count={count}, manifest={manifest_path}")

        self.shards = self.manifest["shards"]
        self.starts = [int(s["start"]) for s in self.shards]
        self.ends = [int(s["start"]) + int(s["samples"]) for s in self.shards]

        npz = np.load(os.path.join(self.dataset, index_file))
        sensors = [
            s.decode("utf-8") if isinstance(s, bytes) else str(s)
            for s in npz["sensors"]]
        sensor_col = sensors.index(sensor)
        full_indices = npz["indices"][:processed, sensor_col]
        self._position_by_sensor = {
            int(sensor_index): int(position)
            for position, sensor_index in enumerate(full_indices)}

    def _open_shard(self, rel: str) -> np.ndarray:
        arr = self._shard_cache.get(rel)
        if arr is not None:
            self._shard_cache.move_to_end(rel)
            return arr

        path = os.path.join(
            self.cache_root, "traces", *self.trace.split("/"), rel)
        arr = np.load(path, mmap_mode="r")
        self._shard_cache[rel] = arr
        if len(self._shard_cache) > self.lru_size:
            self._shard_cache.popitem(last=False)
        return arr

    def _index(self, idx: Index) -> Any:
        position = self._position_by_sensor.get(int(idx))
        if position is None:
            raise IndexError(
                f"Radar sensor index {int(idx)} is not present in "
                f"precomputed trace cache {self.trace}.")

        shard_idx = bisect.bisect_right(self.starts, position) - 1
        if shard_idx < 0 or position >= self.ends[shard_idx]:
            raise IndexError(
                f"Precomputed cache position {position} is out of bounds "
                f"for trace {self.trace}.")

        shard = self.shards[shard_idx]
        arr = self._open_shard(shard["file"])
        local = position - int(shard["start"])
        return np.asarray(arr[local], dtype=np.float32)

    def index(self, idx: Index, aug: dict[str, Any] = {}) -> Any:
        if self._indices is None:
            data = self._index(idx)
        else:
            data = self._index(self._indices[idx])

        for tf in self._transforms:
            data = tf(data, aug=aug, idx=int(idx))

        return data


class MetaChannel(Channel):
    """Sensor metadata dummy data.

    Always returns `None`.
    """

    def _index(self, idx: Index) -> Any:
        return None
