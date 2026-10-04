"""Private prefix/tail views for unpublished retarded-history data.

Scalar indexing and contiguous slices touch only the requested rows. NumPy
conversion is available for legacy providers, but the resolved light-cone path
never materializes the accepted prefix. The prefix stop is captured explicitly;
later growth of the underlying storage does not lengthen a provisional view.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np


class HistoryPrefixView(Sequence[Any]):
    """A stable prefix followed by a privately owned replacement tail."""

    def __init__(self, prefix: Any, stop: int, tail: Any) -> None:
        self.prefix = prefix
        self.stop = int(stop)
        self.tail = tail
        self.shape: tuple[int, ...] = (
            self.stop + len(tail),
            *getattr(tail, "shape", ())[1:],
        )
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.dtype = getattr(tail, "dtype", None)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, key: Any) -> Any:
        row, rest = (key[0], key[1:]) if isinstance(key, tuple) else (key, ())
        if isinstance(row, slice):
            start, stop, stride = row.indices(len(self))
            if stride != 1:
                return np.asarray(self)[key]
            if stop <= start:
                value = self.tail[:0]
            elif stop <= self.stop:
                value = self.prefix[start:stop]
            elif start >= self.stop:
                value = self.tail[start - self.stop : stop - self.stop]
            else:
                value = HistoryPrefixView(
                    self.prefix[start : self.stop],
                    self.stop - start,
                    self.tail[: stop - self.stop],
                )
            if rest:
                if isinstance(value, HistoryPrefixView):
                    return HistoryPrefixView(
                        value.prefix[(slice(None), *rest)],
                        value.stop,
                        value.tail[(slice(None), *rest)],
                    )
                return value[(slice(None), *rest)]
            return value
        index = int(row)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("history row out of bounds")
        value = (
            self.prefix[index] if index < self.stop else self.tail[index - self.stop]
        )
        return value[rest] if rest else value

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        result = np.concatenate(
            (np.asarray(self.prefix[: self.stop]), np.asarray(self.tail)), axis=0
        )
        return np.asarray(result, dtype=dtype)

    def searchsorted(
        self, value: Any, side: Literal["left", "right"] = "left", sorter: Any = None
    ) -> Any:
        if sorter is not None or np.ndim(value):
            return np.asarray(self).searchsorted(value, side=side, sorter=sorter)
        lower, upper = 0, len(self)
        while lower < upper:
            middle = (lower + upper) // 2
            sample = self[middle]
            if sample < value or (side == "right" and sample == value):
                lower = middle + 1
            else:
                upper = middle
        return lower
