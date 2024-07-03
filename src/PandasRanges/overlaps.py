import sys
import heapq
from typing import Tuple, Union, List, Optional

import numpy as np
from numba import jit
from numpy import ndarray
from pandas import Series

from numba import typed, types, int64


_minqueue_kv_type = types.UniTuple(types.int64, 2)


@jit(nopython=True)
def minqueue_new():  #TODO: can this be megred with minqueue_init?
    return typed.List.empty_list(_minqueue_kv_type)


def minqueue_init(data: List[Tuple[int, int]]):
    if len(data) == 0:
        return typed.List.empty_list(_minqueue_kv_type)
    else:
        arr = [(int64(v), int64(k)) for k, v in data]
        heapq.heapify(arr)
        return typed.List(arr)


@jit(nopython=True)
def minqueue_len(arr):
    return len(arr)


@jit(nopython=True)
def minqueue_push(arr, key: int, value: int):
    heapq.heappush(arr, (int64(value), int64(key)))


@jit(nopython=True)
def minqueue_first(arr) -> Tuple[int, int]:
    value, key = arr[0]
    return key, value


@jit(nopython=True)
def minqueue_min(arr, default=None):
    if default is not None and len(arr) == 0:
        return default
    return arr[0][0]


@jit(nopython=True)
def minqueue_pop(arr) -> Tuple[int, int]:
    value, key = heapq.heappop(arr)
    return key, value


@jit(nopython=True)
def minqueue_get(arr, index: int):
    value, key = arr[index]
    return key, value


@jit(nopython=True)
def overlapping_pairs(start_a: ndarray, end_a: ndarray, start_b: ndarray, end_b: ndarray, sort_result: bool = True) \
        -> ndarray:
    n_streams = 2  # "stream" = "list of ranges"
    sentinel = sys.maxsize
    starts = [start_a, start_b]
    ends = [end_a, end_b]
    overlap_indices = []
    open_ranges = [minqueue_new() for _ in range(n_streams)]  # "open" = "intersecting the current position"
    lengths = np.array([len(s) for s in starts], dtype='int')
    if min(lengths) == 0:  # empty intersection
        return np.empty((0, 2), dtype='int')
    stream_pos = np.array([0 for _ in range(n_streams)], dtype='int')
    next_starts = np.array([s[0] for s in starts], dtype='int')
    min_open_ends = np.array([sentinel for _ in range(n_streams)], dtype='int')  # coordinates are > 0
    n_open = n_streams  # n_open = #streams with data + #min_open_ends queues with data
    while n_open > 0:
        if next_starts.min() < min_open_ends.min():  # "=" is important here: starts before ends
            # next point is the start of a region -> we open that region
            stream_no = next_starts.argmin()
            i = stream_pos[stream_no]  # index of current region in stream
            end = ends[stream_no][i]
            # end == start for an empty region
            if end < next_starts[stream_no]:
                raise ValueError(f"Malformed region in argument {stream_no + 1}: f{next_starts[stream_no]}-{end}")
            if len(open_ranges[stream_no]) == 0:
                n_open += 1
            # open the region
            open_ranges[stream_no].push(i, end)
            min_open_ends[stream_no] = open_ranges[stream_no].min()
            # advance in the stream
            i += 1
            if i < lengths[stream_no]:
                stream_pos[stream_no] = i
                start = starts[stream_no][i]
                if start < next_starts[stream_no]:
                    raise ValueError(f"Unsorted data in argument {stream_no + 1}: {start} < {next_starts[stream_no]}")
                next_starts[stream_no] = start
            else:  # end of input -> close stream
                next_starts[stream_no] = sentinel
                n_open -= 1
        else:  # next_starts.min() >= min_open_ends.min()
            # next point is the end of a region -> we close that region
            stream_no = min_open_ends.argmin()
            i, end = open_ranges[stream_no].pop()
            other_stream_idx = (stream_no + 1) % 2
            # add overlap indices
            other_open = open_ranges[other_stream_idx]
            for k in range(len(other_open)):
                j, _ = other_open.get(k)
                res = (i, j) if stream_no == 0 else (j, i)
                overlap_indices.append(res)
            if len(open_ranges[stream_no]) == 0:
                n_open -= 1
                min_open_ends[stream_no] = sentinel
            else:
                min_open_ends[stream_no] = open_ranges[stream_no].min()

    if sort_result:
        overlap_indices.sort()

    return np.array(overlap_indices, dtype='int').reshape(len(overlap_indices), 2)


def overlapping_clusters(starts: ndarray, ends: ndarray, ascending: bool = True, dtype: str='int') \
        -> ndarray:
    length = len(starts)
    assert len(ends) == length
    if length == 0:  # empty set
        return np.empty(0, dtype=dtype)
    cluster_ids = np.empty(length, dtype=dtype)
    if not ascending:  # TODO: better implementation, without copying/views
        starts = starts[::-1]
        ends = ends[::-1]
    _overlapping_clusters_impl(starts, ends, cluster_ids, ascending)
    if not ascending:
        cluster_ids[:] = cluster_ids[::-1]
    return cluster_ids


@jit(nopython=True)
def _overlapping_clusters_impl(starts: ndarray, ends: ndarray, cluster_ids: np.ndarray, ascending: bool):
    sentinel = sys.maxsize
    length = len(starts)
    open_ranges = MinQueue(None)  # "open" = "intersecting the current position"
    idx = 0
    i_cluster = 0
    next_start = starts[0]
    while idx < length or len(open_ranges) > 0:
        if next_start < open_ranges.min(default=sentinel):  # lack of "=" is important here
            # next point is the start of a region -> we open that region
            end = ends[idx]
            if end < next_start:
                raise ValueError(f"Malformed region at {idx}: f{next_start}-{end}")
            cluster_ids[idx] = i_cluster
            open_ranges.push(idx, end)  # open the region
            idx += 1
            if idx < length:
                start = starts[idx]
                if start < next_start:
                    raise ValueError(f"Unsorted data at {idx}: {start} < {next_start}")
                next_start = start
            else:
                next_start = sentinel
        else:  # next_start > open_ranges.min()
            # next point is the end of a region -> we close that region
            open_ranges.pop()
            if len(open_ranges) == 0:
                i_cluster += 1


@jit(nopython=True)
def mergesort_ranges_indices(starts_a: ndarray, ends_a: ndarray, starts_b: ndarray, ends_b: ndarray) \
        -> Tuple[ndarray, ndarray]:
    sentinel = sys.maxsize
    length_a = len(starts_a)
    assert len(ends_a) == length_a
    length_b = len(starts_b)
    assert len(ends_b) == length_b
    idx_a = 0
    idx_b = 0
    idx_res = 0
    total_length = length_a + length_b
    indices_a = np.empty(length_a, dtype='int')
    indices_b = np.empty(length_b, dtype='int')
    while idx_res < total_length:
        s_a = starts_a[idx_a] if idx_a < length_a else sentinel
        s_b = starts_b[idx_b] if idx_b < length_b else sentinel
        if s_a < s_b:  # Take from A
            indices_a[idx_a] = idx_res
            idx_a += 1
        elif s_a > s_b:  # Take from B
            indices_b[idx_b] = idx_res
            idx_b += 1
        else:  # s_a == s_b
            e_a = ends_a[idx_a] if idx_a < length_a else sentinel
            e_b = ends_b[idx_b] if idx_b < length_b else sentinel
            if e_a <= e_b:  # Take from A
                indices_a[idx_a] = idx_res
                idx_a += 1
            else:  # Take from B
                indices_b[idx_b] = idx_res
                idx_b += 1
        idx_res += 1

    return indices_a, indices_b


# TODO: test
@jit(nopython=True)
def target_indices_to_permutation(indices_a: ndarray, indices_b: ndarray) -> ndarray:
    n = len(indices_a) + len(indices_b)
    perm = np.empty(n, dtype='int')
    perm[indices_a] = np.arange(len(indices_a))
    perm[indices_b] = np.arange(len(indices_a), n)
    return perm


def check_if_sorted(
        start: Union[ndarray, Series],
        end: Union[Series, ndarray, None] = None,
        ascending=True, raise_msg=None, exc_type=ValueError
):
    if isinstance(end, Series):
        end = end.to_numpy()
    if not start.ndim == 1:
        raise ValueError('`start` is not a vector')
    if end is None:
        i = get_first_unsorted_index_in_points(start, ascending=ascending)
    else:
        if not end.ndim == 1:
            raise ValueError('`end` is not a vector')
        if start.shape != end.shape:
            raise ValueError('Uneven start and end arrays')
        i = get_first_unsorted_index_in_ranges(start, end, ascending=ascending)
    if i != -1 and exc_type is not None:
        _dir = 'increasing' if ascending else 'decreasing'
        msg = f'Data not sorted in {_dir} order at index {i}'
        if raise_msg is not None:
            msg += f': "{raise_msg}"'
        raise exc_type(msg)
    return i


@jit(nopython=True)
def get_first_unsorted_index_in_points(points: ndarray, ascending=True):
    i = 1
    i_end = len(points)
    mult = 1 if ascending else -1
    while i < i_end:
        diff = points[i] - points[i - 1]
        if diff * mult < 0:
            return i
        i += 1
    return -1


@jit(nopython=True)
def get_first_unsorted_index_in_ranges(start: ndarray, end: ndarray, ascending=True):
    i = 1
    i_end = len(start)
    mult = 1 if ascending else -1
    while i < i_end:
        diff = start[i] - start[i - 1]
        if diff * mult < 0 or (diff == 0 and (end[i] - end[i - 1]) * mult < 0):
            return i
        i += 1
    return -1
