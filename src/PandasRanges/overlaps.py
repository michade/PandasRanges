import sys
from typing import Tuple, Union, List, Optional

import numpy as np
from numpy import ndarray
from pandas import Series

from .overlaps_cy import *


###############################################################
### Overlapping ranges
###############################################################


def count_overlapping_pairs(
    start_a: ndarray, end_a: ndarray,
    start_b: ndarray, end_b: ndarray,
    expand: Union[int, Tuple[int, int]] = 0,
    min_overlap: int = 0
):
    assert len(start_a) == len(end_a)
    assert len(start_b) == len(end_b)
    if not isinstance(expand, tuple):
        expand = expand, expand    
    expand_start_a, expand_end_a = expand
    expand_end_a -= min_overlap
    return overlapping_pairs_impl(
        None,  # no output
        start_a, end_a, start_b, end_b,
        expand_start_a, expand_end_a
    )


def get_overlapping_pairs(
    start_a: ndarray, end_a: ndarray,
    start_b: ndarray, end_b: ndarray,
    expand: Union[int, Tuple[int, int]] = 0,
    sort_result: bool = True,
    min_overlap: int = 0
) -> ndarray:        
    assert len(start_a) == len(end_a)
    assert len(start_b) == len(end_b)
    n_overlaps = count_overlapping_pairs(
        start_a, end_a, start_b, end_b,
        expand=expand,
        min_overlap=min_overlap
    )
    if n_overlaps == 0:
        return np.empty((0, 2), dtype='int')
    if not isinstance(expand, tuple):
        expand = expand, expand
    expand_start_a, expand_end_a = expand
    expand_end_a -= min_overlap
    indices = np.empty((n_overlaps, 2), dtype='int')
    overlapping_pairs_impl(
        indices,
        start_a, end_a, start_b, end_b,
        expand_start_a, expand_end_a
    )
    if sort_result:
        idx = np.lexsort((indices[:, 1], indices[:, 0]))
        indices = indices[idx]
    return indices


def count_overlapping_clusters(
    starts: ndarray,
    ends: ndarray,
    expand: Union[int, Tuple[int, int]] = 0,
    is_ascending: bool = True
) -> int:
    assert len(starts) == len(ends)
    if not isinstance(expand, tuple):
        expand = expand, expand    
    expand_start, expand_end = expand
    return overlapping_clusters_impl(
        None,  # no output
        starts, ends,
        expand_start, expand_end,
        is_ascending
    )


def get_overlapping_clusters(
    starts: ndarray,
    ends: ndarray,
    expand: Union[int, Tuple[int, int]] = 0,
    is_ascending: bool = True
) -> ndarray:
    assert len(starts) == len(ends)
    if not isinstance(expand, tuple):
        expand = expand, expand    
    expand_start, expand_end = expand
    cluster_ids = np.empty(len(starts), dtype='int')
    overlapping_clusters_impl(
        cluster_ids,
        starts, ends,
        expand_start, expand_end,
        is_ascending
    )
    return cluster_ids


###############################################################
### Utilities
###############################################################


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


def target_indices_to_permutation(indices_a: ndarray, indices_b: ndarray) -> ndarray:
    result = np.empty(len(indices_a) + len(indices_b), dtype='int')
    target_indices_to_permutation_impl(result, indices_a, indices_b)    
    return result


def mergesort_ranges_indices(starts_a: ndarray, ends_a: ndarray, starts_b: ndarray, ends_b: ndarray) \
        -> Tuple[ndarray, ndarray]:
    length_a = len(starts_a)
    assert len(ends_a) == length_a
    length_b = len(starts_b)
    assert len(ends_b) == length_b    
    indices_a = np.empty(length_a, dtype='int')
    indices_b = np.empty(length_b, dtype='int')
    mergesort_ranges_indices_impl(
        indices_a, indices_b,
        starts_a, ends_a, starts_b, ends_b
    )
    return indices_a, indices_b


def get_idx_of_cluster_starts(cluster_ids: ndarray):
    if len(cluster_ids) == 0:
        return np.empty(0, dtype='int')
    n_clusters = cluster_ids[-1] + 1
    result = np.empty(n_clusters, dtype='int')
    get_idx_of_cluster_starts_impl(result, cluster_ids)
    return result
