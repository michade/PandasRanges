from __future__ import annotations

import heapq
from typing import Tuple, Union, Hashable, List

import numpy as np
import pandas as pd


# TODO: what's this?
# import matplotlib
# from matplotlib import pyplot as plt
# from numba import int32
# from numba.experimental import jitclass


class MinQueue(object):
    def __init__(self, data=None):
        if data is not None:
            arr = [(v, k) for k, v in data]
            heapq.heapify(arr)
        else:
            arr = []
        self._arr = arr

    def __len__(self):
        return self._arr.__len__()

    def __iter__(self):
        return iter((k, v) for v, k in self._arr)

    def __str__(self):
        return f'MinQueue[{self._arr}]'

    def __repr__(self):
        return str(self)

    def push(self, key: int, value: int):
        heapq.heappush(self._arr, (value, key))

    def first(self) -> Tuple[int, int]:
        value, key = self._arr[0]
        return key, value

    def min(self, default=None):
        if default is not None and len(self._arr) == 0:
            return default
        return self._arr[0][0]

    def pop(self) -> Tuple[int, int]:
        value, key = heapq.heappop(self._arr)
        return key, value


def pandas_merge_threeway(
        left_df: pd.DataFrame, mid_df: pd.DataFrame, right_df: pd.DataFrame,
        mid_to_left: Union[Hashable, List[Hashable]],
        mid_to_right: Union[Hashable, List[Hashable]],
        left_on: Union[Hashable, List[Hashable]],
        right_on: Union[Hashable, List[Hashable], None] = None,
        inner=True,
        suffixes=('_x', '_m', '_y')
):
    if not isinstance(mid_to_left, list):
        mid_to_left = [mid_to_left]
    if not isinstance(mid_to_right, list):
        mid_to_right = [mid_to_right]
    if not isinstance(left_on, list):
        left_on = [left_on]
    if right_on is None:
        right_on = left_on
    elif not isinstance(right_on, list):
        right_on = [right_on]

    column_sets = [left_df.columns, mid_df.columns, right_df.columns]
    new_left_cols, new_mid_cols, new_right_cols = [
        {
            c: f'{c}{suffixes[i]}' for c in column_sets[i]
            if c in column_sets[(i + 1) % 3] or c in column_sets[(i + 2) % 3]
        }
        for i in range(3)  # left, mid, right
    ]

    def _replace(lst, d):
        return [d.get(s, s) for s in lst]

    left_on = _replace(left_on, new_left_cols)
    right_on = _replace(right_on, new_right_cols)
    mid_to_left = _replace(mid_to_left, new_mid_cols)
    mid_to_right = _replace(mid_to_right, new_mid_cols)

    df = mid_df.rename(columns=new_mid_cols)
    df = pd.merge(
        df, left_df.rename(columns=new_left_cols),
        left_on=mid_to_left, right_on=left_on,
        how='inner' if inner else 'left'
    )
    df = pd.merge(
        df, right_df.rename(columns=new_right_cols),
        left_on=mid_to_right, right_on=right_on,
        how='inner' if inner else 'left'
    )
    return df


# def categorical_cmap(nc, nsc, cmap="tab10", continuous=False):
#     if nc > plt.get_cmap(cmap).N:
#         raise ValueError("Too many categories for colormap.")
#     if continuous:
#         ccolors = plt.get_cmap(cmap)(np.linspace(0,1,nc))
#     else:
#         ccolors = plt.get_cmap(cmap)(np.arange(nc, dtype=int))
#     cols = np.zeros((nc*nsc, 3))
#     for i, c in enumerate(ccolors):
#         chsv = matplotlib.colors.rgb_to_hsv(c[:3])
#         arhsv = np.tile(chsv,nsc).reshape(nsc,3)
#         arhsv[:,1] = np.linspace(chsv[1],0.25,nsc)
#         arhsv[:,2] = np.linspace(chsv[2],1,nsc)
#         rgb = matplotlib.colors.hsv_to_rgb(arhsv)
#         cols[i*nsc:(i+1)*nsc,:] = rgb
#     cmap = matplotlib.colors.ListedColormap(cols)
#     return cmap
#
#
# def test_categorical_cmap():
#     c1 = categorical_cmap(4, 3, cmap="tab10")
#     plt.scatter(np.arange(4 * 3), np.ones(4 * 3) + 1, c=np.arange(4 * 3), s=180, cmap=c1)
#
#     c2 = categorical_cmap(2, 5, cmap="tab10")
#     plt.scatter(np.arange(10), np.ones(10), c=np.arange(10), s=180, cmap=c2)
#
#     c3 = categorical_cmap(2, 19, cmap="tab10")
#     plt.scatter(np.arange(2 * 19), np.ones(2 * 19) - 1, c=np.arange(2 * 19), s=180, cmap=c3)
#
#     plt.margins(y=0.3)
#     plt.xticks([])
#     plt.yticks([0, 1, 2], ["(5, 4)", "(2, 5)", "(4, 3)"])
#     plt.show()


# @jitclass([
#     ('_parent', int32[:]),
#     ('_size', int32[:]),
#     ('_ids', int32[:]),
#     ('_n_clusters', int32),
#     ('_track_cc', int32),
#     ('_cc1_root', int32),
#     ('_cc2_root', int32)
# ])
class DisjointSetPlus(object):
    def __init__(self, size: int, track_cc: int = 1, track_ids=False):
        assert track_cc in (0, 1, 2)
        assert size > 0
        self._parent = np.empty(size, dtype=np.int32)
        self._size = np.empty(size, dtype=np.int32)
        self._ids = np.arange(size if track_ids else 0, dtype=np.int32)
        self._n_clusters = -1
        self._track_cc = -1
        self._cc1_root = -1
        self._cc2_root = -1
        self.reset(track_cc)

    def reset(self, track_cc=None, track_ids=None) -> None:
        assert track_cc in (0, 1, 2)
        _range = np.arange(len(self._parent), dtype=np.int32)
        self._parent[:] = _range
        self._size.fill(1)
        if track_ids is True or len(self._ids):
            self._ids[:] = _range
        if track_cc is not None:
            self._track_cc = track_cc
        if self._track_cc >= 1:
            self._cc1_root = 0
        if self._track_cc >= 2:  # TODO: > 2?
            self._cc2_root = 1
        self._n_clusters = len(self._parent)

    def __len__(self) -> int:
        return len(self._parent)

    @property
    def n_clusters(self) -> int:
        return self._n_clusters

    @property
    def track_cc(self) -> int:
        return self._track_cc

    @property
    def cc1_root(self) -> int:
        return self._cc1_root

    @property
    def cc2_root(self) -> int:
        return -1 if self._n_clusters == 1 else self._cc2_root

    @property
    def cc1_size(self) -> int:
        return self._size[self._cc1_root]

    @property
    def cc2_size(self) -> int:
        return 0 if self._n_clusters == 1 else self._size[self._cc2_root]  # TODO: fix?

    def find(self, x: int) -> int:
        parent = self._parent
        parent_x = parent[x]
        while parent_x != x:
            grandpa_x = parent[parent_x]
            parent[x] = grandpa_x
            x, parent_x = parent_x, grandpa_x
        return x

    def _update_cc1_and_cc2(self, root_x: int, root_y: int, new_size: int) -> None:  # size of root_x >= root_y
        _size = self._size
        if new_size >= _size[self._cc1_root]:  # cc1 is replaced
            if root_x == self._cc1_root:  # cc1 is one of the merged clusters
                if root_y == self._cc2_root:  # cc1 and cc2 merged, need to find replacement for cc2
                    _size[root_x] = -1  # temp. remove cc1 from search
                    self._cc2_root = _size.argmax()
                    _size[root_x] = new_size
                # else: cc1 grows, but cc2 remains
            elif new_size > _size[self._cc1_root]:  # cc1 is replaced, and it becomes new cc2
                self._cc2_root = self._cc1_root
                self._cc1_root = root_x
            else:  # new_size == _size[self.cc1_root]:
                # for stability in this case let's keep the old cc1, and the new cluster becomes cc2
                self._cc2_root = root_x
        elif new_size > _size[self._cc2_root]:  # cc2 is replaced
            self._cc2_root = root_x

    def union(self, x: int, y: int) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        self._n_clusters -= 1

        _size = self._size
        if _size[root_x] < _size[root_y]:
            root_x, root_y = root_y, root_x
        new_size = _size[root_x] + _size[root_y]

        _ids = self._ids
        if len(_ids):
            _id_x = _ids[root_x]
            _id_y = _ids[root_y]
            if _id_x < _id_y:
                _ids[root_y] = _id_x
            else:
                _ids[root_x] = _id_y

        if self._track_cc == 2:
            self._update_cc1_and_cc2(root_x, root_y, new_size)
        elif self._track_cc == 1 and new_size > self._size[self._cc1_root]:  # cc1 is replaced
            self._cc1_root = root_x

        self._parent[root_y] = root_x
        _size[root_x] = new_size
        return True

    def get_cluster(self, x) -> Tuple[int, int]:
        root = self.find(x)
        size = self._size[root]
        return root, size

    def get_size(self, x) -> int:
        root = self.find(x)
        return self._size[root]

    def get_cluster_ids(self) -> np.ndarray:
        ids = np.empty_like(self._ids)
        for i in range(len(ids)):
            ids[i] = self._ids[self.find(i)]
        return ids

    def get_sizes(self) -> np.ndarray:
        sizes = np.empty_like(self._size)
        for i in range(len(sizes)):
            sizes[i] = self._size[self.find(i)]
        return sizes
