from __future__ import annotations

import re
from collections import namedtuple
from typing import Optional, Union, Tuple, Generator, Iterable, Hashable, List, Dict, Any

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import Series, DataFrame

from . import overlaps


def split_string_series(series: Series, sep: str) -> DataFrame:
    lists_col = series.str.split(sep, regex=False)
    lens = lists_col.str.len().unique()
    if len(lens) != 1:
        raise ValueError(f'Different number of elements in series: {lens}')
    df = DataFrame(lists_col.to_list(), index=series.index)
    return df


class RangeSeries(object):
    _DEFAULT_NAMES = ('start', 'end')
    __slots__ = ['_start', '_end']

    #######################################################################################
    # 1. CONSTRUCTION
    #######################################################################################

    def __init__(
            self,
            start: Union[RangeSeries, DataFrame, Series, ndarray, Iterable, int],
            end: Union[Series, ndarray, Iterable, int, None] = None,
            prefix: Optional[str] = None,
            suffix: Optional[str] = None,
            validate: bool = True,
            set_names: Optional[Tuple[Hashable, Hashable]] = None,
            set_index: Optional[pd.Index] = None
    ):
        start, end = RangeSeries._extract_coords_as_series(start, end, prefix, suffix, set_index, set_names)
        self._start = start
        self._end = end
        empty = len(start) == 0
        if not empty:
            if not pd.api.types.is_integer_dtype(start):
                raise TypeError(f"Only integer coordinates supported, got start.dtype == {start.dtype}")
            if not pd.api.types.is_integer_dtype(end):
                raise TypeError(f"Only integer coordinates supported, got end.dtype == {end.dtype}")
            if validate:
                is_row_valid = self.is_valid()
                if not is_row_valid.all():
                    raise ValueError(f"Invalid rows: {is_row_valid.sum()}, first {np.where(is_row_valid)[0]}")
                if not self._start.index.equals(self._end.index):
                    raise ValueError(f"Incompatible indices for start and end.")

    @staticmethod
    def from_records(records: Iterable[Tuple], names=None, validate=True) -> RangeSeries:
        df = pd.DataFrame.from_records(records)
        if len(df) == 0:
            return RangeSeries([], set_names=names)
        if len(df.columns) < 2:
            raise ValueError('Not enough columns provided.')
        if names is None:
            names = [None] * (len(df.columns) - 2) + list(RangeSeries._DEFAULT_NAMES)
        df.columns = names
        if len(names) > 2:
            df = df.set_index(names[:-2])
        return RangeSeries(df[names[-2]], df[names[-1]], validate=validate)

    def __call__(self, df: DataFrame) -> RangeSeries:
        if self._start.name not in df:
            raise ValueError(f'Cannot find start column "{self._start.name}" in dataframe.')
        if self._end.name not in df:
            raise ValueError(f'Cannot find end column "{self._end.name}" in dataframe.')
        return RangeSeries(df.loc[:, self._start.name], df.loc[:, self._end.name])

    @staticmethod
    def _extract_coords_as_series(
            start: Union[RangeSeries, DataFrame, Series, ndarray, Iterable, int],
            end: Union[Series, ndarray, Iterable, int, None],
            prefix: Optional[str],
            suffix: Optional[str],
            index: Optional[pd.Index],
            new_names: Optional[Tuple[str, str]]
    ) -> Tuple[Series, Series]:
        prefix = prefix if prefix is not None else ''
        suffix = suffix if suffix is not None else ''
        def_names = [f'{prefix}{s}{suffix}' for s in RangeSeries._DEFAULT_NAMES]
        names = list(def_names)

        if isinstance(start, RangeSeries):
            ranges = start
            start = ranges.start
            end = ranges.end
        elif isinstance(start, DataFrame):
            df = start
            start = df[names[0]]
            end = df[names[1]]
        elif isinstance(start, Series):
            pass
        elif hasattr(start, '__iter__'):
            if end is None:  # `start` is an iterable of (start, end) coord tuples
                tuples = start
                start = Series([s for s, _ in tuples])
                end = Series([e for _, e in tuples], index=start.index)
                del tuples
            # else: `start` is an iterable of start coords, do nothing
        else:
            start = [start]

        if isinstance(start, Series):
            names[0] = start.name
        if isinstance(end, Series):
            names[1] = end.name
        if names[0] == names[1]:
            names = def_names

        # get index
        if index is None:
            n = 1
            for column in (start, end):
                if isinstance(column, Series):
                    index = column.index
                    break
                elif hasattr(column, '__len__') and not isinstance(column, str):
                    n = len(column)
            else:
                index = pd.RangeIndex(n)

        if end is None:
            end = start  # If both start and end are the same, use default names
        if new_names is not None:
            names = new_names

        if len(start) == 0:
            new_start = Series(start, name=names[0], index=index, dtype=int)
            new_end = Series(end, name=names[1], index=index, dtype=int)            
        else:
            new_start = Series(start, name=names[0], copy=False)
            if new_start.index is not index:
                new_start.index = index
            new_end = Series(end, name=names[1], copy=False)
            if new_end.index is not index:
                new_end.index = index

        return new_start, new_end

    @staticmethod
    def from_string(
            strings: Union[Series, ndarray, Iterable[str], str],
            sep='-', idx_sep=':',
            names: Optional[Tuple[Hashable, Hashable]] = None,
            drop_index: bool = False
    ) -> RangeSeries:
        if isinstance(strings, str):
            strings = [strings]
        if not isinstance(strings, pd.Series):
            drop_index = True
            strings = pd.Series(strings, index=pd.RangeIndex(len(strings)))
        index_df = split_string_series(strings, idx_sep)
        coords_col = index_df.iloc[:, -1]
        coords_df = split_string_series(coords_col, sep)
        if len(coords_df.columns) != 2:
            raise ValueError(f'Wrong number of coordinate separators ({len(coords_df.columns)})')
        if len(index_df.columns) == 1:
            df = coords_df
        else:
            df = pd.concat([index_df.iloc[:, :-1], coords_df], axis=1)
        if names is None:
            names = [f'level_{i}' for i in range(1, len(index_df.columns))] + ['start', 'end']
        df.columns = names
        for col in df.columns:
            wrong_idx = df[col].str.len() == 0
            if wrong_idx.any():
                raise ValueError(f'Empty elements in {col}, idx={np.nonzero(wrong_idx)[0]}')
        if drop_index:
            df = df.reset_index(drop=True)
        if len(names) > 2:
            df = df.set_index(names[:-2], append=not drop_index)
        df[names[-2]] = df[names[-2]].astype(int)
        df[names[-1]] = df[names[-1]].astype(int)
        return RangeSeries(df[names[-2]], df[names[-1]])

    #######################################################################################
    # 2. BASIC PROPERTIES
    #######################################################################################

    @property
    def start(self) -> Series:
        return self._start

    @property
    def end(self) -> Series:
        return self._end

    @property
    def index(self) -> pd.Index:
        return self._start.index


    @property
    def names(self) -> Tuple[Hashable, Hashable]:
        return self._start.name, self._end.name

    @property
    def dtype(self):
        return self._start.dtype

    def __len__(self) -> int:
        return len(self._start)

    def set_index(self, new_index: pd.Index) -> RangeSeries:
        return RangeSeries(self._start, self._end, set_index=new_index)

    def reset_index(self, labels=None, drop=True, keep=None) -> Union[RangeSeries, DataFrame]:
        if labels is None:
            if keep is None:
                labels = self.index.names
        elif keep is not None:
            raise ValueError('Cannot specify  both "labels" and "keep"')
        if keep is not None:
            labels = [col for col in self.index.names if col not in keep]
        if drop:
            start = self._start.reset_index(labels, drop=True)
            end = self._end.reset_index(labels, drop=True)
            return RangeSeries(start, end)
        else:
            df = self._start.reset_index(labels, drop=False)
            df.insert(len(df.columns), self._end.name, self._end.reset_index(labels, drop=True))
            return df

    #######################################################################################
    # 3. ITERATION AND SUBSETS
    #######################################################################################

    def head(self, n: int = 5):
        if n < 0:
            raise ValueError(f'n (={n} must be >= 0')
        elif n == 0:
            return RangeSeries([])
        return RangeSeries(self._start.iloc[:n], self._end.iloc[:n])

    def tail(self, n: int = 5):
        if n < 0:
            raise ValueError(f'n (={n} must be >= 0')
        elif n == 0:
            return RangeSeries([])
        return RangeSeries(self._start.iloc[-n:], self._end.iloc[-n:])

    def itertuples(self, index=True) -> Generator:
        if index:
            _tuple_type = namedtuple('Range', ['Index', str(self._start.name), str(self._end.name)])
            for i in range(len(self)):
                yield _tuple_type(self.index[i], self.start.iat[i], self.end.iat[i])  # params are ok
        else:
            _tuple_type = namedtuple('Range', [str(self._start.name), str(self._end.name)])
            for i in range(len(self)):
                yield _tuple_type(self.start.iat[i], self.end.iat[i])  # params are ok

    def drop_empty(self) -> RangeSeries:
        indices = ~self.is_empty()
        return RangeSeries(self._start[indices], self._end[indices])

    def subset(self, indices):
        return RangeSeries(self._start.iloc[indices], self._end.iloc[indices])

    #######################################################################################
    # 4. CONVERSION TO OTHER TYPES
    #######################################################################################

    def to_frame(self, names=None):
        df = DataFrame({self.start.name: self._start, self.end.name: self.end})
        if names is not None:
            if len(names) != 2:
                raise ValueError("Need to specify names for start and end")
            df.columns = names
        return df

    def __str__(self):
        return self.to_frame().to_string()

    def __repr__(self):
        return repr(self.to_frame())

    def to_string(self, sep='-', idx_sep=':', index=True) -> Series:
        result = self._start.astype(str) + sep + self._end.astype(str)
        if index:
            result = result.reset_index(drop=True)
            df = self.to_frame().reset_index(drop=False)
            for col in df.columns[-3::-1]:
                result = df[col].astype(str) + idx_sep + result
            result.index = self.index
        return result

    def format(self, fmt: str, name: Optional[str] = None) -> Series:
        df = self.to_frame().reset_index(drop=False)
        groups = []
        starts = []
        ends = []
        for m in re.finditer(r'\{(\w+)(:[\w.,-]+)?}', fmt):
            starts.append(m.start())
            ends.append(m.end())
            col_fmt = '{' + m.group(2) + '}' if m.group(2) is not None else None
            groups.append((m.group(1), col_fmt))  # name, format (optional)
        if len(groups) == 0:
            return pd.Series(fmt, index=self.index)
        starts.append(len(fmt))
        ends.append(len(fmt))
        result = pd.Series(fmt[:starts[0]], index=df.index)
        for (col_name, col_fmt), s, e in zip(groups, ends[:-1], starts[1:]):
            if col_fmt is None:
                col = df[col_name].astype(str)
            else:
                col = df[col_name].apply(lambda x: col_fmt.format(x))
            result += col + fmt[s:e]
        result.index = self.index
        result.name = name
        return result

    #######################################################################################
    # 5. ELEMENT-WISE PROPERTIES
    #######################################################################################

    def is_empty(self) -> Series:
        return self.end == self.start

    def is_valid(self):
        return self._end >= self._start

    @property
    def length(self) -> Series:
        return self._end - self._start

    @property
    def center(self) -> Series:
        return (self._start + self._end) // 2

    #######################################################################################
    # 6. ELEMENT-WISE OPERATIONS WITH SCALAR
    #######################################################################################

    def clip(
            self,
            left: Union[Series, ndarray, int, None] = None,
            right: Union[Series, ndarray, int, None] = None
    ):
        new_start = self._start
        new_end = self._end
        if left is not None:
            if right is not None:
                is_wrong = left > right
                if hasattr(is_wrong, '__len__'):
                    is_wrong = is_wrong.any()
                if is_wrong:
                    raise ValueError(f'left must be <= right')
            new_start = np.maximum(new_start, left)
            new_end = np.maximum(new_end, left)
        if right is not None:
            new_start = np.minimum(new_start, right)
            new_end = np.minimum(new_end, right)
        return RangeSeries(new_start, new_end, set_names=self.names)
    
    def expand(self, left: Union[Series, ndarray, int], right: Union[Series, ndarray, int, None] = None, *, towards: Union[str, int, ndarray, Series] = 'center') -> RangeSeries:
        if right is None:
            right = left
        if isinstance(towards, str):
            if towards == 'center':
                to_pos = self.center
            elif towards == 'start':
                to_pos = self._start
            elif towards == 'end':
                to_pos = self._end
            else:
                raise ValueError(f'Invalid value for "to": {towards}')
        else:  # wither number or vector
            to_pos = towards
        new_start = np.minimum(self._start - left, to_pos)
        new_end = np.maximum(self._end + right, to_pos)
        return RangeSeries(new_start, new_end, set_names=self.names)

    def shift(self, d: Union[Series, int]) -> RangeSeries:
        return RangeSeries(self._start + d, self._end + d, set_names=self.names)

    def split_at(self, split_point: Union[Series, int]) -> Tuple[RangeSeries, RangeSeries]:
        left_start = np.minimum(self._start, split_point)
        left_end = np.minimum(split_point, self._end)
        right_start = np.maximum(self._start, split_point)
        right_end = np.maximum(self._end, split_point)
        return (
            RangeSeries(left_start, left_end, set_names=self.names),
            RangeSeries(right_start, right_end, set_names=self.names)
        )

    def split_at_length(self, length: Union[Series, ndarray, int]) -> Tuple[RangeSeries, RangeSeries]:
        coord = np.where(length >= 0, self._start, self._end) + length
        return self.split_at(coord)

    def split_at_ratio(self, ratio: Union[Series, ndarray, float]) -> Tuple[RangeSeries, RangeSeries]:
        coord = np.where(ratio >= 0, self._start, self._end) + (self.length * ratio).round().astype(self.dtype)
        return self.split_at(coord)

    def bin_indices(self, size: int, start: int = 0) -> RangeSeries:
        first_bin = bin_coords(self.start, size, start)
        last_bin = bin_coords(np.maximum(self.start, self.end - 1), size, start) + 1
        return RangeSeries(first_bin, last_bin, set_names=self.names)

    def bin(self, size: int, start: int = 0) -> RangeSeries:
        bin_ids = self.bin_indices(size, start)
        bin_start = start + bin_ids.start * size
        bin_end = start + bin_ids.end * size
        return RangeSeries(bin_start, bin_end, set_names=bin_ids.names)

    #######################################################################################
    # 7. HORIZONTAL OPERATIONS (ELEMENT-WISE WITH OTHER RangeSeries INSTANCE)
    #######################################################################################

    def __eq__(self, other: RangeSeries) -> Series:
        return (self._start == other.start) & (self._end == other.end)

    def __ne__(self, other: RangeSeries) -> Series:
        return ~(self == other)

    def __and__(self, other: RangeSeries) -> RangeSeries:
        new_start = np.maximum(self._start, other.start)
        new_end = np.minimum(self._end, other.end)
        return RangeSeries(new_start, np.maximum(new_end, new_start), set_names=self.names)

    def __or__(self, other: RangeSeries) -> Tuple[RangeSeries, RangeSeries]:
        left_start = np.minimum(self._start, other.start)
        right_end = np.maximum(self._end, other.end)
        touching = self.distance(other) <= 0
        left_end = np.where(touching, right_end, np.minimum(self._end, other.end))
        right_start = np.where(touching, right_end, np.maximum(self._start, other.start))
        left = RangeSeries(left_start, left_end, set_names=self.names)
        right = RangeSeries(right_start, right_end, set_names=self.names)
        return left, right

    def sup(self, other: RangeSeries) -> RangeSeries:
        new_start = np.minimum(self._start, other.start)
        new_end = np.maximum(self._end, other.end)
        return RangeSeries(new_start, new_end, set_names=self.names)

    def gap_between(self, other: RangeSeries) -> RangeSeries:
        new_start = np.minimum(self._end, other.end)
        new_end = np.maximum(self._start, other.start)
        return RangeSeries(new_start, np.maximum(new_start, new_end), set_names=self.names)

    @staticmethod
    def _ranges_contains_impl(start_A, end_A, empty_A, start_B, end_B, empty_B) -> Union[Series, ndarray]:
        return (start_A <= start_B) & ((start_B < end_A) | empty_A) & \
            ((start_A < end_B) | empty_B) & (end_B <= end_A)

    def contains(self, other: Union[RangeSeries, Series, ndarray, Tuple[int, int], int]) -> Series:
        if isinstance(other, RangeSeries):
            return RangeSeries._ranges_contains_impl(
                self._start, self._end, self.is_empty(),
                other._start, other._end, other.is_empty()
            )
        elif isinstance(other, tuple):
            return RangeSeries._ranges_contains_impl(
                self._start, self._end, self.is_empty(),
                other[0], other[1], other[1] <= other[0]
            )
        else:  # Series, array, int, float
            if not isinstance(other, Series):
                other = Series(other, index=self.index)
            return (self._start <= other) & ((other < self._end) | self.is_empty())

    def is_superset_of(self, other: Union[RangeSeries, Series, ndarray, int]) -> Series:
        result = self.contains(other)
        if isinstance(other, RangeSeries):
            result |= other.is_empty()
        return result

    def is_contained_in(self, other: Union[RangeSeries, Tuple[int, int]]) -> Series:
        if isinstance(other, RangeSeries):
            return other.contains(self)
        else:  # tuple of coords:
            return RangeSeries._ranges_contains_impl(
                other[0], other[1], other[1] <= other[0],
                self._start, self._end, self.is_empty()
            )

    def is_subset_of(self, other: Union[RangeSeries, Tuple[int, int]]) -> Series:
        if isinstance(other, RangeSeries):
            result = other.contains(self)
        else:  # tuple of coords:
            result = RangeSeries._ranges_contains_impl(
                self._start, self._end, self.is_empty(),
                other[0], other[1], other[1] <= other[0]
            )
        return result | self.is_empty()

    def intersects_elementwise(self, other: Union[RangeSeries, Tuple[int, int]]) -> Series:
        return self.overlap(other) > 0

    def is_separated_from(self, other: Union[RangeSeries, Tuple[int, int]]) -> Series:
        return self.offset(other) > 0

    def distance(self, other: Union[RangeSeries, Tuple[int, int], Series, ndarray, int]) -> Series:
        return np.maximum(self.offset(other), 0)

    def offset(self, other: Union[RangeSeries, Tuple[int, int], Series, ndarray, int]) -> Series:
        if isinstance(other, RangeSeries):
            other_start, other_end = other._start, other._end
        elif isinstance(other, tuple):
            other_start, other_end = other
        else:
            other_start = other_end = other
        d_left = other_start - self._end
        d_right = self._start - other_end
        d = np.maximum(d_left, d_right)
        return d

    def overlap(self, other: Union[RangeSeries, Tuple[int, int]]) -> Series:
        if isinstance(other, RangeSeries):
            other_length = other.length
        else:  # tuple
            other_length = np.maximum(other[1] - other[0], 0)
        shorter_len = np.minimum(self.length, other_length)
        cropped_offset = np.minimum(self.offset(other), 0)
        return np.minimum(shorter_len, -cropped_offset)

    #######################################################################################
    # 8. AGGREGATING OR VERTICAL OPERATIONS
    #######################################################################################

    def count(self) -> int:
        return len(self._start)

    def min(self):
        return self._start.min()

    def max(self, by: Union[Series, List[Series], str, List[str], None] = None):
        return self._end.max()

    def range_tuple(self) -> Tuple[int, int]:
        return self._start.min(), self._end.max()

    def range(self) -> RangeSeries:
        if len(self) == 0:
            return RangeSeries(0, 0)
        else:
            return RangeSeries(*self.range_tuple())

    def sort(self, by=None, ascending=True, by_index=True) -> RangeSeries:
        return self(self.to_sorted_dataframe(by, ascending, by_index))

    def to_sorted_dataframe(self, by=None, ascending=True, by_index=True):
        df = self.to_frame()
        coord_names = list(df.columns[-2:])
        idx_names = list(df.columns[:-2])
        if by is None:
            by = idx_names + coord_names if by_index else coord_names
        df = df.sort_values(by=by, ascending=ascending)
        if len(idx_names) > 0:
            df = df.set_index(idx_names)
        return df

    def is_disjoint(self) -> bool:
        if len(self) == 0:
            return True
        cluster_ids = overlapping_clusters_grouped(self)
        return cluster_ids.max() == len(cluster_ids) - 1

    def clusters(self) -> Series:
        cluster_ids = overlapping_clusters_grouped(self)
        return Series(cluster_ids, index=self.index)

    def union_self(self) -> RangeSeries:
        if len(self) == 0:
            return self
        cluster_ids = overlapping_clusters_grouped(self)
        if cluster_ids.max() == len(cluster_ids) - 1:
            return self
        start = self.start.groupby(cluster_ids).min()
        end = self.end.groupby(cluster_ids).max()
        return RangeSeries(start, end)

    def intersect_self(self) -> RangeSeries:
        if len(self) == 0:
            return RangeSeries([])
        max_start = self.start.max()
        min_end = self.end.min()
        if min_end < max_start:
            return RangeSeries([])
        else:
            return RangeSeries(max_start, min_end)

    #######################################################################################
    # 9. SET-LIKE OPERATIONS (ALL RANGES IN THE SERIES TREATED AS A SET)
    #######################################################################################

    def union_other(self, other: RangeSeries) -> RangeSeries:
        return range_set_union(self, other, None, None)

    def intersect_other(self, other: RangeSeries) -> RangeSeries:
        return range_set_intersection(self, other, None, None)

    #######################################################################################
    # 10. MISCELLANEOUS
    #######################################################################################

    # TODO: test
    def groupby(self, groups: Union[Iterable[Union[Series, np.ndarray, str]], Series, str], observed: bool = False)\
            -> RangeSeriesGroupBy:
        return RangeSeriesGroupBy(self, groups, observed)

    # TODO: test
    def assign_to(self, df: DataFrame):
        if self._start.name not in df:
            raise ValueError(f'Cannot find start column "{self._start.name}" in dataframe')
        if self._end.name not in df:
            raise ValueError(f'Cannot find end column "{self._end.name}" in dataframe')
        df[self._start.name] = self._start
        df[self._end.name] = self._end
        return self

    # TODO: test
    @staticmethod
    def sort_with_df(df, by, *ranges, ascending=True, kind='quicksort', na_position='last', key=None):
        df = df.sort_values(by, ascending=ascending, kind=kind, na_position=na_position, key=key)
        ranges = [r(df) for r in ranges]
        return df, *ranges


#######################################################################################
# GROUPING
#######################################################################################


class RangeSeriesGroupBy(object):
    __slots__ = ['_ranges', '_groups', '_indices', '_level_names']

    # TODO: test
    def __init__(
            self,
            ranges: RangeSeries,
            groups: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None],
            observed: bool = False
    ):
        if groups is None:
            groups = tuple()
        if isinstance(groups, Series) or isinstance(groups, str):
            groups = [groups]
        self._ranges = ranges        
        self._groups = groups
        if len(groups) == 0:
            self._indices = None
        else:            
            self._indices = ranges.start.groupby(groups, observed=observed).indices        
        names = []
        for i, g in enumerate(groups):
            if isinstance(g, str):
                names.append(g)
            elif isinstance(g, Series):
                names.append(g.name)
            else:
                names.append(f'_level_{i}')
        self._level_names = tuple(names)

    #######################################################################################
    # BASIC PROPERTIES (1. in RangeSeries)
    #######################################################################################

    # TODO: test
    @property
    def indices(self) -> Optional[Dict[Hashable, np.ndarray]]:
        return self._indices

    # TODO: test
    @property
    def grouping_names(self) -> Tuple[Hashable | str]:
        return self._level_names

    # TODO: test
    @property
    def dtype(self) -> np.dtype:
        return self._ranges.dtype

    # TODO: test
    def __len__(self) -> int:
        return len(self._groups)

    # TODO: test
    @property
    def names(self) -> Tuple[Hashable, Hashable]:
        return self._ranges.names
    
    @property
    def has_groups(self) -> bool:
        return len(self._groups) > 0

    #######################################################################################
    # ITERATION AND SUBSETS (2. in RangeSeries)
    #######################################################################################

    # TODO: cache groups?
    def groups(self):
        for key, indices in self._indices.items():
            yield key, self._ranges.subset(indices)

    #######################################################################################
    # AGGREGATING OR VERTICAL OPERATIONS (8. in RangeSeries)
    #######################################################################################

    # TODO: test
    def count(self) -> Series:
        return Series(map(len, self._indices.values()))

    # TODO: test
    def _run_by_group(self, method, *args, **kwargs) -> Union[Dict[Series], Dict[RangeSeries]]:
        results = [method(group_range, *args, **kwargs) for _, group_range in self.groups()]
        return results
    
    # TODO: test
    def _run_by_group_and_concat(self, method, _concat_fun = pd.concat, *args, **kwargs) -> Union[Series, RangeSeries]:
        group_results = self._run_by_group(method, *args, **kwargs)
        return _concat_fun(group_results, names=self._level_names)

    # TODO: test
    def min(self) -> Series:
        return self._run_by_group_and_concat(RangeSeries.min)

    # TODO: test
    def max(self) -> Series:
        return self._run_by_group_and_concat(RangeSeries.max)

    # TODO: test
    def range(self) -> RangeSeries:
        return self._run_by_group_and_concat(RangeSeries.range)

    # TODO: test
    def is_disjoint(self) -> Series:
        cluster_ids = overlapping_clusters_grouped(self._ranges, groups=self._indices)
        return pd.Series({
            keys: cluster_ids[idx].nunique() == 1 for keys, idx in self._indices
        })

    # TODO: test
    def clusters(self) -> Series:
        cluster_ids = overlapping_clusters_grouped(self._ranges, groups=self._indices)
        return Series(cluster_ids, index=self._ranges.index)

    # TODO: test
    def union_self(self) -> RangeSeries:
        cluster_ids = overlapping_clusters_grouped(self._ranges, groups=self._indices)        
        print(cluster_ids)
        to_drop = [lvl for lvl in self._ranges.index.names if lvl not in self._level_names]
        new_index = self._ranges.index
        if len(to_drop) > 0:
            if set(to_drop) == set(new_index.names):                
                new_index = pd.RangeIndex(len(new_index))
            else:
                new_index = new_index.droplevel(to_drop)
        print(new_index)        
        cluster_starts = overlaps.get_idx_of_cluster_starts(cluster_ids)
        print(cluster_starts)
        idx = new_index[cluster_starts]
        start = self._ranges.start.groupby(cluster_ids).min()
        end = self._ranges.end.groupby(cluster_ids).max()
        return RangeSeries(start, end, set_index=idx)

    # TODO: test
    def intersect_self(self) -> RangeSeries:
        if len(self._ranges) == 0:
            return RangeSeries([])
        max_start = self.max()
        min_end = self.min()
        if min_end < max_start:
            return RangeSeries(max_start, max_start)
        else:
            return RangeSeries(max_start, min_end)


#######################################################################################
# FREE FUNCTIONS
#######################################################################################


def overlapping_clusters_grouped(
        ranges: RangeSeries,
        groups: Union[Series, List[Series], str, List[str], Dict[Hashable, np.ndarray], None] = None,
        is_ascending: bool = True
) -> np.ndarray:
    if groups is None:
        groups = []
    elif isinstance(groups, Series) or isinstance(groups, str):
        groups = [groups]
    if len(groups) > 0:
        cluster_idxs = np.empty(len(ranges), dtype=int)
        if isinstance(groups, dict):
            group_indexes = groups
        else:
            group_indexes = ranges.start.groupby(groups, observed=True).indices
        next_cluster_idx = 0
        for key, grp_idx in group_indexes.items():
            if len(grp_idx) == 0:
                continue
            start = ranges.start.iloc[grp_idx].to_numpy()
            end = ranges.end.iloc[grp_idx].to_numpy()            
            first_unsorted = overlaps.check_if_sorted(start, end, exc_type=None, ascending=is_ascending)
            if first_unsorted != -1:
                print(start[:10])  # TODO: temp
                print(end[:10])  # TODO: temp
                raise ValueError(f"Group {key} is not sorted at index {first_unsorted}.")
            grp_cluster_idxs = overlaps.get_overlapping_clusters(start, end, is_ascending=is_ascending)
            n_found = grp_cluster_idxs.max() + 1
            cluster_idxs[grp_idx] = grp_cluster_idxs + next_cluster_idx
            next_cluster_idx += n_found
    else:
        start = ranges.start.to_numpy()
        end = ranges.end.to_numpy()
        first_unsorted = overlaps.check_if_sorted(start, end, exc_type=None, ascending=is_ascending)        
        if first_unsorted != -1:
            raise ValueError(f"Data not sorted at index {first_unsorted}.")
        cluster_idxs = overlaps.get_overlapping_clusters(start, end, is_ascending=is_ascending)

    return cluster_idxs


def _get_group_indices(
        ranges: RangeSeries,
        groups: Union[None, Series, str, Iterable[Series]],
        observed: bool
    ) -> Tuple[Dict[Hashable, np.ndarray], int]:
    if groups is None:
        groups = []
    if isinstance(groups, Series) or isinstance(groups, str):
        groups = [groups]
    if len(groups) == 0:
        return None, 0
    if isinstance(groups[0], Series):
        to_group = pd.concat(groups, axis=1)
    else:  # str -> treat as index
        to_group = ranges.to_frame().reset_index().loc[:, groups]    
    return dict(to_group.groupby(groups, observed=observed).indices), len(to_group.columns)


def overlapping_pairs_grouped(
        ranges_a: RangeSeries,
        ranges_b: RangeSeries,
        groups_a: Union[Series, List[Series], str, List[str], None] = None,
        groups_b: Union[Series, List[Series], str, List[str], None] = None
) -> np.ndarray:
    _indices_a, _n_grouping_cols_a = _get_group_indices(ranges_a, groups_a, observed=True)
    _indices_b, _n_grouping_cols_b = _get_group_indices(ranges_b, groups_b, observed=True)
    if _n_grouping_cols_a != _n_grouping_cols_b:
        raise ValueError(
            f"Must provide the same number of group levels (current: {_n_grouping_cols_a} vs {_n_grouping_cols_b})."
        )
    if _n_grouping_cols_a > 0:  # and len(groups_b) > 0
        mergred_indexes = _match_dict_items(_indices_a, _indices_b)
        matched_a = []
        matched_b = []
        for group, idx_a, idx_b in mergred_indexes:
            start_a = ranges_a.start.iloc[idx_a].to_numpy()
            end_a = ranges_a.end.iloc[idx_a].to_numpy()
            start_b = ranges_b.start.iloc[idx_b].to_numpy()
            end_b = ranges_b.end.iloc[idx_b].to_numpy()
            overlaps.check_if_sorted(start_a, end_a, raise_msg='RangeSeries A')
            overlaps.check_if_sorted(start_b, end_b, raise_msg='RangeSeries B')
            group_pairs = overlaps.get_overlapping_pairs(start_a, end_a, start_b, end_b, sort_result=False)
            if group_pairs.size > 0:
                matched_a.append(idx_a[group_pairs[:, 0]])
                matched_b.append(idx_b[group_pairs[:, 1]])
        matched_a = np.concatenate(matched_a)
        matched_b = np.concatenate(matched_b)
    else:
        start_a = ranges_a.start.to_numpy()
        end_a = ranges_a.end.to_numpy()
        start_b = ranges_b.start.to_numpy()
        end_b = ranges_b.end.to_numpy()
        overlaps.check_if_sorted(start_a, end_a, raise_msg='RangeSeries A')
        overlaps.check_if_sorted(start_b, end_b, raise_msg='RangeSeries B')
        group_pairs = overlaps.get_overlapping_pairs(start_a, end_a, start_b, end_b, sort_result=False)
        matched_a = group_pairs[:, 0]
        matched_b = group_pairs[:, 1]
    sorting_idx = np.lexsort((matched_b, matched_a))  # Notice the order
    return np.column_stack([matched_a[sorting_idx], matched_b[sorting_idx]])


def join_by_indices(df1, df2, idx):
    res_df = pd.concat([
        df1.iloc[idx[:,0],:].reset_index(drop=True),
        df2.iloc[idx[:,1],:].reset_index(drop=True)
    ], axis=1)
    return res_df


# TODO: efficiency
def range_set_union(
        ranges_a: RangeSeries,
        ranges_b: RangeSeries,
        groups_a: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None,
        groups_b: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None
) -> RangeSeries:
    if groups_a is not None and groups_b is None:
        groups_b = groups_a
    if groups_a is None:
        groups_a = []
    if groups_b is None:
        groups_b = []
    prepared_a = ranges_a.reset_index(keep=groups_a, drop=True).sort(by=groups_a)
    prepared_b = ranges_b.reset_index(keep=groups_b, drop=True).sort(by=groups_b)
    merged = merge_sorted_ranges(prepared_a, prepared_b, groups_a, groups_b)
    new_names = [f'level_{i}' if name is None else name for i, name in enumerate(merged.index.names)]
    merged.index.names = new_names
    cluster_ids = overlapping_clusters_grouped(merged, groups_a)
    starts = merged.start.groupby(cluster_ids).min()
    ends = merged.end.groupby(cluster_ids).max()
    if len(groups_a) > 0:
        _, cluster_first_ocurrence = np.unique(cluster_ids, return_index=True)
        index = merged.index[cluster_first_ocurrence]
    else:
        index = pd.RangeIndex(len(starts))
    return RangeSeries(starts, ends, set_index=index)


# TODO: efficiency
def range_set_intersection(
    ranges_a: RangeSeries,
    ranges_b: RangeSeries,
    groups_a: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None,
    groups_b: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None
) -> RangeSeries:
    if groups_a is not None and groups_b is None:
        groups_b = groups_a
    disjoint_a = ranges_a.sort(by=groups_a)
    print(disjoint_a.to_frame())  # TODO: temp
    disjoint_a = disjoint_a.groupby(groups_a, observed=True).union_self()
    disjoint_b = ranges_b.sort(by=groups_b)
    disjoint_b = disjoint_b.groupby(groups_b, observed=True).union_self()
    pair_ids = overlapping_pairs_grouped(disjoint_a, disjoint_b, groups_a, groups_b)
    print(disjoint_a.to_frame())  # TODO: temp
    print(disjoint_b.to_frame())  # TODO: temp
    print(pair_ids)  # TODO: temp
    starts_a = disjoint_a.start.iloc[pair_ids[:, 0]]  #.to_numpy()
    starts_b = disjoint_b.start.iloc[pair_ids[:, 1]]  #.to_numpy()    
    ends_a = disjoint_a.end.iloc[pair_ids[:, 0]]  #.to_numpy()
    ends_b = disjoint_b.end.iloc[pair_ids[:, 1]]  #.to_numpy()
    # print(starts_a)  # TODO: temp
    # print(starts_b)  # TODO: temp
    # print(ends_a)  # TODO: temp
    # print(ends_b)  # TODO: temp    
    starts = np.maximum(starts_a, starts_b)
    ends = np.minimum(ends_a, ends_b)
    merged = RangeSeries(starts, ends)
    print(merged.to_frame())  # TODO: temp
    result = merged.groupby(groups_a, observed=True).union_self()
    print(result.to_frame())  # TODO: temp
    return result


# TODO: test
def concat(ranges: List[RangeSeries] | Dict[Hashable, RangeSeries], names: Optional[List[str]] = None) -> RangeSeries:
    if isinstance(ranges, dict):
        starts = {k: r.start for k, r in ranges.items()}
        ends = {k: r.end for k, r in ranges.items()}
    else:
        starts = [r.start for r in ranges]
        ends = [r.end for r in ranges]
    # if names is not None and len(names) == 0:  # Don't concat if there are no groups
    #     assert len(starts) == 1        
    #     new_start, *_ = starts.values()
    #     new_end, *_ = ends.values()
    # else:
    new_start = pd.concat(starts, names=names)
    new_end = pd.concat(ends, names=names)
    res = RangeSeries(new_start, new_end)
    print(res.to_frame())  # TODO: temp
    return res


def _extract_index(index: pd.Index, groups: List):
    group_names = [
        f'level_{i}' if name is None else name for i, name in
        enumerate(g.name if isinstance(g, Series) else g for g in groups)
    ]
    has_series = sum(isinstance(g, Series) for g in groups) > 0
    if has_series:
        raise NotImplementedError('Not implemented. Use names instead.')
    output_names = [name for name in index.names if name not in group_names] + group_names
    if has_series:
        df = index.to_frame()
        for g, name in zip(groups, group_names):
            if isinstance(g, Series):
                df[name] = g
        df = df.set_index(group_names)
        index = df.index
    if index.nlevels > 1 and index.names != output_names:
        index = index.reorder_levels(output_names)
    return index


def merge_sorted_ranges(
        ranges_a: RangeSeries,
        ranges_b: RangeSeries,
        groups_a: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None,
        groups_b: Union[Iterable[Union[Series, np.ndarray, str]], Series, str, None] = None
) -> RangeSeries:
    if groups_a is None:
        groups_a = []
    if groups_b is None:
        groups_b = []
    if len(groups_a) != len(groups_b):
        raise ValueError(
            f"Must provide the same number of group levels (current: {len(groups_a)} vs {len(groups_b)})."
        )
    n = len(ranges_a) + len(ranges_b)
    new_starts = np.empty(n, dtype=int)
    new_ends = np.empty(n, dtype=int)
    index_a = _extract_index(ranges_a.index, groups_a)
    index_b = _extract_index(ranges_b.index, groups_b)
    concatenated_index = index_a.append(index_b)
    concatenated_index.names = index_a.names
    if len(groups_a) > 0:  # and len(groups_b) > 0
        _indices_a, _n_grouping_cols_a = _get_group_indices(ranges_a, groups_a, observed=True)
        _indices_b, _n_grouping_cols_b = _get_group_indices(ranges_b, groups_b, observed=True)
        mergred_indexes = list(_match_dict_items(_indices_a, _indices_b))
        group_start = 0
        b_offset = len(ranges_a)
        index_perm = np.empty(n, dtype=int)
        for _, group_idx_a, group_idx_b in mergred_indexes:
            group_size = len(group_idx_a) + len(group_idx_b)
            start_a = ranges_a.start.iloc[group_idx_a].to_numpy()
            end_a = ranges_a.end.iloc[group_idx_a].to_numpy()
            start_b = ranges_b.start.iloc[group_idx_b].to_numpy()
            end_b = ranges_b.end.iloc[group_idx_b].to_numpy()
            overlaps.check_if_sorted(start_a, end_a, raise_msg='RangeSeries A')
            overlaps.check_if_sorted(start_b, end_b, raise_msg='RangeSeries B')
            idx_a, idx_b = overlaps.mergesort_ranges_indices(start_a, end_a, start_b, end_b)
            new_starts[idx_a + group_start] = start_a
            new_starts[idx_b + group_start] = start_b
            new_ends[idx_a + group_start] = end_a
            new_ends[idx_b + group_start] = end_b
            index_perm[idx_a + group_start] = group_idx_a
            index_perm[idx_b + group_start] = group_idx_b + b_offset
            group_start += group_size
    else:
        start_a = ranges_a.start.to_numpy()
        end_a = ranges_a.end.to_numpy()
        start_b = ranges_b.start.to_numpy()
        end_b = ranges_b.end.to_numpy()
        overlaps.check_if_sorted(start_a, end_a, raise_msg='RangeSeries A')
        overlaps.check_if_sorted(start_b, end_b, raise_msg='RangeSeries B')
        idx_a, idx_b = overlaps.mergesort_ranges_indices(start_a, end_a, start_b, end_b)
        new_starts[idx_a] = start_a
        new_starts[idx_b] = start_b
        new_ends[idx_a] = end_a
        new_ends[idx_b] = end_b
        index_perm = overlaps.target_indices_to_permutation(idx_a, idx_b)
    return RangeSeries(new_starts, new_ends, set_index=concatenated_index[index_perm])


def bin_coords(coords: Union[ndarray, Series], size: int, start: int = 0) \
        -> Union[ndarray, Series]:
    assert size > 0
    bin_no = (coords - start) // size
    return bin_no


#######################################################################################
# UTILS
#######################################################################################


def assert_ranges_equal(actual: RangeSeries, expected: RangeSeries, check_names=True):
    assert isinstance(actual, RangeSeries)
    assert isinstance(expected, RangeSeries)
    pd.testing.assert_series_equal(
        actual.start, expected.start, obj='Starts',
        check_index_type=False, check_names=check_names
    )
    pd.testing.assert_series_equal(
        actual.end, expected.end, obj='Ends',
        check_index_type=False, check_names=check_names
    )
    pd.testing.assert_index_equal(actual.index, expected.index, check_names=check_names)


_MATCH_DICT_ITEMS_HOW_SET = ('inner', 'outer', 'left', 'right')

def _match_dict_items(d1: Dict, d2: Dict, how='inner') -> Generator[Tuple[Any, Any, Any], None, None]:
    # Note: 'None' not supported as dict value because of non-inner join behavior
    if how not in _MATCH_DICT_ITEMS_HOW_SET:
        raise ValueError(f'"how" must be one of: {",".join(_MATCH_DICT_ITEMS_HOW_SET)}')
    all1 = how == 'left' or how == 'outer'
    all2 = how == 'right' or how == 'outer'
    for key, v1 in d1.items():
        v2 = d2.get(key)
        if v2 is None and not all1:
            continue
        yield key, v1, v2
    if not all2:
        return
    for key, v2 in d2.items():
        if key not in d1.keys():
            yield key, None, v2
