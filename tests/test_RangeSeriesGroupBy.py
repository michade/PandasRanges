import itertools
from functools import partial
from typing import Dict
from math import prod

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal, assert_index_equal
from pytest import param

from PandasRanges.ranges import *
from PandasRanges.testing_utils import TestCasesCsv, Vectorizer, make_pytest_id


def make_simple_df(start, end, names=('start', 'end')):
    return DataFrame({names[0]: start, names[1]: end}, dtype=int)


def add_groups(df, keep_index=True, **name_levels):
    index_levels = [list(lvls) for lvls in name_levels.values()]
    index_names = list(name_levels.keys())
    if keep_index:
        index_levels.append(df.index)
        index_names.append(df.index.name)
    n_groups = prod(len(lvls) for lvls in index_levels[:-1])
    index = pd.MultiIndex.from_product(index_levels)
    index.names = index_names
    df = pd.concat([df] * n_groups)
    df.index = index
    return df


def make_grouped_df(start, end, names=('start', 'end'), keep_index=True, **name_levels):
    df = make_simple_df(start, end, names=names)
    return add_groups(df, keep_index=keep_index, **name_levels)


def make_grouped_ranges(df, keep_index=True, **name_levels):
    df = add_groups(df, keep_index=keep_index, **name_levels)
    def _make_isntance():
        return RangeSeries(df)
    return _make_isntance


class RangesBasicTestCases:
    disjoint_1 = make_simple_df(
        [10, 50, 60],
        [30, 50, 80]
    )
    disjoint_2 = make_simple_df(
        [20, 50],
        [40, 70]
    )
    clustered_1 = make_simple_df(
        [10, 40, 50, 70],
        [30, 80, 60, 90]
    )
    clustered_2 = make_simple_df(
        [20, 30, 0],
        [40, 50, 90]
    )
    empty = make_simple_df([], [])
    onerow = make_simple_df([10], [20])


basic_dfs = RangesBasicTestCases()


def test__make_simple_df():
    expected = pd.DataFrame({'start': [1, 2], 'end': [3, 4]})
    result = make_simple_df([1, 2], [3, 4])
    assert_frame_equal(result, expected)


def test__add_groups__drop_index():
    expected = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], list('XY')],
        names=['l1', 'l2', 'l3']
    )
    df = make_simple_df([10, 20], [30, 40])
    result = add_groups(df, keep_index=False, l1='ABC', l2=[5, 7], l3=['X', 'Y'])
    assert_frame_equal(result, expected)


def test__add_groups__keep_index():
    expected = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], range(2)],
        names=['l1', 'l2', None]
    )
    df = make_simple_df([10, 20], [30, 40])
    result = add_groups(df, l1='ABC', l2=[5, 7])
    assert_frame_equal(result, expected)


def test__make_grouped_df__drop_index():
    expected = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], list('XY')],
        names=['l1', 'l2', 'l3']
    )
    result = make_grouped_df([10, 20], [30, 40], keep_index=False, l1='ABC', l2=[5, 7], l3=['X', 'Y'])
    assert_frame_equal(result, expected)


def test__make_grouped_df__keep_index():
    expected = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], range(2)],
        names=['l1', 'l2', None]
    )
    result = make_grouped_df([10, 20], [30, 40], l1='ABC', l2=[5, 7])
    assert_frame_equal(result, expected)


def test__make_grouped_ranges__drop_index():
    expected_df = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected_df.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], list('XY')],
        names=['l1', 'l2', 'l3']
    )
    df = make_simple_df([10, 20], [30, 40])
    expected = RangeSeries(expected_df)
    result = make_grouped_ranges(df, keep_index=False, l1='ABC', l2=[5, 7], l3=['X', 'Y'])()
    assert_ranges_equal(result, expected)


def test__make_grouped_ranges__keep_index():
    expected_df = DataFrame({
        'start': [10, 20] * 6,
        'end': [30, 40] * 6,
    })
    expected_df.index = pd.MultiIndex.from_product(
        [list('ABC'), [5, 7], range(2)],
        names=['l1', 'l2', None]
    )
    expected = RangeSeries(expected_df)
    df = make_simple_df([10, 20], [30, 40])
    result = make_grouped_ranges(df, l1='ABC', l2=[5, 7])()
    assert_ranges_equal(result, expected)


#######################################################################################
# GROUPING
#######################################################################################


@pytest.mark.parametrize(
    'ranges,groups', [
        # (make_grouped_ranges(basic_dfs.disjoint_1, l1='ABC', l2='XY'), []),  # TODO: what then?
        (make_grouped_ranges(basic_dfs.disjoint_1, l1='ABC', l2='XY'), ['l1']),
        (make_grouped_ranges(basic_dfs.disjoint_1, l1='ABC', l2='XY'), ['l2']),
        (make_grouped_ranges(basic_dfs.disjoint_1, l1='ABC', l2='XY'), ['l1', 'l2'])
    ],
    scope='session'
)
def test__RangeSeriesGroupBy_init(ranges, groups):
    ranges = ranges()
    result = RangeSeriesGroupBy(ranges, groups)
    assert result.grouping_names == tuple(groups)
    # TODO: more tests
