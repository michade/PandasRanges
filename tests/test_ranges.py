from functools import partial

import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal
from pandas import DataFrame, Series

from PandasRanges.ranges import *


_overlapping_clusters_grouped_test_cases = {
    'no_groups': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'D', 'E'],
            'start': [10, 20, 60, 70, 80],
            'end': [30, 40, 70, 90, 90]
        }).set_index(['g1']),
        [],
        [0, 0, 1, 2, 2]
    ),
    'single_str_index': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B'],
            'start': [10, 10, 0, 20, 20],
            'end': [30, 30, 90, 40, 40]
        }).set_index(['g1']),
        ['g1'],
        [0, 1, 2, 0, 1]
    ),
    'double_str_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'B', 'B', 'B'],
            'g2': ['X', 'Y', 'X', 'Z', 'Y', 'X', 'Y'],
            'start': [10, 10, 20, 10, 10, 20, 60],
            'end': [30, 30, 40, 90, 30, 40, 70]
        }).set_index(['g1', 'g2']),
        ['g1', 'g2'],
        [0, 1, 0, 2, 4, 3, 5]
    ),
    'only_level_1_index': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B'],
            'g2': ['X', 'X', 'X', 'Y', 'Y'],
            'start': [10, 10, 0, 20, 20],
            'end': [30, 30, 90, 40, 40]
        }).set_index(['g1', 'g2']),
        ['g1'],
        [0, 1, 2, 0, 1]
    ),
    'only_level_2_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y'],
            'g2': ['A', 'B', 'C', 'A', 'B'],
            'start': [10, 10, 0, 20, 20],
            'end': [30, 30, 90, 40, 40]
        }).set_index(['g1', 'g2']),
        ['g2'],
        [0, 1, 2, 0, 1]
    ),
    'single_int_index': (
        DataFrame({
            'g1': [1, 2, 3, 1, 2],
            'start': [10, 10, 0, 20, 20],
            'end': [30, 30, 90, 40, 40]
        }).set_index(['g1']),
        ['g1'],
        [0, 1, 2, 0, 1]
    ),
    'empty_frame': (
        DataFrame({
            'start': [],
            'end': []
        }, dtype=int),
        [],
        []
    )
}


@pytest.fixture(
    params=_overlapping_clusters_grouped_test_cases.values(),
    ids=list(_overlapping_clusters_grouped_test_cases.keys()),
    scope='module'
)
def overlapping_clusters_grouped_cases(request):
    return request.param


def test__overlapping_clusters_grouped__indexes(overlapping_clusters_grouped_cases):
    df, groups, expected_indices = overlapping_clusters_grouped_cases
    iv = RangeSeries(df)
    expected_indices = np.array(expected_indices)
    result_indices = overlapping_clusters_grouped(iv, groups)
    assert_array_equal(result_indices, expected_indices)


def test__overlapping_clusters_grouped__series(overlapping_clusters_grouped_cases):
    df, groups, expected_indices = overlapping_clusters_grouped_cases
    df = df.reset_index()
    iv = RangeSeries(df)
    expected_indices = np.array(expected_indices)
    group_series = [df[grp] for grp in groups]
    result_indices = overlapping_clusters_grouped(iv, group_series)
    assert_array_equal(result_indices, expected_indices)


_overlapping_pairs_grouped_test_cases = {
    'no_groups': (
        DataFrame({
            'g1': ['X', 'X'],
            'start': [10, 80],
            'end': [30, 90]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 60, 70],
            'end': [30, 40, 70, 90]
        }).set_index(['g1']),
        [], [],
        [(0, 0), (0, 1), (1, 3)]
    ),
    'single_str_index': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B', 'D'],
            'start': [10, 10, 10, 50, 50, 80],
            'end': [30, 30, 30, 70, 70, 90]
        }).set_index('g1'),
        DataFrame({
            'g1': ['D', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index('g1'),
        ['g1'], ['g1'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7), (5, 0)]
    ),
    'double_str_index': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y'],
            'start': [10, 10, 10, 50, 50, 80],
            'end': [30, 30, 30, 70, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'g2': ['Y', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index(['g1', 'g2']),
        ['g1', 'g2'], ['g1', 'g2'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7), (5, 0)]
    ),
    'only_level_1_index': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B'],
            'g2': ['X', 'X', 'X', 'X', 'X'],
            'start': [10, 10, 10, 50, 50],
            'end': [30, 30, 30, 70, 70]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['D', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'g2': ['X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index(['g1', 'g2']),
        ['g1'], ['g1'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7)]
    ),
    'only_level_2_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'X', 'X'],
            'g2': ['A', 'B', 'C', 'A', 'B'],
            'start': [10, 10, 10, 50, 50],
            'end': [30, 30, 30, 70, 70]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['X', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'g2': ['D', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index(['g1', 'g2']),
        ['g2'], ['g2'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7)]
    ),
    'single_int_index': (
        DataFrame({
            'g1': [1, 2, 3, 1, 2, 4],
                'start': [10, 10, 10, 50, 50, 80],
            'end': [30, 30, 30, 70, 70, 90]
        }).set_index('g1'),
        DataFrame({
            'g1': [4, 1, 2, 3, 1, 2, 1, 2],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index('g1'),
        ['g1'], ['g1'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7), (5, 0)]
    ),
    'different_index_names': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y'],
            'start': [10, 10, 10, 50, 50, 80],
            'end': [30, 30, 30, 70, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p1': ['A', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'p2': ['Y', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index(['p1', 'p2']),
        ['g1', 'g2'], ['p1', 'p2'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7), (5, 0)]
    ),
    'different_index_order': (
        DataFrame({
            'g1': ['A', 'B', 'C', 'A', 'B', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y'],
            'start': [10, 10, 10, 50, 50, 80],
            'end': [30, 30, 30, 70, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p2': ['Y', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
            'p1': ['A', 'A', 'B', 'C', 'A', 'B', 'A', 'B'],
            'start': [10, 20, 20, 80, 40, 40, 50, 50],
            'end': [90, 40, 40, 90, 60, 60, 80, 80]
        }).set_index(['p2', 'p1']),
        ['g1', 'g2'], ['p1', 'p2'],
        [(0, 1), (1, 2), (3, 4), (3, 6), (4, 5), (4, 7), (5, 0)]
    ),
    'left_frame_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({
            'start': [10, 10],
            'end': [30, 30]
        }),
        [], [],
        []
    ),
    'right_frame_empty': (
        DataFrame({
            'start': [10, 10],
            'end': [30, 30]
        }),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        []
    ),
    'both_frames_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        []
    ),
}


@pytest.fixture(
    params=_overlapping_pairs_grouped_test_cases.values(),
    ids=list(_overlapping_pairs_grouped_test_cases.keys()),
    scope='module'
)
def overlapping_pairs_grouped_cases(request):
    return request.param


def test__overlapping_pairs_grouped__indexes(overlapping_pairs_grouped_cases):
    df1, df2, groups1, groups2, expected_indices = overlapping_pairs_grouped_cases
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    expected_indices = np.array(expected_indices, dtype=int).reshape((len(expected_indices), 2))
    result_indices = overlapping_pairs_grouped(iv1, iv2, groups1, groups2)
    assert_array_equal(result_indices, expected_indices)


def test__overlapping_pairs_grouped__series(overlapping_pairs_grouped_cases):
    df1, df2, groups1, groups2, expected_indices = overlapping_pairs_grouped_cases
    df1 = df1.reset_index()
    df2 = df2.reset_index()
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    expected_indices = np.array(expected_indices, dtype=int).reshape((len(expected_indices), 2))
    group_series_1 = [df1[grp] for grp in groups1]
    group_series_2 = [df2[grp] for grp in groups2]
    result_indices = overlapping_pairs_grouped(iv1, iv2, group_series_1, group_series_2)
    assert_array_equal(result_indices, expected_indices)


@pytest.mark.parametrize(
    'coords,size,start,expected_bins',
    [
        ([], 10, 0, []),
        ([0, 1, 9, 10, 11, 55], 10, 0, [0, 0, 0, 1, 1, 5]),
        ([0, 1, 9, 10, 11, 55], 10, 1, [-1, 0, 0, 0, 1, 5]),
        ([0, 1, 9, 10, 11, 55], 5, 0, [0, 0, 1, 2, 2, 11])
    ],
    scope='session', ids=['empty', 'step10start0', 'step10start1', 'step5start0']
)
def test__bin_coords(coords, size, start, expected_bins):
    expected_array = np.array(expected_bins, dtype=int)
    result_array = bin_coords(np.array(coords, dtype=int), size=size, start=start)
    assert_array_equal(result_array, expected_array, strict=True)
    expected_series = pd.Series(expected_bins)
    result_series = bin_coords(pd.Series(coords), size=size, start=start)
    assert_series_equal(result_series, expected_series)


@pytest.mark.parametrize(
    'left,right',
    [
        (
                partial(RangeSeries.from_records, [('a', 10, 20), ('b', 30, 40)]),
                partial(RangeSeries.from_records, [('a', 10, 20), ('b', 30, 40)])
        ),
        (
                partial(RangeSeries.from_records, []),
                partial(RangeSeries.from_records, [])
        )
    ],
    scope='session', ids=['nonempty', 'empty']
)
def test__assert_ranges_equal_valid(left, right):
    assert_ranges_equal(left(), right())


@pytest.mark.parametrize(
    'left,right',
    [
        (
                partial(RangeSeries.from_records, [(10, 20), (30, 40)]),
                partial(RangeSeries.from_records, [(10, 20), (30, 50)])
        ),
        (
                partial(RangeSeries.from_records, [(10, 20), (30, 40)]),
                partial(RangeSeries.from_records, [(30, 40), (10, 20)])
        ),
        (
                partial(RangeSeries.from_records, [('a', 10, 20), ('b', 30, 40)]),
                partial(RangeSeries.from_records, [('a', 10, 20), ('X', 30, 40)])
        )
    ],
    scope='session', ids=['different_value', 'different_order', 'different_index']
)
def test__assert_ranges_equal_invalid(left, right):
    with pytest.raises(AssertionError):
        assert_ranges_equal(left(), right())


_merge_sorted_ranges_test_cases = {
    'no_groups': (
        DataFrame({
            'g1': ['X', 'X', 'X'],
            'start': [10, 50, 80],
            'end': [30, 70, 90]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y'],
            'start': [20, 70],
            'end': [40, 90]
        }).set_index(['g1']),
        [], [],
        DataFrame({
            'g1': ['X', 'Y', 'X', 'Y', 'X'],
            'start': [10, 20, 50, 70, 80],
            'end': [30, 40, 70, 90, 90]
        }).set_index(['g1'])
    ),
    'single_str_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': ['X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 50, 70, 80, 15, 25, 55, 75, 85],
            'end': [30, 40, 70, 90, 90, 35, 45, 75, 95, 95]
        }).set_index(['g1'])
    ),
    'double_str_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g1', 'g2'], ['g1', 'g2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 50, 70, 80, 15, 25, 55, 75, 85],
            'end': [30, 40, 70, 90, 90, 35, 45, 75, 95, 95]
        }).set_index(['g1', 'g2'])
    ),
    'only_level_1_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'Y', 'Y'],
            'start': [10, 50, 50, 80],
            'end': [30, 70, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A'],
            'g2': ['Y', 'X'],
            'start': [20, 70],
            'end': [40, 90]
        }).set_index(['g1', 'g2']),
        ['g1'], ['g1'],
        DataFrame({
            'g2': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'start': [10, 20, 50, 50, 70, 80],
            'end': [30, 40, 70, 70, 90, 90]
        }).set_index(['g2', 'g1'])
    ),
    'only_level_2_index': (
        DataFrame({
            'g1': ['X', 'X', 'Y', 'Y'],
            'g2': ['A', 'A', 'A', 'A'],
            'start': [10, 50, 50, 80],
            'end': [30, 70, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['Y', 'X'],
            'g2': ['A', 'A'],
            'start': [20, 70],
            'end': [40, 90]
        }).set_index(['g1', 'g2']),
        ['g2'], ['g2'],
        DataFrame({
            'g1': ['X', 'Y', 'X', 'Y', 'X', 'Y'],
            'g2': ['A', 'A', 'A', 'A', 'A', 'A'],
            'start': [10, 20, 50, 50, 70, 80],
            'end': [30, 40, 70, 70, 90, 90]
        }).set_index(['g1', 'g2'])
    ),
    'single_int_index': (
        DataFrame({
            'g1': [1, 1, 1, 2, 2, 2],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': [2, 2, 1, 1],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            'start': [10, 20, 50, 70, 80, 15, 25, 55, 75, 85],
            'end': [30, 40, 70, 90, 90, 35, 45, 75, 95, 95]
        }).set_index(['g1'])
    ),
    'different_index_names': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p1': ['A', 'A', 'A', 'A'],
            'p2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p1', 'p2']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 50, 70, 80, 15, 25, 55, 75, 85],
            'end': [30, 40, 70, 90, 90, 35, 45, 75, 95, 95]
        }).set_index(['g1', 'g2'])
    ),
    'different_index_order': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p2': ['Y', 'Y', 'X', 'X'],
            'p1': ['A', 'A', 'A', 'A'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p2', 'p1']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'X', 'X', 'Y', 'Y', 'Y', 'Y', 'Y'],
            'start': [10, 20, 50, 70, 80, 15, 25, 55, 75, 85],
            'end': [30, 40, 70, 90, 90, 35, 45, 75, 95, 95]
        }).set_index(['g1', 'g2'])
    ),
    'left_frame_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        [], [],
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
    ),
    'right_frame_empty': (
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
    ),
    'both_frames_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({'start': [], 'end': []}, dtype=int)
    ),
}


@pytest.fixture(
    params=_merge_sorted_ranges_test_cases.values(),
    ids=list(_merge_sorted_ranges_test_cases.keys()),
    scope='module'
)
def merge_sorted_ranges_cases(request):
    return request.param


def test__merge_sorted_ranges__indexes(merge_sorted_ranges_cases):
    df1, df2, groups1, groups2, expected_df = merge_sorted_ranges_cases
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    expected = RangeSeries(expected_df)
    result = merge_sorted_ranges(iv1, iv2, groups1, groups2)
    assert_ranges_equal(result, expected)


# def test__merge_sorted_ranges__series(merge_sorted_ranges_cases):
#     df1, df2, groups1, groups2, expected_df = merge_sorted_ranges_cases
#     df1 = df1.reset_index()
#     df2 = df2.reset_index()
#     iv1 = RangeSeries(df1)
#     iv2 = RangeSeries(df2)
#     expected = RangeSeries(expected_df)
#     group_series_1 = [df1[grp] for grp in groups1]
#     group_series_2 = [df2[grp] for grp in groups2]
#     result = merge_sorted_ranges(iv1, iv2, group_series_1, group_series_2)
#     assert_ranges_equal(result, expected)


_range_set_union_test_cases = {
    'no_groups': (
        DataFrame({
            'g1': ['X', 'X', 'X'],
            'start': [10, 50, 80],
            'end': [30, 70, 90]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y'],
            'start': [20, 70],
            'end': [40, 90]
        }).set_index(['g1']),
        [], [],
        DataFrame({
            'start': [10, 50, 70],
            'end': [40, 70, 90]
        })
    ),
    'single_str_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 70, 15, 55, 75],
            'end': [40, 70, 90, 45, 75, 95]
        }).set_index(['g1'])
    ),
    'double_str_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g1', 'g2'], ['g1', 'g2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 70, 15, 55, 75],
            'end': [40, 70, 90, 45, 75, 95]
        }).set_index(['g1', 'g2'])
    ),
    'only_level_1_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 10, 50, 80],
            'end': [30, 70, 90, 30, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['Y', 'Y', 'X', 'X'],
            'start': [20, 70, 20, 70],
            'end': [40, 90, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': ['A', 'A', 'A'],
            'start': [10, 50, 70],
            'end': [40, 70, 90]
        }).set_index(['g1'])
    ),
    'only_level_2_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'g2': ['A', 'A', 'A', 'A', 'A', 'A'],
            'start': [10, 50, 80, 10, 50, 80],
            'end': [30, 70, 90, 30, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['Y', 'Y', 'X', 'X'],
            'g2': ['A', 'A', 'A', 'A'],
            'start': [20, 70, 20, 70],
            'end': [40, 90, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g2'], ['g2'],
        DataFrame({
            'g2': ['A', 'A', 'A'],
            'start': [10, 50, 70],
            'end': [40, 70, 90]
        }).set_index(['g2'])
    ),
    'single_int_index': (
        DataFrame({
            'g1': [1, 1, 1, 2, 2, 2],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': [2, 2, 1, 1],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': [1, 1, 1, 2, 2, 2],
            'start': [10, 50, 70, 15, 55, 75],
            'end': [40, 70, 90, 45, 75, 95]
        }).set_index(['g1'])
    ),
    'different_index_names': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p1': ['A', 'A', 'A', 'A'],
            'p2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p1', 'p2']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 70, 15, 55, 75],
            'end': [40, 70, 90, 45, 75, 95]
        }).set_index(['g1', 'g2'])
    ),
    'different_index_order': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p2': ['Y', 'Y', 'X', 'X'],
            'p1': ['A', 'A', 'A', 'A'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p2', 'p1']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 70, 15, 55, 75],
            'end': [40, 70, 90, 45, 75, 95]
        }).set_index(['g1', 'g2'])
    ),
    'left_frame_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        [], [],
        DataFrame({
            'start': [10, 50],
            'end': [40, 60]
        })
    ),
    'right_frame_empty': (
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({
            'start': [10, 50],
            'end': [40, 60]
        })
    ),
    'both_frames_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({'start': [], 'end': []}, dtype=int)
    ),
}


@pytest.fixture(
    params=_range_set_union_test_cases.values(),
    ids=list(_range_set_union_test_cases.keys()),
    scope='module'
)
def range_set_union_cases(request):
    return request.param


def test__range_set_union__indexes(range_set_union_cases):
    df1, df2, groups1, groups2, expected_df = range_set_union_cases
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    expected = RangeSeries(expected_df)
    result = range_set_union(iv1, iv2, groups1, groups2)
    assert_ranges_equal(result, expected)


# def test__range_set_union__series(range_set_union_cases):
#     df1, df2, groups1, groups2, expected_df = range_set_union_cases
#     iv1 = RangeSeries(df1)
#     iv2 = RangeSeries(df2)
#     expected = RangeSeries(expected_df)
#     group_series_1 = [df1[grp] for grp in groups1]
#     group_series_2 = [df2[grp] for grp in groups2]
#     result = range_set_union(iv1, iv2, group_series_1, group_series_2)
#     assert_ranges_equal(result, expected)


_range_set_intersection_test_cases = {
    'no_groups': (
        DataFrame({
            'g1': ['X', 'X', 'X'],
            'start': [10, 50, 80],
            'end': [30, 70, 90]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y'],
            'start': [20, 70],
            'end': [40, 90]
        }).set_index(['g1']),
        [], [],
        DataFrame({
            'start': [20, 80],
            'end': [30, 90]
        })
    ),
    'single_str_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': ['X', 'X', 'Y', 'Y'],
            'start': [20, 80, 25, 85],
            'end': [30, 90, 35, 95]
        })
    ),
    'double_str_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1', 'g2'], ['g1', 'g2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'Y', 'Y'],
            'start': [20, 80, 25, 85],
            'end': [30, 90, 35, 95]
        })
    ),
    'only_level_1_index': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 10, 50, 80],
            'end': [30, 70, 90, 30, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['Y', 'Y', 'X', 'X'],
            'start': [20, 70, 20, 70],
            'end': [40, 90, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': ['A', 'A'],
            'start': [20, 80],
            'end': [30, 90]
        })
    ),
    'only_level_2_index': (
        DataFrame({
            'g1': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'g2': ['A', 'A', 'A', 'A', 'A', 'A'],
            'start': [10, 50, 80, 10, 50, 80],
            'end': [30, 70, 90, 30, 70, 90]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'g1': ['Y', 'Y', 'X', 'X'],
            'g2': ['A', 'A', 'A', 'A'],
            'start': [20, 70, 20, 70],
            'end': [40, 90, 40, 90]
        }).set_index(['g1', 'g2']),
        ['g2'], ['g2'],
        DataFrame({
            'g1': ['A', 'A'],
            'start': [20, 80],
            'end': [30, 90]
        })
    ),
    'single_int_index': (
        DataFrame({
            'g1': [1, 1, 1, 2, 2, 2],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1']),
        DataFrame({
            'g1': [2, 2, 1, 1],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['g1']),
        ['g1'], ['g1'],
        DataFrame({
            'g1': [1, 1, 2, 2],
            'start': [20, 80, 25, 85],
            'end': [30, 90, 35, 95]
        })
    ),
    'different_index_names': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p1': ['A', 'A', 'A', 'A'],
            'p2': ['Y', 'Y', 'X', 'X'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p1', 'p2']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'Y', 'Y'],
            'start': [20, 80, 25, 85],
            'end': [30, 90, 35, 95]
        })
    ),
    'different_index_order': (
        DataFrame({
            'g1': ['A', 'A', 'A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'X', 'Y', 'Y', 'Y'],
            'start': [10, 50, 80, 15, 55, 85],
            'end': [30, 70, 90, 35, 75, 95]
        }).set_index(['g1', 'g2']),
        DataFrame({
            'p2': ['Y', 'Y', 'X', 'X'],
            'p1': ['A', 'A', 'A', 'A'],
            'start': [25, 75, 20, 70],
            'end': [45, 95, 40, 90]
        }).set_index(['p2', 'p1']),
        ['g1', 'g2'], ['p1', 'p2'],
        DataFrame({
            'g1': ['A', 'A', 'A', 'A'],
            'g2': ['X', 'X', 'Y', 'Y'],
            'start': [20, 80, 25, 85],
            'end': [30, 90, 35, 95]
        })
    ),
    'left_frame_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        [], [],
        DataFrame({'start': [], 'end': []}, dtype=int)
    ),
    'right_frame_empty': (
        DataFrame({
            'start': [10, 20, 50],
            'end': [30, 40, 60]
        }),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({'start': [], 'end': []}, dtype=int)
    ),
    'both_frames_empty': (
        DataFrame({'start': [], 'end': []}, dtype=int),
        DataFrame({'start': [], 'end': []}, dtype=int),
        [], [],
        DataFrame({'start': [], 'end': []}, dtype=int)
    ),
}


@pytest.fixture(
    params=_range_set_intersection_test_cases.values(),
    ids=list(_range_set_intersection_test_cases.keys()),
    scope='module'
)
def range_set_intersection_cases(request):
    return request.param


def test__range_set_intersection__indexes(range_set_intersection_cases):
    df1, df2, groups1, groups2, expected_df = range_set_intersection_cases
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    expected = RangeSeries(expected_df)
    result = range_set_intersection(iv1, iv2, groups1, groups2)
    assert_ranges_equal(result, expected)


def test__range_set_intersection__series(range_set_intersection_cases):
    df1, df2, groups1, groups2, expected_df = range_set_intersection_cases
    iv1 = RangeSeries(df1)
    iv2 = RangeSeries(df2)
    group_series_1 = [df1[grp] for grp in groups1]
    group_series_2 = [df2[grp] for grp in groups2]
    expected = RangeSeries(expected_df)
    result = range_set_intersection(iv1, iv2, group_series_1, group_series_2)
    assert_ranges_equal(result, expected)
