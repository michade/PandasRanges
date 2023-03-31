from contextlib import nullcontext

import pytest
from numpy.testing import assert_array_equal

from PandasRanges.overlaps import *

# (start_a, end_a, start_b, end_b), matching_index_pairs
_overlapping_pairs_test_cases = {
    'first-empty': (
        ([], [], [10, 60], [40, 70]), []
    ),
    'second-empty': (
        ([10, 60], [40, 70], [], []), []
    ),
    'both-empty': (
        ([], [], [], []), []
    ),
    'no-overlaps': (
        (
            [10, 60],
            [20, 70],
            [25, 45],
            [35, 55]
        ),
        []
    ),
    'simple-one-to-one': (
        (
            [10, 40, 90],
            [15, 60, 95],
            [20, 70],
            [50, 80]
        ),
        [(1, 0)]
    ),
    'simple-two-to-one': (
        (
            [10, 40, 90],
            [30, 60, 95],
            [20, 70],
            [50, 80]
        ),
        [(0, 0), (1, 0)]
    ),
    'touching-ends-left': (
        ([30], [40], [20], [30]), [(0, 0)]
    ),
    'touching-ends-right': (
        ([30], [40], [40], [50]), [(0, 0)]
    ),
    'almost-touching-ends-left': (
        ([30], [40], [20], [29]), []
    ),
    'almost-touching-ends-right': (
        ([30], [40], [41], [50]), []
    ),
    'two-separate-overlaps': (
        (
            [10, 20, 60],
            [30, 40, 80],
            [20, 60],
            [30, 80]
        ),
        [(0, 0), (1, 0), (2, 1)]
    ),
    'duplicated-region': (
        (
            [10, 10, 20],
            [60, 60, 40],
            [20, 70],
            [50, 80]
        ),
        [(0, 0), (1, 0), (2, 0)]
    ),
    'two-by-two-plus-one': (
        (
            [10, 30],
            [30, 40],
            [10, 10, 20],
            [20, 40, 40]
        ),
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]
    ),
    'three-by-three': (
        (
            [10, 20, 30],
            [70, 80, 90],
            [40, 50, 60],
            [50, 60, 70]
        ),
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
    ),
    'regions-of-length-1': (
        (
            [10, 30, 50],
            [20, 40, 50],
            [35, 50],
            [35, 50]
        ),
        [(1, 0), (2, 1)]
    ),
    'empty-regions': (
        (
            [10, 30, 50],
            [20, 40, 50],
            [35, 50],
            [34, 49]
        ),
        [(1, 0), (2, 1)]
    ),
    'gap-in-streams': (
        (
            [10, 60],
            [30, 80],
            [20, 70],
            [40, 90]
        ),
        [(0, 0), (1, 1)]
    )
}

@pytest.fixture(
    params=_overlapping_pairs_test_cases.values(),
    ids=list(_overlapping_pairs_test_cases.keys()),
    scope='module'
)
def overlapping_pairs_cases(request):
    return request.param


def test__overlapping_pairs(overlapping_pairs_cases):
    (start_a, end_a, start_b, end_b), expected = overlapping_pairs_cases
    expected = np.array(sorted(expected), dtype=int).reshape(len(expected), 2)
    result = overlapping_pairs(start_a, end_a, start_b, end_b)
    assert_array_equal(result, expected)


def test__overlapping_pairs__nosort():
    # heavily implementation-dependent
    start_a = np.array([10, 20])
    end_a = np.array([70, 80])
    start_b = np.array([40, 50])
    end_b = np.array([50, 60])
    expected_list = [(0, 0), (1, 0), (0, 1), (1, 1)]
    expected = np.array(expected_list).reshape(len(expected_list), 2)
    result = overlapping_pairs(start_a, end_a, start_b, end_b, sort_result=False)
    assert_array_equal(result, expected)


# def test_merge_by_overlapping_RangeSeries_no_groups(overlapping_RangeSeries_df):
#     df_a, df_b, df_exp = overlapping_RangeSeries_df
#     if df_a.group_a.nunique() > 1 or df_b.group_b.nunique() > 1:
#         return  # TODO: mark as ignored?
#     df_exp = df_exp.sort_values(list(df_exp.columns)).reset_index(drop=True)
#     df_res = merge_by_overlapping_ranges(
#         df_a, df_b,
#         left_region=('start_a', 'end_a'),
#         right_region=('start_b', 'end_b')
#     )
#     df_res = df_res.sort_values(list(df_res.columns)).reset_index(drop=True)
#     assert_frame_equal(
#         df_res, df_exp,
#         check_dtype=len(df_exp) > 0,
#         check_column_type=len(df_exp) > 0,
#         check_index_type=len(df_exp) > 0
#     )
#
#
# def test_merge_by_overlapping_RangeSeries_str_groups(overlapping_RangeSeries_df):
#     df_a, df_b, df_exp = overlapping_RangeSeries_df
#     df_exp = df_exp.sort_values(list(df_exp.columns)).reset_index(drop=True)
#     df_res = merge_by_overlapping_RangeSeries(
#         df_a, df_b,
#         left_region=('start_a', 'end_a'),
#         right_region=('start_b', 'end_b'),
#         left_on='group_a',
#         right_on='group_b'
#     )
#     df_res = df_res.sort_values(list(df_res.columns)).reset_index(drop=True)
#     assert_frame_equal(
#         df_res, df_exp,
#         check_dtype=len(df_exp) > 0,
#         check_column_type=len(df_exp) > 0,
#         check_index_type=len(df_exp) > 0
#     )
#
#
# def test_merge_by_overlapping_RangeSeries_RegionView(overlapping_RangeSeries_df):
#     df_a, df_b, df_exp = overlapping_RangeSeries_df
#     df_exp = df_exp.sort_values(list(df_exp.columns)).reset_index(drop=True)
#     region_a = RangeSeries(df_a.group_a, df_a.start_a, df_a.end_a)
#     region_b = RangeSeries(df_b.group_b, df_b.start_b, df_b.end_b)
#     df_res = merge_by_overlapping_RangeSeries(
#         df_a, df_b,
#         left_region=region_a,
#         right_region=region_b
#     )
#     df_res = df_res.sort_values(list(df_res.columns)).reset_index(drop=True)
#     assert_frame_equal(
#         df_res, df_exp,
#         check_dtype=len(df_exp) > 0,
#         check_column_type=len(df_exp) > 0,
#         check_index_type=len(df_exp) > 0
#     )
#
#
# def test_merge_by_overlapping_RangeSeries_suffixes(overlapping_RangeSeries_df):
#     df_a, df_b, df_exp = overlapping_RangeSeries_df
#     df_a = df_a.rename(columns={'end_a': 'end', 'extra_a': 'extra'})
#     df_b = df_b.rename(columns={'end_b': 'end', 'extra_b': 'extra'})
#     df_exp = df_exp.rename(columns={
#         'end_a': 'end_suffix_a', 'end_b': 'end_suffix_b',
#         'extra_a': 'extra_suffix_a', 'extra_b': 'extra_suffix_b'
#     })
#     df_exp = df_exp.sort_values(list(df_exp.columns)).reset_index(drop=True)
#     region_a = RangeSeries(df_a.group_a, df_a.start_a, df_a.end)
#     region_b = RangeSeries(df_b.group_b, df_b.start_b, df_b.end)
#     df_res = merge_by_overlapping_RangeSeries(
#         df_a, df_b,
#         left_region=region_a,
#         right_region=region_b,
#         suffixes=('_suffix_a', '_suffix_b')
#     )
#     df_res = df_res.sort_values(list(df_res.columns)).reset_index(drop=True)
#     assert_frame_equal(
#         df_res, df_exp,
#         check_dtype=len(df_exp) > 0,
#         check_column_type=len(df_exp) > 0,
#         check_index_type=len(df_exp) > 0
#     )


_overlapping_clusters_test_cases = {
    'empty': (
        ([], []), []
    ),
    'no-overlaps': (
        (
            [10, 40, 60],
            [20, 50, 70],
        ),
        [0, 1, 2]
    ),
    'simple-one-to-one': (
        (
            [10, 20, 50],
            [30, 40, 60],
        ),
        [0, 0, 1]
    ),
    'simple-two-to-one': (
        (
            [10, 30, 40, 60],
            [20, 80, 50, 70],
        ),
        [0, 1, 1, 1]
    ),
    'touching-ends': (
        (
            [10, 20],
            [20, 30],
        ),
        [0, 1]
    ),
    'overlap-size-1': (
        (
            [10, 20],
            [21, 30],
        ),
        [0, 0]
    ),
    'duplicated-region': (
        (
            [10, 30, 30, 30, 50],
            [20, 40, 40, 40, 60]
        ),
        [0, 1, 1, 1, 2]
    ),
    'chain-of-three': (
        (
            [10, 30, 40, 60, 90],
            [20, 50, 70, 80, 95]
        ),
        [0, 1, 1, 1, 2]
    ),
    'three-plus-two': (
        (
            [10, 20, 30, 60, 70],
            [50, 40, 50, 80, 90]
        ),
        [0, 0, 0, 1, 1]
    )
}


@pytest.fixture(
    params=_overlapping_clusters_test_cases.values(),
    ids=list(_overlapping_clusters_test_cases.keys()),
    scope='module'
)
def overlapping_clusters_cases(request):
    return request.param


def test__overlapping_clusters(overlapping_clusters_cases):
    (start, end), expected = overlapping_clusters_cases
    start = np.array(start, dtype=int)
    end = np.array(end, dtype=int)
    expected = np.array(expected, dtype=int)
    result = overlapping_clusters(start, end)
    assert_array_equal(result, expected)


@pytest.mark.parametrize(
    'starts_a,ends_a,starts_b,ends_b,expected_indices_a,expected_indices_b',
    [
        (
            [10, 10, 20],
            [40, 90, 80],
            [20, 30, 60],
            [40, 50, 70],
            [0, 1, 3],
            [2, 4, 5]
        ),
        (
            [20, 30, 60, 70],
            [30, 40, 60, 80],
            [10, 30, 60],
            [50, 50, 70],
            [1, 2, 4, 6],
            [0, 3, 5]
        ),
        (
            [10, 30, 60],
            [50, 50, 60],
            [], [],
            [0, 1, 2], [],
        ),
        (
            [], [], [], [], [], []
        )
    ],
    scope='session', ids=['equal_length', 'unequal_length', 'one_empty', 'both_empty']
)
def test_mergesort_ranges_indices(starts_a, ends_a, starts_b, ends_b, expected_indices_a, expected_indices_b):
    starts_a = np.array(starts_a, dtype=int)
    starts_b = np.array(starts_b, dtype=int)
    ends_a = np.array(ends_a, dtype=int)
    ends_b = np.array(ends_b, dtype=int)
    expected_indices_a = np.array(expected_indices_a, dtype=int)
    expected_indices_b = np.array(expected_indices_b, dtype=int)
    # normal order
    indices_a, indices_b = mergesort_ranges_indices(starts_a, ends_a, starts_b, ends_b)
    assert_array_equal(indices_a, expected_indices_a)
    assert_array_equal(indices_b, expected_indices_b)
    # swapped argument order
    indices_b_swap, indices_a_swap = mergesort_ranges_indices(starts_b, ends_b, starts_a, ends_a)
    assert_array_equal(indices_a_swap, expected_indices_a)
    assert_array_equal(indices_b_swap, expected_indices_b)


@pytest.mark.parametrize(
    'indices_a,indices_b,expected',
    [
        ([0, 2, 3], [1, 4], [0, 3, 1, 2, 4]),
        ([0, 2, 1], [], [0, 2, 1]),
        ([], [0, 2, 1], [0, 2, 1]),
        ([], [], [])
    ],
    scope='session', ids=['notempty', 'left_empty', 'right_empty', 'both_empty']
)
def test__target_indices_to_permutation(indices_a, indices_b, expected):
    indices_a = np.array(indices_a, dtype=int)
    indices_b = np.array(indices_b, dtype=int)
    result = target_indices_to_permutation(indices_a, indices_b)
    assert_array_equal(result, expected)


@pytest.fixture(params=[
    (np.array([], dtype=int), np.array([], dtype=int), -1),
    (np.array([3]), np.array([5]), -1),
    (np.array([3, 2]), np.array([5, 5]), 1),
    (np.array([3, 3]), np.array([5, 4]), 1),
    (np.array([1, 3, 3, 4]), np.array([6, 6, 6, 5]), -1),
    (np.array([1, 3, 1, 4]), np.array([6, 6, 6, 5]), 2),
    (np.array([1, 3, 3, 4]), np.array([6, 6, 5, 5]), 2)
],
    ids=['empty', 'len1', 'start_unsorted', 'end_unsorted', 'len4_sorted', 'tie_in_start', 'tie_in_end'],
    scope='session'
)
def sorted_ranges_test_cases(request):
    return request.param


def test__get_first_unsorted_index_in_ranges(sorted_ranges_test_cases):
    start, end, expected = sorted_ranges_test_cases
    result = get_first_unsorted_index_in_ranges(start, end, ascending=True)
    assert result == expected
    start_rev = start[::-1]
    end_rev = end[::-1]
    expected_rev = -1 if expected == -1 else len(start) - expected
    result = get_first_unsorted_index_in_ranges(start_rev, end_rev, ascending=False)
    assert result == expected_rev


def test__get_first_unsorted_index_in_points(sorted_ranges_test_cases):
    start, end, expected = sorted_ranges_test_cases
    points = 1000 * start + end
    result = get_first_unsorted_index_in_points(points, ascending=True)
    assert result == expected
    points_rev = points[::-1]
    expected_rev = -1 if expected == -1 else len(start) - expected
    result = get_first_unsorted_index_in_points(points_rev, ascending=False)
    assert result == expected_rev


def test__check_if_sorted_dont_raise(sorted_ranges_test_cases):
    start, end, expected = sorted_ranges_test_cases
    result = check_if_sorted(start, end, ascending=True, exc_type=None)
    assert result == expected
    start_rev = start[::-1]
    end_rev = end[::-1]
    expected_rev = -1 if expected == -1 else len(start) - expected
    result = check_if_sorted(start_rev, end_rev, ascending=False, exc_type=None)
    assert result == expected_rev


def test__check_if_sorted_raise(sorted_ranges_test_cases):
    start, end, expected = sorted_ranges_test_cases
    cm = nullcontext() if expected == -1 else pytest.raises(ValueError)
    with cm:
        result = check_if_sorted(start, end, ascending=True)
        assert result == expected
    start_rev = start[::-1]
    end_rev = end[::-1]
    expected_rev = -1 if expected == -1 else len(start) - expected
    cm = nullcontext() if expected == -1 else pytest.raises(ValueError)
    with cm:
        result = check_if_sorted(start_rev, end_rev, ascending=False)
        assert result == expected_rev
