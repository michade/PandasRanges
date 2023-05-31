import itertools
from dataclasses import dataclass
from functools import partial

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal, assert_index_equal
from pytest import param

from PandasRanges.ranges import *
from PandasRanges.testing_utils import TestCasesCsv, Vectorizer, make_pytest_id


#######################################################################################
# 1. CONSTRUCTION
#######################################################################################

@pytest.fixture(
    params=[
        Vectorizer(1, cls=pd.Series),
        Vectorizer(3, cls=pd.Series)
    ],
    ids=['1row', '3rows'],
    scope='session'
)
def as_series(request):
    return request.param


@pytest.fixture(
    params=[
        Vectorizer(1),
        Vectorizer(3)
    ],
    ids=['1row', '3rows'],
    scope='session'
)
def as_dataframe(request):
    return request.param


@pytest.fixture(
    params=[
        Vectorizer(1, cls=RangeSeries),
        Vectorizer(3, cls=RangeSeries)
    ],
    ids=['1row', '3rows'],
    scope='session'
)
def as_ranges(request):
    return request.param


def seq_along(scalar, vector):
    n = len(vector)
    res_vector = [scalar for _ in range(n)]
    if hasattr(vector, 'index'):
        res_vector = pd.Series(res_vector, index=vector.index)
    return res_vector


@pytest.fixture(params=[
    (4, 8),
    (4, 5),
    (4, 4),
    (-2, -1),
    (-3, 2),
    (0, 0)
], scope='session', ids=make_pytest_id)
def valid_range_init_args(request):
    return request.param


def test__RangeSeries_init__empty():
    iv = RangeSeries([])
    expected_start = pd.Series([], index=pd.RangeIndex(0), name='start', dtype=int)
    expected_end = pd.Series([], index=pd.RangeIndex(0), name='end', dtype=int)
    assert_series_equal(iv.start, expected_start)
    assert_series_equal(iv.end, expected_end)


def test__RangeSeries_init__scalar(valid_range_init_args):
    start, end = valid_range_init_args
    iv = RangeSeries(start, end)
    expected_start = pd.Series([start], index=pd.RangeIndex(1), name='start')
    expected_end = pd.Series([end], index=pd.RangeIndex(1), name='end')
    assert_series_equal(iv.start, expected_start)
    assert_series_equal(iv.end, expected_end)


def test__RangeSeries_init__series(valid_range_init_args, as_series):
    start, end = map(as_series, valid_range_init_args)
    start.name, end.name = 'start', 'end'
    iv = RangeSeries(start, end)
    index = start.index
    assert_series_equal(iv.start, start)
    assert_series_equal(iv.end, end)
    assert_index_equal(iv.start.index, index, exact=True)
    assert_index_equal(iv.end.index, index, exact=True)


@pytest.fixture(params=[
    (4, 2, ValueError),
    (4, 1, ValueError),
    (4, 0, ValueError),
    (4, -1, ValueError),
    (0, -1, ValueError),
    (-1, -2, ValueError),
    ('1', 2, TypeError),
    (1, '2', TypeError),
    (2.0, 5, TypeError),
    (2, 5.0, TypeError)
], scope='session', ids=make_pytest_id)
def invalid_range_init_args(request):
    return request.param


def test__RangeSeries_init__invalid(invalid_range_init_args, as_series):
    start, end, expected_exception = invalid_range_init_args
    start = as_series(start)
    end = as_series(end)
    with pytest.raises(expected_exception):
        RangeSeries(start, end)


@pytest.mark.parametrize(
    'call,expected_cols,expected_names',
    [
        (
                partial(RangeSeries),
                ('start', 'end'), ('start', 'end')
        ),
        (
                partial(RangeSeries, prefix='foo_'),
                ('foo_start', 'foo_end'), ('foo_start', 'foo_end')
        ),
        (
                partial(RangeSeries, suffix='_foo'),
                ('start_foo', 'end_foo'), ('start_foo', 'end_foo')
        ),
        (
                partial(RangeSeries, prefix='foo_', set_names=('test_start', 'test_end')),
                ('foo_start', 'foo_end'), ('test_start', 'test_end')
        )
    ],
    scope='session', ids=['default', 'prefix', 'suffix', 'set_names']
)
def test__RangeSeries_init__from_dataframe(call, expected_cols, expected_names, as_dataframe):
    df = as_dataframe(1, 2, 3, 4, 5, 6)
    df.columns = ['start', 'end', 'foo_start', 'foo_end', 'start_foo', 'end_foo']
    expected_start = df[expected_cols[0]]
    expected_start.name = expected_names[0]
    expected_end = df[expected_cols[1]]
    expected_end.name = expected_names[1]
    result = call(df)
    assert result.names == expected_names
    assert_series_equal(result.start, expected_start, check_index_type=True)
    assert_series_equal(result.end, expected_end, check_index_type=True)


def test__RangeSeries_init__set_index():
    old_index = pd.RangeIndex(2)
    start = pd.Series([10, 20], name='start', index=old_index)
    end = pd.Series([30, 40], name='end', index=old_index)
    new_index = pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')], names=['idx_1', 'idx_2'])
    expected_start = pd.Series(start.array, name='start', index=new_index)
    expected_end = pd.Series(end.array, name='end', index=new_index)
    result = RangeSeries(start, end, set_index=new_index)
    assert start.index is old_index, "Original series must not change"
    assert end.index is old_index, "Original series must not change"
    assert_index_equal(result.index, new_index)
    assert_series_equal(result.start, expected_start, check_index_type=True)
    assert_series_equal(result.end, expected_end, check_index_type=True)


def test__RangeSeries_init__set_names():
    start = pd.Series([10, 20], name='old_start')
    end = pd.Series([30, 40], name='old_end')
    expected_start = pd.Series(start, name='new_start')
    expected_end = pd.Series(end, name='new_end')
    new_names = ('new_start', 'new_end')
    result = RangeSeries(start, end, set_names=new_names)
    assert start.name == 'old_start', "Original series must not change"
    assert end.name == 'old_end', "Original series must not change"
    assert result.names == new_names
    assert_series_equal(result.start, expected_start, check_index_type=True)
    assert_series_equal(result.end, expected_end, check_index_type=True)


@pytest.mark.parametrize(
    'result,expected',
    [
        (
                partial(RangeSeries.from_records, [(10, 20), (30, 40)]),
                partial(RangeSeries, [10, 30], [20, 40])
        ),
        (
                partial(RangeSeries.from_records, []),
                partial(RangeSeries, [], [])
        ),
        (
                partial(RangeSeries.from_records, [(10, 20), (30, 40)], names=['foo_start', 'foo_end']),
                partial(RangeSeries, [10, 30], [20, 40], set_names=('foo_start', 'foo_end'))
        ),
        (
                partial(
                    RangeSeries.from_records,
                    [('a', 'x', 10, 20), ('b', 'y', 30, 40)],
                    names=['idx_1', 'idx_2', 'foo_start', 'foo_end']
                ),
                partial(
                    RangeSeries, [10, 30], [20, 40], set_names=('foo_start', 'foo_end'),
                    set_index=pd.MultiIndex.from_tuples([('a', 'x'), ('b', 'y')], names=['idx_1', 'idx_2'])
                )
        )
    ],
    scope='session', ids=['nonempty', 'empty', 'nondefault_names', 'with_multiindex']
)
def test__RangeSeries_from_records(result, expected):
    assert_ranges_equal(result(), expected())


def test_RangeSeries_call(as_ranges):
    df1 = as_ranges(20, 30).to_frame(names=['foo_start', 'foo_end'])
    df2 = as_ranges(50, 60).to_frame(names=['foo_start', 'foo_end'])
    iv = RangeSeries(df1, prefix='foo_')
    expected = RangeSeries(df2, prefix='foo_')
    result = iv(df2)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'strings,names,expected_df',
    [
        pytest.param(
            '12-34', None,
            DataFrame({'start': 12, 'end': 34}, index=pd.RangeIndex(1)),
            id='scalar'
        ),
        pytest.param(
            ['12-34', '56-78'], None,
            DataFrame({'start': [12, 56], 'end': [34, 78]}, index=pd.RangeIndex(2)),
            id='list_ony_coords'
        ),
        pytest.param(
            ['A:12-34', 'B:56-78'], None,
            DataFrame({'start': [12, 56], 'end': [34, 78]}, index=pd.Index(['A', 'B'], name='level_1')),
            id='list_single_index'
        ),
        pytest.param(
            ['X:A:12-34', 'Y:B:56-78'], None,
            DataFrame(
                {'start': [12, 56], 'end': [34, 78]},
                index=pd.MultiIndex.from_tuples([('X', 'A'), ('Y', 'B')], names=['level_1', 'level_2'])
            ),
            id='list_double_index'
        ),
        pytest.param(
            ['12-34', '56-78'], ['foo_start', 'foo_end'],
            DataFrame({'foo_start': [12, 56], 'foo_end': [34, 78]}, index=pd.RangeIndex(2)),
            id='list_ony_coords_names'
        ),
        pytest.param(
            ['A:12-34', 'B:56-78'], ['level_1', 'foo_start', 'foo_end'],
            DataFrame(
                {'foo_start': [12, 56], 'foo_end': [34, 78]},
                index=pd.Index(['A', 'B'], name='level_1')
            ),
            id='list_single_index_names'
        ),
        pytest.param(
            ['X:A:12-34', 'Y:B:56-78'], ['level_1', 'level_2', 'foo_start', 'foo_end'],
            DataFrame(
                {'foo_start': [12, 56], 'foo_end': [34, 78]},
                index=pd.MultiIndex.from_tuples([('X', 'A'), ('Y', 'B')], names=['level_1', 'level_2'])
            ),
            id='list_double_index_names'
        ),
        pytest.param(
            pd.Series(['12-34', '56-78'], index=pd.Index([3, 5], name='orig_idx')), None,
            DataFrame({'start': [12, 56], 'end': [34, 78]}, index=pd.Index([3, 5], name='orig_idx')),
            id='series_ony_coords'
        ),
        pytest.param(
            pd.Series(['A:12-34', 'B:56-78'], index=pd.Index([3, 5], name='orig_idx')), None,
            DataFrame(
                {'start': [12, 56], 'end': [34, 78]},
                index=pd.MultiIndex.from_tuples([(3, 'A'), (5, 'B')], names=['orig_idx', 'level_1'])
            ),
            id='series_single_index'
        ),
        pytest.param(
            pd.Series(['A:12-34', 'B:56-78'], index=pd.Index([3, 5], name='orig_idx')),
            ['idx_1', 'foo_start', 'foo_end'],
            DataFrame(
                {'foo_start': [12, 56], 'foo_end': [34, 78]},
                index=pd.MultiIndex.from_tuples([(3, 'A'), (5, 'B')], names=['orig_idx', 'idx_1'])
            ),
            id='series_single_index_names'
        )
    ],
    scope='session'
)
def test__RangeSeries_from_string(strings, names, expected_df):
    result = RangeSeries.from_string(strings, names=names)
    expected_df = expected_df.copy()
    start_col, end_col = names[-2:] if names is not None else ['start', 'end']
    expected = RangeSeries(expected_df[start_col], expected_df[end_col])
    assert_ranges_equal(result, expected)


def test__RangeSeries_from_string__mixed_separators():
    result = RangeSeries.from_string(['X|A|12x34', 'Y|B|56x78'], sep='x', idx_sep='|')
    expected_df = DataFrame(
        {'start': [12, 56], 'end': [34, 78]},
        index=pd.MultiIndex.from_tuples([('X', 'A'), ('Y', 'B')], names=['level_1', 'level_2'])
    )
    expected = RangeSeries(expected_df)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'strings,names',
    [
        ('12', None),
        ('-12', None),
        ('12-', None),
        ('12-34-56', None),
        ('A-12-34', None),
        (':12-34', None),
        ('12-34:', None),
        ('12-34:A', None),
        (':A:12-34', None),
        ('A-C', None),
        ('12-34', []),
        ('12-34', ['foo_start']),
        ('12-34', ['foo_start', 'foo_end', 'extra']),
        ('X:A:12-34', ['idx_1', 'foo_start', 'foo_end'])
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_from_string__invalid(strings, names):
    with pytest.raises(ValueError):
        RangeSeries.from_string(strings, names=names)


#######################################################################################
# 2. BASIC PROPERTIES
#######################################################################################


def test__RangeSeries_start():
    start = pd.Series([1, 2, 3], name='foo_start', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    end = pd.Series([4, 5, 6], name='foo_end', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    iv = RangeSeries(start.copy(), end.copy())
    assert_series_equal(iv.start, start)


def test__RangeSeries_end():
    start = pd.Series([1, 2, 3], name='foo_start', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    end = pd.Series([4, 5, 6], name='foo_end', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    iv = RangeSeries(start.copy(), end.copy())
    assert_series_equal(iv.end, end)


def test__RangeSeries_index():
    idx = pd.Index(['A', 'B', 'C'], name='foo_index')
    start = pd.Series([1, 2, 3], name='foo_start', index=idx.copy())
    end = pd.Series([4, 5, 6], name='foo_end', index=idx.copy())
    iv = RangeSeries(start, end)
    assert_index_equal(iv.index, idx)


def test__RangeSeries_names():
    start = pd.Series([1, 2, 3], name='foo_start', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    end = pd.Series([4, 5, 6], name='foo_end', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    iv = RangeSeries(start, end)
    assert iv.names == ('foo_start', 'foo_end')


def test__RangeSeries_dtype():
    start = pd.Series([1, 2, 3], name='foo_start', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    end = pd.Series([4, 5, 6], name='foo_end', index=pd.Index(['A', 'B', 'C'], name='foo_index'))
    iv = RangeSeries(start, end)
    assert iv.dtype == start.dtype
    assert iv.dtype == end.dtype


@pytest.mark.parametrize(
    'start,end,expected_len',
    [
        ([], [], 0),
        ([10], [20], 1),
        ([10] * 3, [20] * 3, 3)
    ],
    scope='session', ids=['len_0', 'len_1', 'len_3']
)
def test__RangeSeries_len(start, end, expected_len):
    ranges = RangeSeries(start, end)
    assert len(ranges) == expected_len


def test__RangeSeries_set_index():
    old_index = pd.Index(['A', 'B', 'C'], name='old_index')
    iv = RangeSeries([0, 1, 2], [3, 4, 5], set_index=old_index)
    new_index = pd.Index(['D', 'E', 'F'], name='new_index')
    iv2 = iv.set_index(new_index)
    assert iv.index is old_index, "Original seriesm must remain unchanged"
    assert iv2.index is new_index


@pytest.mark.parametrize(
    'labels,keep,expected_index',
    [
        (None, False, pd.RangeIndex(3)),
        ([], False, None),
        (['l2'], False, pd.MultiIndex.from_tuples([('A', 1), ('B', 1), ('B', 1)], names=['l1', 'l3'])),
        (['l1', 'l2'], False, pd.Index([1, 1, 1], name='l3')),
        (['l1', 'l2', 'l3'], False, pd.RangeIndex(3)),
        ([], True, pd.RangeIndex(3)),
        (['l2'], True, pd.Index([3, 3, 4], name='l2')),
        (['l1', 'l3'], True, pd.MultiIndex.from_tuples([('A', 1), ('B', 1), ('B', 1)], names=['l1', 'l3'])),
        (['l1', 'l2', 'l3'], True, None)
    ],
    scope='session',
    ids=['no_params', 'drop_empty', 'drop_2', 'drop_12', 'drop_123', 'keep_empty', 'keep_2', 'keep_13', 'keep_123']
)
def test__RangeSeries_reset_index(labels, keep, expected_index):
    old_index = pd.MultiIndex.from_tuples(
        [('A', 3, 1), ('B', 3, 1), ('B', 4, 1)],
        names=['l1', 'l2', 'l3'],
    )
    old_index.name = 'old_index'
    if expected_index is None:
        expected_index = old_index.copy()
    iv = RangeSeries([1, 2, 3], [4, 5, 6], set_index=old_index)
    if keep:
        iv2 = iv.reset_index(keep=labels)
    else:
        iv2 = iv.reset_index(labels=labels)
    assert iv.index is old_index, "Original seriesm must remain unchanged"
    assert_index_equal(iv2.index, expected_index)


@pytest.mark.parametrize(
    'kwargs,expected_df',
    [
        (
                {'labels': ['l1']},
                pd.DataFrame({
                    'l1': ['A', 'B', 'B'],
                    'start': [1, 2, 3],
                    'end': [4, 5, 6]
                }, index=pd.Index([3, 3, 4], name='l2'))
        ),
        (
                {'keep': ['l2']},
                pd.DataFrame({
                    'l1': ['A', 'B', 'B'],
                    'start': [1, 2, 3],
                    'end': [4, 5, 6]
                }, index=pd.Index([3, 3, 4], name='l2'))
        )
    ],
    scope='session', ids=['using_labels', 'using_keep']
)
def test__RangeSeries_reset_index__nodrop(kwargs, expected_df):
    old_index = pd.MultiIndex.from_tuples(
        [('A', 3), ('B', 3), ('B', 4)],
        names=['l1', 'l2']
    )
    old_index.name = 'old_index'
    iv = RangeSeries([1, 2, 3], [4, 5, 6], set_index=old_index)
    result_df = iv.reset_index(drop=False, **kwargs)
    assert iv.index is old_index, "Original seriesm must remain unchanged"
    assert_frame_equal(result_df, expected_df)


@pytest.mark.parametrize(
    'n',
    [0, 1, 5, 100],
    scope='session', ids=make_pytest_id
)
def test_RangeSeries_head(n, as_ranges):
    iv = as_ranges(10, 20)
    expected = iv(iv.to_frame().head(n))
    result = iv.head(n)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'n',
    [0, 1, 5, 100],
    scope='session', ids=make_pytest_id
)
def test_RangeSeries_tail(n, as_ranges):
    iv = as_ranges(10, 20)
    expected = iv(iv.to_frame().tail(n))
    result = iv.tail(n)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'data,suffix',
    [
        (DataFrame({'start': [10, 20], 'end': [30, 40]}, index=pd.RangeIndex(2)), ''),
        (DataFrame({'start': [], 'end': []}, index=pd.RangeIndex(0)), ''),
        (DataFrame({'start_foo': [10, 20], 'end_foo': [30, 40]}, index=pd.RangeIndex(2)), '_foo'),
        (DataFrame({'start_foo': [], 'end_foo': []}, index=pd.RangeIndex(0)), '_foo')
    ],
    scope='session', ids=['default', 'default_empty', 'named', 'named_empty']
)
def test__RangeSeries_itertuples(data, suffix):
    iv = RangeSeries(data, suffix=suffix)
    # without index:
    expected = list(data.itertuples())
    result = list(iv.itertuples())
    assert result == expected
    # with index:
    expected = list(data.itertuples(index=True))
    result = list(iv.itertuples(index=True))
    assert result == expected


@pytest.mark.parametrize(
    'data,i_to_keep',
    [
        param(
            DataFrame({'start': [10, 20, 30, 40], 'end': [11, 20, 31, 40]}, index=pd.RangeIndex(4)), [0, 2],
            id='some_empty'
        ),
        param(
            DataFrame({'start': [10, 20, 30, 40], 'end': [11, 21, 31, 41]}, index=pd.RangeIndex(4)), [0, 1, 2, 3],
            id='no_empty'
        ),
        param(
            DataFrame({'start': [10, 20, 30, 40], 'end': [10, 20, 30, 40]}, index=pd.RangeIndex(4)), [],
            id='all_empty'
        ),
        param(
            DataFrame({'start': [], 'end': []}, index=pd.RangeIndex(0)), [],
            id='empty_frame'
        )
    ],
    scope='session'
)
def test__RangeSeries_drop_empty(data: pd.DataFrame, i_to_keep):
    iv = RangeSeries(data)
    expected = RangeSeries(data.iloc[i_to_keep, :])
    result = iv.drop_empty()
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'indices',
    [
        [],
        [1],
        [0, 2],
        [0, 1, 2],
        [2, 0, 1],
        [1, 1, 1],
        [2, 1, 2, 0, 1]
    ],
    ids=['take_empty', 'len_1', 'len_2', 'take_all', 'reorder', 'duplicates', 'longer'],
    scope='session'
)
def test__RangeSeries_subset(indices):
    df = DataFrame({
        'idx': ['A', 'B', 'C'],
        'start': [10, 20, 30],
        'end': [30, 40, 50]
    }).set_index('idx')
    iv = RangeSeries(df)
    expected = RangeSeries(df.iloc[indices])
    result = iv.subset(indices)
    assert_ranges_equal(result, expected)


def test__RangeSeries_subset__out_of_bounds():
    df = DataFrame({
        'idx': ['A', 'B', 'C'],
        'start': [10, 20, 30],
        'end': [30, 40, 50]
    }).set_index('idx')
    iv = RangeSeries(df)
    with pytest.raises(IndexError):
        iv.subset([3])


def test__RangeSeries_subset__not_integer():
    df = DataFrame({
        'idx': ['A', 'B', 'C'],
        'start': [10, 20, 30],
        'end': [30, 40, 50]
    }).set_index('idx')
    iv = RangeSeries(df)
    with pytest.raises(ValueError):
        iv.subset(['A', 'C'])


#######################################################################################
# 4. CONVERSION TO OTHER TYPES
#######################################################################################


@pytest.mark.parametrize(
    'names',
    [
        None,
        ['foo_start', 'foo_end']
    ],
    scope='session', ids=['default_names', 'set_names']
)
def test_RangeSeries_to_frame(names, as_ranges):
    iv = as_ranges(10, 20)
    expected = DataFrame({'start': iv.start, 'end': iv.end}, index=iv.index)
    if names is not None:
        expected.columns = names
    result = iv.to_frame(names=names)
    assert_frame_equal(result, expected)


_test__RangeSeries_fmt_df_1idx = DataFrame(
    {'start': [1, 4], 'end': [23, 567]},
    index=pd.Index(list('AB'), name='idx_1')
)

_test__RangeSeries_fmt_df_2idx = DataFrame(
    {'start': [1, 4], 'end': [23, 567]},
    index=pd.MultiIndex.from_tuples(zip('AB', range(2)), name=['idx_1', 'idx_2'])
)


@pytest.mark.parametrize(
    'data,kwargs,expected',
    [
        (_test__RangeSeries_fmt_df_1idx, {'index': False}, ['1-23', '4-567']),
        (_test__RangeSeries_fmt_df_1idx, {'index': True}, ['A:1-23', 'B:4-567']),
        (_test__RangeSeries_fmt_df_2idx, {'index': True}, ['A:0:1-23', 'B:1:4-567']),
        (_test__RangeSeries_fmt_df_2idx, {'sep': 'x', 'idx_sep': '|'}, ['A|0|1x23', 'B|1|4x567'])
    ],
    scope='session', ids=['no_index', 'single_index', 'double_index', 'changed_separators']
)
def test__RangeSeries_to_string(data, kwargs, expected):
    iv = RangeSeries(data)
    expected = pd.Series(expected, index=data.index)
    result = iv.to_string(**kwargs)
    assert_series_equal(result, expected)


@pytest.mark.parametrize(
    'data,fmt,expected',
    [
        (_test__RangeSeries_fmt_df_1idx, '{start}-{end:03d}', ['1-023', '4-567']),
        (_test__RangeSeries_fmt_df_1idx, '{idx_1}:{start}-{end:03d}', ['A:1-023', 'B:4-567']),
        (_test__RangeSeries_fmt_df_2idx, '{idx_1}:{idx_2}:{start}-{end:03d}', ['A:0:1-023', 'B:1:4-567']),
        (_test__RangeSeries_fmt_df_2idx, '{end}x{start:03d}|{idx_2}', ['23x001|0', '567x004|1'])
    ],
    scope='session', ids=['no_index', 'single_index', 'double_index', 'mixed_format']
)
def test__RangeSeries_format(data, fmt, expected):
    iv = RangeSeries(data)
    expected = pd.Series(expected, index=data.index)
    result = iv.format(fmt)
    assert_series_equal(result, expected)


#######################################################################################
# 5. ELEMENT-WISE PROPERTIES
#######################################################################################

@pytest.mark.parametrize(
    'range_tuple,expected_bool_value',
    [
        ((4, 8), False),
        ((0, 1), False),
        ((-1, 0), False),
        ((-2, 1), False),
        ((4, 4), True),
        ((0, 0), True),
        ((-2, -2), True)
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_is_empty(range_tuple, expected_bool_value, as_ranges):
    iv = as_ranges(range_tuple)
    expected_bool_value = seq_along(expected_bool_value, iv)
    assert_series_equal(iv.is_empty(), expected_bool_value)


@pytest.mark.parametrize(
    'range_tuple,expected_bool_value',
    [
        ((4, 8), True),
        ((0, 1), True),
        ((-1, 0), True),
        ((-2, 1), True),
        ((4, 4), True),
        ((0, 0), True),
        ((-2, -2), True),
        ((1, 0), False),
        ((0, -1), False),
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_is_valid(range_tuple, expected_bool_value):
    iv = RangeSeries.from_records([range_tuple] * 3, validate=False)
    expected_bool_value = seq_along(expected_bool_value, iv)
    assert_series_equal(iv.is_valid(), expected_bool_value)


@pytest.mark.parametrize(
    'range_tuple,expected_length',
    [
        ((4, 8), 4),
        ((0, 1), 1),
        ((-1, 0), 1),
        ((-2, 1), 3),
        ((4, 4), 0),
        ((0, 0), 0),
        ((-2, -2), 0)
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_length(range_tuple, expected_length, as_ranges):
    iv = as_ranges(range_tuple)
    expected_length = seq_along(expected_length, iv)
    assert_series_equal(iv.length, expected_length)


@pytest.mark.parametrize(
    'range_tuple,expected_center',
    [
        ((4, 9), 6),
        ((4, 8), 6),
        ((4, 7), 5),
        ((4, 5), 4),
        ((4, 4), 4),
        ((3, 9), 6),
        ((3, 8), 5),
        ((3, 7), 5),
        ((3, 4), 3),
        ((3, 3), 3)
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_center(range_tuple, expected_center, as_ranges):
    iv = as_ranges(range_tuple)
    expected_center = seq_along(expected_center, iv)
    assert_series_equal(iv.center, expected_center)


#######################################################################################
# 6. ELEMENT-WISE OPERATIONS WITH SCALAR
#######################################################################################

@pytest.mark.parametrize(
    'side', ['left', 'right'],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_clip__one_side(side, scalar_coord_test_cases, as_ranges):
    range = as_ranges(scalar_coord_test_cases.A_start, scalar_coord_test_cases.A_end)
    coord = seq_along(scalar_coord_test_cases.get_scalar('coord'), range)
    expected = as_ranges(*scalar_coord_test_cases.get_coord_pair(f'clip_{side}'))
    kwargs = {side: coord}
    result = range.clip(**kwargs)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'direction', ['AB', 'BA'],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_clip__both_sides(direction, binary_op_test_cases, as_ranges):
    range_A = as_ranges(binary_op_test_cases.A_start, binary_op_test_cases.A_end)
    range_B = as_ranges(binary_op_test_cases.B_start, binary_op_test_cases.B_end)
    method = 'clip'
    expected = as_ranges(*binary_op_test_cases.get_coord_pair(method, direction))
    bound_method, target = binary_op_test_cases.get_binary_op_from_objs(method, range_A, range_B, direction)
    result = bound_method(left=target.start, right=target.end)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'side', ['left', 'right', 'both'],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_expand(side, scalar_offset_test_cases, as_ranges):
    range = as_ranges(scalar_offset_test_cases.A_start, scalar_offset_test_cases.A_end)
    offset = seq_along(scalar_offset_test_cases.get_scalar('offset'), range)
    expected = as_ranges(*scalar_offset_test_cases.get_coord_pair(f'expand_{side}'))
    if side == 'left':
        result = range.expand(left=offset, right=0)
    elif side == 'right':
        result = range.expand(left=0, right=offset)
    else:  # side == 'both
        result = range.expand(offset)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'range,left,right,expected',
    [
        ((20, 30), 5, 0, (15, 30)),
        ((20, 30), 0, 5, (20, 35)),
        ((20, 30), 5, 3, (15, 33)),
        ((20, 30), -5, 0, (25, 30)),
        ((20, 30), 0, -5, (20, 25)),
        ((20, 30), -5, -3, (25, 27)),
        ((20, 30), -5, -5, (25, 25)),
        ((20, 30), 5, -5, (15, 25)),
        ((20, 30), -5, 5, (25, 35)),
        ((20, 30), 10, -5, (10, 25)),
        ((20, 30), -5, 10, (25, 40)),
        ((20, 30), 10, -10, (10, 20)),
        ((20, 30), -10, 10, (30, 40)),
        ((20, 30), 10, -15, (10, 15)),
        ((20, 30), -15, 10, (30, 40)),
        ((20, 30), 10, -30, (10, 10)),
        ((20, 30), -30, 10, (30, 40))
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_expand__asymmetric(range, left, right, expected, as_ranges):
    range = as_ranges(range)
    left = seq_along(left, range)
    right = seq_along(right, range)
    expected = as_ranges(expected)
    result = range.expand(left=left, right=right)
    assert_ranges_equal(result, expected)


def test__RangeSeries_split_at(scalar_coord_test_cases, as_ranges):
    range = as_ranges(scalar_coord_test_cases.A_start, scalar_coord_test_cases.A_end)
    coord = seq_along(scalar_coord_test_cases.get_scalar('coord'), range)
    expected_left = as_ranges(*scalar_coord_test_cases.get_coord_pair(f'split_at_left'))
    expected_right = as_ranges(*scalar_coord_test_cases.get_coord_pair(f'split_at_right'))
    result_left, result_right = range.split_at(coord)
    assert_ranges_equal(result_left, expected_left)
    assert_ranges_equal(result_right, expected_right)


@pytest.mark.parametrize(
    'range,length,expected_left,expected_right',
    [
        ((20, 30), 0, (20, 20), (20, 30)),
        ((20, 30), 1, (20, 21), (21, 30)),
        ((20, 30), 2, (20, 22), (22, 30)),
        ((20, 30), 10, (20, 30), (30, 30)),
        ((20, 30), 11, (20, 30), (31, 31)),
        ((20, 30), -1, (20, 29), (29, 30)),
        ((20, 30), -2, (20, 28), (28, 30)),
        ((20, 30), -10, (20, 20), (20, 30)),
        ((20, 30), -11, (19, 19), (20, 30))
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_split_at_length(range, length, expected_left, expected_right, as_ranges):
    range = as_ranges(range)
    length = seq_along(length, range)
    expected_left = as_ranges(expected_left)
    expected_right = as_ranges(expected_right)
    result_left, result_right = range.split_at_length(length)
    assert_ranges_equal(result_left, expected_left)
    assert_ranges_equal(result_right, expected_right)


@pytest.mark.parametrize(
    'range,ratio,expected_left,expected_right',
    [
        ((20, 30), 0.0, (20, 20), (20, 30)),
        ((20, 30), 0.01, (20, 20), (20, 30)),
        ((20, 30), 0.09, (20, 21), (21, 30)),
        ((20, 30), 0.5, (20, 25), (25, 30)),
        ((20, 31), 0.5, (20, 26), (26, 31)),
        ((20, 30), 0.9, (20, 29), (29, 30)),
        ((20, 30), 1.0, (20, 30), (30, 30)),
        ((20, 30), 1.1, (20, 30), (31, 31)),
        ((20, 30), -0.01, (20, 30), (30, 30)),
        ((20, 30), -0.09, (20, 29), (29, 30)),
        ((20, 30), -0.5, (20, 25), (25, 30)),
        ((20, 31), -0.5, (20, 25), (25, 31)),
        ((20, 30), -0.9, (20, 21), (21, 30)),
        ((20, 30), -1.0, (20, 20), (20, 30)),
        ((20, 30), -1.1, (19, 19), (20, 30))
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_split_at_ratio(range, ratio, expected_left, expected_right, as_ranges):
    range = as_ranges(range)
    ratio = seq_along(ratio, range)
    expected_left = as_ranges(expected_left)
    expected_right = as_ranges(expected_right)
    result_left, result_right = range.split_at_ratio(ratio)
    assert_ranges_equal(result_left, expected_left)
    assert_ranges_equal(result_right, expected_right)



@pytest.mark.parametrize(
    'coords_start,coords_end,size,start,expected_start_bin,expected_end_bin',
    [
        ([], [], 10, 0, [], []),
        (
            [
                -1, -1, -1, -1, -1, -1,
                0, 0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2, 2,
                3, 3,
                4
            ],
            [
                -1, 0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
                1, 2, 3, 4,
                2, 3, 4,
                3, 4,
                4
            ],
            3, 0,
            [  # ends = -1, 0, 1, 2, 3, 4
                -1, -1, -1, -1, -1, -1,  # start = -1
                0, 0, 0, 0, 0,  # start = 0
                0, 0, 0, 0,  # start = 1
                0, 0, 0,  # start = 2
                1, 1,  # start = 3
                1,  # start = 4
            ],
            [  # ends = -1, 0, 1, 2, 3, 4
                0, 0, 1, 1, 1, 2,  # start = -1
                1, 1, 1, 1, 2,  # start = 0
                1, 1, 1, 2,  # start = 1
                1, 1, 2,  # start = 2
                2, 2,  # start = 3
                2,  # start = 4
            ],
        ),
        (
            [9, 10, 11, 20, 20, 20],
            [39, 40, 41, 29, 50, 31],
            10, 0,
            [0, 1, 1, 2, 2, 2],
            [4, 4, 5, 3, 5, 4]
        ),
        (
            [9, 10, 11, 21, 21, 21],
            [40, 41, 42, 29, 50, 31],
            10, 1,
            [0, 0, 1, 2, 2, 2],
            [4, 4, 5, 3, 5, 3]
        )
    ],
    scope='session',  ids=['empty', 'step3start0grid', 'step10start0', 'step10start1']
)
def test__RangeSeries_bin_indices(coords_start, coords_end, size, start, expected_start_bin, expected_end_bin):
    iv = RangeSeries(coords_start, coords_end)
    expected = RangeSeries(expected_start_bin, expected_end_bin)
    result = iv.bin_indices(size=size, start=start)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'coords_start,coords_end,size,start,expected_start,expected_end',
    [
        ([], [], 10, 0, [], []),
        (
                [
                    -1, -1, -1, -1, -1, -1,
                    0, 0, 0, 0, 0,
                    1, 1, 1, 1,
                    2, 2, 2,
                    3, 3,
                    4
                ],
                [
                    -1, 0, 1, 2, 3, 4,
                    0, 1, 2, 3, 4,
                    1, 2, 3, 4,
                    2, 3, 4,
                    3, 4,
                    4
                ],
                3, 0,
                [  # ends = -1, 0, 1, 2, 3, 4
                    -3, -3, -3, -3, -3, -3,  # start = -1
                    0, 0, 0, 0, 0,  # start = 0
                    0, 0, 0, 0,  # start = 1
                    0, 0, 0,  # start = 2
                    3, 3,  # start = 3
                    3,  # start = 4
                ],
                [  # ends = -1, 0, 1, 2, 3, 4
                    0, 0, 3, 3, 3, 6,  # start = -1
                    3, 3, 3, 3, 6,  # start = 0
                    3, 3, 3, 6,  # start = 1
                    3, 3, 6,  # start = 2
                    6, 6,  # start = 3
                    6,  # start = 4
                ],
        ),
        (
                [9, 10, 11, 20, 20, 20],
                [39, 40, 41, 29, 50, 31],
                10, 0,
                [0, 10, 10, 20, 20, 20],
                [40, 40, 50, 30, 50, 40]
        ),
        (
                [9, 10, 11, 21, 21, 21],
                [40, 41, 42, 29, 50, 31],
                10, 1,
                [1, 1, 11, 21, 21, 21],
                [41, 41, 51, 31, 51, 31]
        )
    ],
    scope='session', ids=['empty', 'step10start0', 'step10start1', 'step5start0']
)
def test__RangeSeries_bin(coords_start, coords_end, size, start, expected_start, expected_end):
    iv = RangeSeries(coords_start, coords_end)
    expected = RangeSeries(expected_start, expected_end)
    result = iv.bin(size=size, start=start)
    assert_ranges_equal(result, expected)


#######################################################################################
# 7. HORIZONTAL OPERATIONS (ELEMENT-WISE WITH OTHER RangeSeries INSTANCE)
#######################################################################################

class RangeSeriesTestCases(TestCasesCsv):
    def get_scalar(self, name):
        result = self[name]
        if result == 'T':
            return True
        elif result == 'F':
            return False
        else:
            return result

    def get_coord_pair(self, name):
        start = self[f'{name}_start']
        end = self[f'{name}_end']
        return start, end

    def get_method(self, obj, method):
        if method.endswith('_'):
            method = f'__{method}_'
        return getattr(obj, method)


class BinaryOpTestCases(RangeSeriesTestCases):
    path = './binary_op_test_cases.csv'

    def get_scalar(self, name, direction='AB'):
        try:
            result = super().get_scalar(f'{direction}_{name}')
        except AttributeError:
            if direction == 'BA':
                direction = 'AB'
                result = super().get_scalar(f'{direction}_{name}')
            else:
                raise
        return result

    def get_coord_pair(self, name, direction='AB'):
        try:
            start, end = super().get_coord_pair(f'{direction}_{name}')
        except AttributeError:
            if direction == 'BA':
                direction = 'AB'
                start, end = super().get_coord_pair(f'{direction}_{name}')
            else:
                raise
        return start, end

    def get_binary_op_from_objs(self, method, objA, objB, direction="AB"):
        if direction == 'AB':
            return self.get_method(objA, method), objB
        else:
            return self.get_method(objB, method), objA


@pytest.fixture(scope='session', params=BinaryOpTestCases.params())
def binary_op_test_cases(request):
    return BinaryOpTestCases(request.param)


class ScalarOffsetTestCases(RangeSeriesTestCases):
    path = './scalar_offset_test_cases.csv'


@pytest.fixture(scope='session', params=ScalarOffsetTestCases.params())
def scalar_offset_test_cases(request):
    return ScalarOffsetTestCases(request.param)


class ScalarCoordTestCases(RangeSeriesTestCases):
    path = './scalar_coord_test_cases.csv'


@pytest.fixture(scope='session', params=ScalarCoordTestCases.params())
def scalar_coord_test_cases(request):
    return ScalarCoordTestCases(request.param)


@pytest.mark.parametrize(
    'method,direction',
    [
        (name, dir_)
        for name in ['and_', 'sup', 'gap_between']
        for dir_ in ['AB', 'BA']
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_binary_ops_to_RangeSeries(method, direction, binary_op_test_cases, as_ranges):
    range_A = as_ranges(binary_op_test_cases.A_start, binary_op_test_cases.A_end)
    range_B = as_ranges(binary_op_test_cases.B_start, binary_op_test_cases.B_end)
    expected = as_ranges(*binary_op_test_cases.get_coord_pair(method, direction))
    bound_method, target = binary_op_test_cases.get_binary_op_from_objs(method, range_A, range_B, direction)
    result = bound_method(target)
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'method,direction',
    [
        (name, dir_)
        for name in ['or_']
        for dir_ in ['AB', 'BA']
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_binary_ops_to_2_RangeSeries(method, direction, binary_op_test_cases, as_ranges):
    range_A = as_ranges(binary_op_test_cases.A_start, binary_op_test_cases.A_end)
    range_B = as_ranges(binary_op_test_cases.B_start, binary_op_test_cases.B_end)
    expected_left = as_ranges(*binary_op_test_cases.get_coord_pair(f'{method}_left', direction))
    expected_right = as_ranges(*binary_op_test_cases.get_coord_pair(f'{method}_right', direction))
    bound_method, target = binary_op_test_cases.get_binary_op_from_objs(method, range_A, range_B, direction)
    result_left, result_right = bound_method(target)
    assert_ranges_equal(result_left, expected_left)
    assert_ranges_equal(result_right, expected_right)


@pytest.mark.parametrize(
    'method,direction',
    [
        (name, dir_)
        for name in ['eq_',
                     'ne_',
                     'contains',
                     'is_contained_in',
                     'is_superset_of',
                     'is_subset_of',
                     'is_separated_from',
                     'intersects_elementwise',
                     'distance',
                     'offset',
                     'overlap'
                     ]
        for dir_ in ['AB', 'BA']
    ],
    scope='session', ids=make_pytest_id
)
def test__RangeSeries_binary_ops_to_scalar(method, direction, binary_op_test_cases, as_ranges):
    range_A = as_ranges(binary_op_test_cases.A_start, binary_op_test_cases.A_end)
    range_B = as_ranges(binary_op_test_cases.B_start, binary_op_test_cases.B_end)
    expected_value = binary_op_test_cases.get_scalar(method, direction)
    expected = seq_along(expected_value, range_A)
    bound_method, target = binary_op_test_cases.get_binary_op_from_objs(method, range_A, range_B, direction)
    result = bound_method(target)
    assert_series_equal(result, expected)


#######################################################################################
# 8. AGGREGATING OR VERTICAL OPERATIONS
#######################################################################################


@pytest.mark.parametrize(
    'start,end,expected',
    [
        ([], [], 0),
        ([10], [20], 1),
        ([10] * 3, [20] * 3, 3)
    ],
    scope='session', ids=['len_0', 'len_1', 'len_3']
)
def test__RangeSeries_count(start, end, expected):
    ranges = RangeSeries(start, end)
    assert ranges.count() == expected


@pytest.fixture(params=[
    (DataFrame({'start': [10, 50, 30, 20], 'end': [15, 55, 35, 25]}), 10, 55),
    (DataFrame({'start': [], 'end': []}), np.nan, np.nan)
], ids=['nonempty', 'empty'])
def ranges_test_cases(request):
    return request.param


def test__RangeSeries_min(ranges_test_cases):
    df, _min, _ = ranges_test_cases
    iv = RangeSeries(df)
    result = iv.min()
    np.testing.assert_equal(result, _min)


def test__RangeSeries_max(ranges_test_cases):
    df, _, _max = ranges_test_cases
    iv = RangeSeries(df)
    result = iv.max()
    np.testing.assert_equal(result, _max)


def test__RangeSeries_range_tuple(ranges_test_cases):
    df, _min, _max = ranges_test_cases
    iv = RangeSeries(df)
    expected = (_min, _max)
    result = iv.range_tuple()
    assert result == expected


def test__RangeSeries_range(ranges_test_cases):
    df, _min, _max = ranges_test_cases
    iv = RangeSeries(df)
    expected = RangeSeries(_min, _max) if type(_min) == int else RangeSeries(0, 0)
    result = iv.range()
    assert_ranges_equal(result, expected)


@pytest.mark.parametrize(
    'data,kwargs',
    [
        param(
            DataFrame({'start': [10, 50, 30, 20], 'end': [15, 55, 35, 25]}, index=pd.RangeIndex(4)),
            {}, id='by_start'
        ),
        param(
            DataFrame({'start': [10, 50, 30, 20], 'end': [15, 55, 35, 25]}, index=pd.RangeIndex(4)),
            {'ascending': False}, id='by_start_descending'
        ),
        param(
            DataFrame({'start': [10, 50, 30, 20], 'end': [90, 80, 70, 60]}, index=pd.RangeIndex(4)),
            {}, id='different_end_order'
        ),
        param(
            DataFrame({'start': [10, 50, 30, 20], 'end': [90, 80, 70, 60]}, index=pd.RangeIndex(4)),
            {'by': 'end'}, id='by_end'
        ),
        param(
            DataFrame({'start': [20, 10, 50, 30, 20], 'end': [25, 15, 55, 35, 25]}, index=pd.RangeIndex(5)),
            {}, id='stability'
        ),
        param(
            DataFrame({'start': [], 'end': []}, index=pd.RangeIndex(0), dtype=int),
            {}, id='empty_frame'
        ),
        param(
            DataFrame(
                {'start': [10, 50, 30, 20], 'end': [15, 55, 35, 25]},
                index=pd.CategoricalIndex(['a', 'a', 'b', 'b'], name='group')
            ),
            {'by': 'group'}, id='grouped_start'
        ),
    ],
    scope='session'
)
def test__RangeSeries_sort(data: pd.DataFrame, kwargs: Dict):
    iv = RangeSeries(data)
    by = kwargs.get('by', 'start')
    ascending = kwargs.get('ascending', True)
    expected_df = data.sort_values(by=by, ascending=ascending)
    result_df = iv.to_sorted_dataframe(**kwargs)
    assert_frame_equal(result_df, expected_df)
    expected_iv = iv(expected_df)
    result_iv = iv.sort(**kwargs)
    assert_ranges_equal(result_iv, expected_iv)


@dataclass
class VerticalSetOpsTestCase:
    name: str
    start: List
    end: List
    start_union: List
    end_union: List
    start_intersection: List
    end_intersection: List
    clusters: List
    is_disjoint: bool

    def get_ranges(self):
        return RangeSeries(self.start, self.end)

    def get_clusters(self):
        return Series(self.clusters, dtype=int)

    def get_union(self):
        return RangeSeries(self.start_union, self.end_union)

    def get_intersection(self):
        return RangeSeries(self.start_intersection, self.end_intersection)


_vertical_set_ops_test_cases = {
    tc.name: tc for tc in [
        VerticalSetOpsTestCase(
            'empty',
            [], [],
            [], [],
            [], [],
            [], True
        ),
        VerticalSetOpsTestCase(
            'disjoint',
            [10, 40, 50, 60],
            [30, 40, 60, 70],
            [10, 40, 50, 60],
            [30, 40, 60, 70],
            [], [],
            [0, 1, 2, 3], True
        ),
        VerticalSetOpsTestCase(
            'all_intersecting',
            [10, 10, 30],
            [50, 80, 70],
            [10], [80],
            [30], [50],
            [0, 0, 0], False
        ),
        VerticalSetOpsTestCase(
            'all_intersecting_with_empty_one',
            [10, 30, 40],
            [60, 80, 40],
            [10], [80],
            [40], [40],
            [0, 0, 0], False
        ),
        VerticalSetOpsTestCase(
            'partially_intersecting',
            [10, 20, 40],
            [30, 40, 60],
            [10, 40],
            [40, 60],
            [], [],
            [0, 0, 1], False
        )
    ]
}


@pytest.fixture(
    scope='session',
    params=_vertical_set_ops_test_cases.values(),
    ids=list(_vertical_set_ops_test_cases.keys())
)
def vertical_set_ops_cases(request):
    return request.param


def test__RangeSeries_is_disjoint(vertical_set_ops_cases):
    iv = vertical_set_ops_cases.get_ranges()
    result = iv.is_disjoint()
    assert result == vertical_set_ops_cases.is_disjoint


def test__RangeSeries_clusters(vertical_set_ops_cases):
    iv = vertical_set_ops_cases.get_ranges()
    expected = vertical_set_ops_cases.get_clusters()
    result = iv.clusters()
    assert_series_equal(result, expected)


def test__RangeSeries_union_self(vertical_set_ops_cases):
    iv = vertical_set_ops_cases.get_ranges()
    expected = vertical_set_ops_cases.get_union()
    result = iv.union_self()
    assert_ranges_equal(result, expected)


def test__RangeSeries_intersect_self(vertical_set_ops_cases):
    iv = vertical_set_ops_cases.get_ranges()
    expected = vertical_set_ops_cases.get_intersection()
    result = iv.intersect_self()
    assert_ranges_equal(result, expected)


#######################################################################################
# 9. SET-LIKE OPERATIONS (ALL RANGES IN THE SERIES TREATED AS A SET)
#######################################################################################


def test__RangeSeries_union_other():
    # test unsorted
    iv = RangeSeries(
        [20, 40, 50][::-1],
        [30, 60, 80][::-1],
        set_index=pd.Index(list('abc'))
    )
    other = RangeSeries(
        [30, 70][::-1],
        [40, 90][::-1],
        set_index=pd.Index(list('de'))
    )
    expected = RangeSeries(
        [20, 30, 40],
        [30, 40, 90]
    )
    result = iv.union_other(other)
    assert_ranges_equal(result, expected)


def test__RangeSeries_intersect_other():
    # test unsorted
    iv = RangeSeries(
        [20, 50, 60][::-1],
        [40, 50, 80][::-1],
        set_index=pd.Index(list('abc'))
    )
    other = RangeSeries(
        [10, 50, 70][::-1],
        [30, 50, 90][::-1],
        set_index=pd.Index(list('def'))
    )
    expected = RangeSeries(
        [20, 50, 70],
        [30, 50, 80]
    )
    result = iv.intersect_other(other)
    assert_ranges_equal(result, expected)


#######################################################################################
# 10. MISCELLANEOUS
#######################################################################################


# TODO: better tests?
def test_RangeSeries_groupby():
    iv = RangeSeries(
        [10, 20, 50, 60],
        [10, 40, 50, 80],
        set_index=pd.MultiIndex.from_product(
            [list('ab'), [6, 7]],
            names=['l1', 'l2']
        )
    )
    result = iv.groupby('l1')
    assert isinstance(result, RangeSeriesGroupBy)


def test_RangeSeries_groupby_empty():
    iv = RangeSeries(
        [], [],
        set_index=pd.MultiIndex.from_tuples([], names=['l1', 'l2'])
    )
    result = iv.groupby('l1')
    assert isinstance(result, RangeSeriesGroupBy)


def test_RangeSeries_assign_to(as_ranges):
    df = as_ranges(20, 30).to_frame()
    df = pd.concat([df, df], axis=1)
    df['idx'] = [2 * i + 1 for i in range(len(df))]
    df = df.set_index('idx')
    df.columns = ['start', 'end', 'foo_start', 'foo_end']
    df = df[['start', 'foo_end', 'end', 'foo_start']]

    expected = df.copy()
    expected['start'] = 10
    expected['end'] = 40
    modified_default_names = RangeSeries(df).expand(10)
    modified_default_names.assign_to(df)
    assert_frame_equal(df, expected)

    modified_foo = RangeSeries(df, prefix='foo_').expand(-1)
    expected = df.copy()
    expected['foo_start'] = 21
    expected['foo_end'] = 29
    modified_foo.assign_to(df)
    assert_frame_equal(df, expected)

