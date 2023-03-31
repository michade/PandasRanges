from functools import partial
from typing import Dict

import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal, assert_frame_equal, assert_index_equal
from pytest import param

from PandasRanges.ranges import *
from .testing_utils import TestCasesCsv, Vectorizer, make_pytest_id

# _RangeSeries_min_max_test_cases = {
#     'ungrouped': (
#         DataFrame({
#             'start': [50, 10],
#             'end': [70, 30]
#         }),
#         None,
#         10, 70
#     ),
#     'one_level': (
#         DataFrame({
#             'g1': ['A', 'B', 'A', 'B'],
#             'start': [50, 20, 10, 40],
#             'end': [70, 20, 30, 50]
#         }).set_index(['g1']),
#         ['g1'],
#         pd.Series([10, 20], index=pd.Index(['A', 'B'], name='g1')),
#         pd.Series([70, 50], index=pd.Index(['A', 'B'], name='g1'))
#     ),
#     'two_level': (
#         DataFrame({
#             'g1': ['A', 'B', 'A', 'B', 'B'],
#             'g2': ['X', 'X', 'X', 'X', 'Y'],
#             'start': [50, 20, 10, 40, 5],
#             'end': [70, 20, 30, 50, 10]
#         }).set_index(['g1', 'g2']),
#         ['g1', 'g2'],
#         pd.Series([10, 20, 5],
#                   index=pd.MultiIndex.from_tuples([('A', 'X'), ('B', 'X'), ('B', 'Y')], names=['g1', 'g2'])),
#         pd.Series([70, 50, 10],
#                   index=pd.MultiIndex.from_tuples([('A', 'X'), ('B', 'X'), ('B', 'Y')], names=['g1', 'g2']))
#     ),
#     '1st_level_only': (
#         DataFrame({
#             'g1': ['A', 'B', 'A', 'B'],
#             'g2': ['X', 'X', 'Y', 'Y'],
#             'start': [50, 20, 10, 40],
#             'end': [70, 20, 30, 50]
#         }).set_index(['g1', 'g2']),
#         ['g1'],
#         pd.Series([10, 20], index=pd.Index(['A', 'B'], name='g1')),
#         pd.Series([70, 50], index=pd.Index(['A', 'B'], name='g1'))
#     ),
#     '2nd_level_only': (
#         DataFrame({
#             'g1': ['A', 'B', 'A', 'B'],
#             'g2': ['X', 'X', 'Y', 'Y'],
#             'start': [50, 20, 10, 40],
#             'end': [70, 20, 30, 50]
#         }).set_index(['g1', 'g2']),
#         ['g2'],
#         pd.Series([20, 10], index=pd.Index(['X', 'Y'], name='g2')),
#         pd.Series([70, 50], index=pd.Index(['X', 'Y'], name='g2'))
#     ),
# }


# def test__RangeSeries_max(RangeSeries_min_max_cases):
#     df, _, expected_max = RangeSeries_min_max_cases
#     iv = RangeSeries(df)
#     result = iv.max(by=groups)
#     if isinstance(expected_max, int):
#         assert result == expected_max
#     else:
#         assert_series_equal(result, expected_max, check_names=False)