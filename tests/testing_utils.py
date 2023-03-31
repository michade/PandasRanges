import dataclasses
import itertools
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional, Hashable, Tuple, List, Generator

import pandas as pd
import pytest


@dataclass
class Vectorizer:
    size: int
    cls: Optional[Callable] = None
    index: Optional[pd.Index] = None
    name: Optional[Hashable] = None
    columns: Optional[Tuple[Hashable]] = None

    def __call__(self, *data, cls=None, **kwargs):
        if len(data) == 0:
            raise ValueError("Sppecify at least one argument to vectorize")
        if len(data) == 1:
            data = data[0]
        if cls is None:
            if self.cls is not None:
                cls = self.cls
            elif hasattr(data, '__len__'):
                cls = pd.DataFrame
            else:
                cls = pd.Series
        v = cls([data] * self.size)
        to_set = dataclasses.asdict(self)
        to_set.update(kwargs)
        for name in ('size', 'cls'):
            del to_set[name]
        for name, value in to_set.items():
            if value is None:
                continue
            if not hasattr(v, name):
                raise AttributeError(f'Object of type {type(v)} appears to have no attribute {name}')
            setattr(v, name, value)
        index = to_set.get('index', None)
        if index is not None and self.size != len(index):
            raise ValueError(f"Mismatch of index ({len(index)}) and data ({self.size}) length.")
        return v


class TestCasesFile:
    __slots__ = ['_param']
    path: Optional[str] = None
    n_cases: Optional[int] = None

    def __init__(self, param_):
        self._param = param_

    def __repr__(self):
        return repr(self._param)

    def __str__(self):
        return str(self._param)

    @classmethod
    def read_cases_file(cls, path: str) -> Tuple[List[Tuple[Any]], Optional[List[str]]]:
        raise NotImplementedError()

    @classmethod
    def params(cls) -> Generator[pytest.param, None, None]:
        if cls.n_cases is None:
            cases, comments = cls.read_cases_file(cls.path)
            cls.n_cases = len(cases)
        else:
            cases = None
            comments = None
        if comments is None:
            comments = itertools.repeat(None)
        for case, comment in zip(cases, comments):
            yield pytest.param(case, id=comment)

    @property
    def param(self):
        return self._param


class TestCasesCsv(TestCasesFile):
    columns = None

    @classmethod
    def read_cases_file(cls, path: str) -> Tuple[List[Any], Optional[List[str]]]:
        if not os.path.exists(path):  # TODO: fix this?
            path = os.path.join('tests', path)
        cases_df = pd.read_csv(path)
        cls.columns = list(cases_df.columns)
        if 'comment' in cases_df.columns:
            comments = [f'[L{idx + 2}:"{comment}"]' for idx, comment in cases_df.comment.iteritems()]
            cases_df = cases_df.drop(columns=['comment'])
        else:
            comments = None
        cases = list(cases_df.itertuples(index=False))
        return cases, comments

    def __getitem__(self, item: str):
        return getattr(self._param, item)

    def __getattr__(self, item: str):
        return getattr(self._param, item)

    def __contains__(self, item: str):
        return hasattr(self._param, item)


def make_pytest_id(args, names=None):
    if not hasattr(args, '__iter__'):
        return None
    if names is None:
        names = itertools.repeat('')
    if isinstance(args, str):
        return args
    return '[' + ', '.join([
        f'{name}{"=" if name else ""}{arg}'
        for name, arg in zip(names, args)
    ]) + ']'
