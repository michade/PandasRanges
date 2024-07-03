from PandasRanges.overlaps import *

import pytest


@pytest.fixture(params=[
    [(1, 20)],
    [(1, 20), (1, 30), (1, 10), (1, 50), (1, 30)]
], scope='module', ids=lambda l: f'{type(l[0]).__name__}_{len(l)}')
def mq_data(request):
    return list(enumerate(request.param))


@pytest.fixture(scope='module')
def smallest_element_in_MinQueue_data():
    def _getmin(data):
        return min(data, key=lambda i_v: i_v[1])

    return _getmin


def test_MinQueue_new():
    q = minqueue_new()
    assert len(q) == 0


def test_MinQueue_init_empty():
    q = minqueue_init([])
    assert len(q) == 0

def test_minqueue_init(mq_data):
    q = minqueue_init(mq_data)
    assert all(sorted(list(q)) == sorted(mq_data))


def test_MinQueue_len(minqueue):
    q = minqueue_init(mq_data)
    assert len(q) == len(minqueue)


def test_MinQueue_min(minqueue, smallest_element_in_MinQueue_data):
    q = minqueue_init(mq_data)
    key, val = smallest_element_in_MinQueue_data(minqueue)
    assert q.min() == val


def test_MinQueue_min_default():
    q = minqueue_new()
    default_val = 123
    assert q.min(default_val) == default_val
    assert q.min(default=default_val) == default_val


def test_MinQueue_min_default_notused():
    val = 456
    q = minqueue_init([(0, val)])
    default_val = 123
    assert q.min(default_val) == val
    assert q.min(default=default_val) == val


def test_MinQueue_min_empty(mq_data):
    q = minqueue_init()
    with pytest.raises(IndexError):
        q.min()


def test_MinQueue_first(mq_data, smallest_element_in_MinQueue_data):
    q = minqueue_init(mq_data)
    expected = smallest_element_in_MinQueue_data(minqueue)
    assert q.first() == expected


def test_MinQueue_first_empty(mq_data):
    q = minqueue_new()
    with pytest.raises(IndexError):
        q.first()


def test_MinQueue_pop(mq_data):
    q = minqueue_init(mq_data)
    for i in range(len(mq_data)):
        res = q.pop()
        assert res in mq_data
        assert res not in list(q)
        assert len(q) == len(mq_data) - i - 1


def test_MinQueue_push(minqueue):
    q = minqueue_new()
    for k, v in minqueue:
        q.push(k, v)
        assert (k, v) in list(q)
        assert len(q) == k + 1


def test_MinQueue_push_pop_sequence(mq_data, smallest_element_in_MinQueue_data):
    q = minqueue_new()
    if len(mq_data) < 5:
        return
    for pushes, pops in [
        (mq_data[::-1], len(mq_data)),
        (mq_data[:-1], 2),
        (mq_data[2:], 1)
    ]:
        for k, v in pushes:
            q.push(k, v)
            data = list(q)
            expected = smallest_element_in_MinQueue_data(data)
            assert q.first() == expected
        for _ in range(pops):
            data = list(q)
            k, v = q.pop()
            expected = smallest_element_in_MinQueue_data(data)
            assert (k, v) == expected
