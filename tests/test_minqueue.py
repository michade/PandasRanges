from PandasRanges.overlaps_cy import IterableMinQueue

import pytest


@pytest.fixture(params=[
    [(20, 1)],
    [(20, 1), (30, 2), (10, 3), (50, 4), (30, 5)]
], scope='module', ids=lambda l: f'{type(l[0]).__name__}_{len(l)}')
def minqueue_data(request):
    return list(request.param)


@pytest.fixture(scope='module')
def minqueue(minqueue_data):
    return IterableMinQueue(minqueue_data)


@pytest.fixture(scope='module')
def smallest_element_in_IterableMinQueue_data():
    def _getmin(data):
        return min(data, key=lambda i_v: i_v[0])

    return _getmin


def test_IterableMinQueue_init():
    minqueue = IterableMinQueue()
    assert len(minqueue) == 0


def test_IterableMinQueue_init_empty():
    minqueue = IterableMinQueue([])
    assert len(minqueue) == 0


def test_IterableMinQueue_iter(minqueue_data):
    minqueue = IterableMinQueue(minqueue_data)
    q_as_list = list(minqueue)
    assert sorted(q_as_list) == sorted(minqueue_data)


def test_IterableMinQueue_get_out_of_bounds(minqueue):
    with pytest.raises(IndexError):
        n = len(minqueue)
        minqueue.get(n)


def test_IterableMinQueue_len(minqueue_data):
    minqueue = IterableMinQueue(minqueue_data)
    return len(minqueue) == len(minqueue_data)


def test_IterableMinQueue_top(minqueue, smallest_element_in_IterableMinQueue_data):
    expected = smallest_element_in_IterableMinQueue_data(minqueue)
    assert minqueue.top() == expected


def test_IterableMinQueue_top_empty():
    minqueue = IterableMinQueue()
    with pytest.raises(IndexError):
        minqueue.top()


def test_IterableMinQueue_pop_empty():
    minqueue = IterableMinQueue()
    with pytest.raises(IndexError):
        minqueue.pop()


def test_IterableMinQueue_pop(minqueue_data):
    minqueue = IterableMinQueue(minqueue_data)
    for i in range(len(minqueue)):
        res = minqueue.pop()
        assert res in minqueue_data
        assert res not in list(minqueue)
        assert len(minqueue) == len(minqueue_data) - i - 1


def test_IterableMinQueue_push(minqueue):
    minqueue = IterableMinQueue()
    for k, v in minqueue:
        minqueue.push(k, v)
        assert (k, v) in list(minqueue)
        assert len(minqueue) == k + 1


def test_IterableMinQueue_push_pop_sequence(minqueue_data, smallest_element_in_IterableMinQueue_data):
    minqueue = IterableMinQueue()
    if len(minqueue_data) < 5:
        return
    for pushes, pops in [
        (minqueue_data[::-1], len(minqueue_data)),
        (minqueue_data[:-1], 2),
        (minqueue_data[2:], 1)
    ]:
        for k, v in pushes:
            minqueue.push((k, v))
            data = list(minqueue)
            expected = smallest_element_in_IterableMinQueue_data(data)
            assert minqueue.top() == expected
        for _ in range(pops):
            data = list(minqueue)
            k, v = minqueue.pop()
            expected = smallest_element_in_IterableMinQueue_data(data)
            assert (k, v) == expected
