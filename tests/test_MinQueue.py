from PandasRanges.utils import MinQueue

import pytest


@pytest.fixture(params=[
    [0],
    [20],
    [20, 30, 10, 50, 30],
    [''],
    ['BB'],
    ['BB', 'CC', 'AA', 'DD', 'CC'],
    [tuple()],
    [(1, 20)],
    [(1, 20), (1, 30), (1, 10), (1, 50), (1, 30)]
], scope='module', ids=lambda l: f'{type(l[0]).__name__}_{len(l)}')
def data_for_MinQueue(request):
    return list(enumerate(request.param))


@pytest.fixture(scope='module')
def smallest_element_in_MinQueue_data():
    def _getmin(data):
        return min(data, key=lambda i_v: i_v[1])

    return _getmin


def test_MinQueue_init(data_for_MinQueue):
    q = MinQueue(data_for_MinQueue)
    assert sorted(list(q)) == sorted(data_for_MinQueue)


def test_MinQueue_init_empty():
    q = MinQueue([])
    assert list(q) == []
    assert len(q) == 0
    q = MinQueue()
    assert list(q) == []
    assert len(q) == 0


def test_MinQueue_len(data_for_MinQueue):
    q = MinQueue(data_for_MinQueue)
    assert len(q) == len(data_for_MinQueue)


def test_MinQueue_min(data_for_MinQueue, smallest_element_in_MinQueue_data):
    q = MinQueue(data_for_MinQueue)
    key, val = smallest_element_in_MinQueue_data(data_for_MinQueue)
    assert q.min() == val


def test_MinQueue_min_default():
    q = MinQueue()
    default_val = 123
    assert q.min(default_val) == default_val
    assert q.min(default=default_val) == default_val


def test_MinQueue_min_default_notused():
    val = 456
    q = MinQueue([(0, val)])
    default_val = 123
    assert q.min(default_val) == val
    assert q.min(default=default_val) == val


def test_MinQueue_min_empty(data_for_MinQueue):
    q = MinQueue()
    with pytest.raises(IndexError):
        q.min()


def test_MinQueue_first(data_for_MinQueue, smallest_element_in_MinQueue_data):
    q = MinQueue(data_for_MinQueue)
    expected = smallest_element_in_MinQueue_data(data_for_MinQueue)
    assert q.first() == expected


def test_MinQueue_first_empty(data_for_MinQueue):
    q = MinQueue()
    with pytest.raises(IndexError):
        q.first()


def test_MinQueue_pop(data_for_MinQueue):
    q = MinQueue(data_for_MinQueue)
    for i in range(len(data_for_MinQueue)):
        res = q.pop()
        assert res in data_for_MinQueue
        assert res not in list(q)
        assert len(q) == len(data_for_MinQueue) - i - 1


def test_MinQueue_push(data_for_MinQueue):
    q = MinQueue()
    for k, v in data_for_MinQueue:
        q.push(k, v)
        assert (k, v) in list(q)
        assert len(q) == k + 1


def test_MinQueue_push_pop_sequence(data_for_MinQueue, smallest_element_in_MinQueue_data):
    q = MinQueue()
    if len(data_for_MinQueue) < 5:
        return
    for pushes, pops in [
        (data_for_MinQueue[::-1], len(data_for_MinQueue)),
        (data_for_MinQueue[:-1], 2),
        (data_for_MinQueue[2:], 1)
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
