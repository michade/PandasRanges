# distutils: language=c++
# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False

cimport cython
from typing import Tuple
from cython cimport Py_ssize_t
from cython.operator import dereference as deref
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.queue cimport priority_queue


# TODO: more types
ctypedef long coord_t


###############################################################
### Utilities
###############################################################


cdef class BinaryMinHeapOnVector():
    cdef vector[pair[coord_t, Py_ssize_t]] buf

    def __init__(self):
        self.buf = vector[pair[coord_t, Py_ssize_t]]()

    def reserve(self, Py_ssize_t n):
        self.buf.reserve(n)

    cdef push(self, coord_t coord, Py_ssize_t index):
        self.buf.push_back(pair[coord_t, Py_ssize_t](coord, index))
        # up-heap
        cdef Py_ssize_t i = self.buf.size() - 1
        cdef Py_ssize_t parent
        while i > 0:
            parent = (i - 1) // 2
            if self.buf[parent].first <= self.buf[i].first:
                break
            self.buf[parent], self.buf[i] = self.buf[i], self.buf[parent]
            i = parent

    cdef pop(self):
        if self.size() == 1:
            self.buf.pop_back()
            return
        self.buf[0] = self.buf.back()
        self.buf.pop_back()
        # down-heap
        cdef Py_ssize_t i = 0
        cdef Py_ssize_t left, right, smallest
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < self.buf.size() and self.buf[left].first < self.buf[smallest].first:
                smallest = left
            if right < self.buf.size() and self.buf[right].first < self.buf[smallest].first:
                smallest = right
            if smallest == i:
                break
            self.buf[i], self.buf[smallest] = self.buf[smallest], self.buf[i]
            i = smallest        

    cdef pair[coord_t, Py_ssize_t]* peek(self):
        return &self.buf.front()

    cdef bint empty(self):
        return self.buf.empty()

    cdef Py_ssize_t size(self):
        return self.buf.size()

    cdef pair[coord_t, Py_ssize_t]* get(self, Py_ssize_t i):
        return &self.buf[i]


cdef class IterableMinQueue():
    cdef BinaryMinHeapOnVector _heap
    
    def __init__(self, iterable = None):        
        self._heap = BinaryMinHeapOnVector()
        if iterable is not None:
            for item in iterable:
                self.push(item)
    
    def push(self, Tuple[coord_t, Py_ssize_t] item):
        cdef coord_t coord = item[0]
        cdef Py_ssize_t index = item[1]
        self._heap.push(coord, index)

    def top(self) -> Tuple[coord_t, Py_ssize_t]:
        if self.empty():
            raise IndexError("Top on empty queue.")
        cdef pair[coord_t, Py_ssize_t]* res = self._heap.peek()
        return (res.first, res.second)

    def get(self, Py_ssize_t i):
        if self._heap.size() <= i:
            raise IndexError("Index out of bounds.")
        cdef pair[coord_t, Py_ssize_t]* res = self._heap.get(i)
        return (res.first, res.second)
    
    def pop(self) -> Tuple[coord_t, Py_ssize_t]:
        item = self.top()
        self._heap.pop()
        return item

    def empty(self) -> bool:
        return self._heap.empty()

    def __len__(self) -> Py_ssize_t:
        return self._heap.size()

    def __iter__(self) -> Tuple[coord_t, Py_ssize_t]:
        cdef Py_ssize_t i
        cdef pair[coord_t, Py_ssize_t]* res
        for i in range(len(self)):
            res = self._heap.get(i)
            yield (res.first, res.second)

    def __str__(self) -> str:
        return str(list(self))


def get_first_unsorted_index_in_points(cython.integral[:] coord, bint ascending = True) -> int:    
    cdef Py_ssize_t i
    cdef Py_ssize_t n = coord.shape[0]
    if n < 2:
        return -1
    if ascending:
        for i in range(1, n):
            if coord[i] < coord[i - 1]:
                return i
    else:
        for i in range(1, n):
            if coord[i] > coord[i - 1]:
                return i
    return -1


def get_first_unsorted_index_in_ranges(cython.integral[:] start, cython.integral[:] end, bint ascending = True) -> int:
    cdef Py_ssize_t i    
    cdef Py_ssize_t n = start.shape[0]
    if n < 2:
        return -1
    cdef coord_t curr_start = start[1]
    cdef coord_t prev_start = start[0]
    if ascending:
        for i in range(1, n):
            curr_start = start[i]
            if curr_start < prev_start:
                return i
            elif curr_start == prev_start:
                if end[i] < end[i - 1]:
                    return i
            else:
                prev_start = curr_start
    else:
        for i in range(1, n):
            curr_start = start[i]
            if curr_start > prev_start:
                return i
            elif curr_start == prev_start:
                if end[i] > end[i - 1]:
                    return i
            else:
                prev_start = curr_start
    return -1


def get_first_invalid_range_index(cython.integral[:] start, cython.integral[:] end):
    cdef Py_ssize_t i
    for i in range(start.shape[0]):
        if start[i] > end[i]:
            return i
    return -1


###############################################################
### Overlaps
###############################################################


def overlapping_pairs_impl(
    Py_ssize_t[:,:] result,
    coord_t[:] start_a not None, coord_t[:] end_a not None,
    coord_t[:] start_b not None, coord_t[:] end_b not None,
    coord_t expand_start_a,
    coord_t expand_end_a
) -> Py_ssize_t:
    cdef bint only_count = result is None
    cdef Py_ssize_t n_overlaps = 0  # doubles as next position in result!

    # "stream" = "list of ranges"                 
    cdef Py_ssize_t len_a = start_a.shape[0]
    cdef Py_ssize_t len_b = start_b.shape[0]    

    if len_a == 0 or len_b == 0:
        return 0  # no elements in one of the streams, so intersection is empty

    cdef coord_t SENTINEL = max(end_a[len_a - 1], end_b[len_b - 1]) + max(expand_end_a, 0) + 1 # sentinel is greater than any other value

    cdef BinaryMinHeapOnVector open_a = BinaryMinHeapOnVector()
    cdef BinaryMinHeapOnVector open_b = BinaryMinHeapOnVector()

    cdef Py_ssize_t idx_a = 0
    cdef Py_ssize_t idx_b = 0
    cdef coord_t next_end_a = SENTINEL
    cdef coord_t next_end_b = SENTINEL
    cdef coord_t next_start_a
    cdef coord_t next_start_b
    # skip empty regions
    while idx_a < len_a:
        next_start_a = start_a[idx_a] - expand_start_a
        if next_start_a < end_a[idx_a]:
            break
        idx_a += 1            
    while idx_b < len_b:
        next_start_b = start_b[idx_b]
        if next_start_b < end_b[idx_b]:
            break
        idx_b += 1

    if idx_a == len_a or idx_b == len_b:
        return 0  # no non-empty elements in one of the streams, so intersection is empty
        
    while idx_a < len_a or idx_b < len_b:
        if min(next_start_a, next_start_b) < min(next_end_a, next_end_b):  # "=" is important here            
            # next point is the start of a region -> we open that region
            # here there is no way that the streams have ended, bc sentinel is always greater than any other value
            # intersection found, perform the appropriate action
            if next_start_a <= next_start_b:
                if only_count:  # counting:
                    n_overlaps += open_b.size()
                else:  # getting indices
                    for i in range(open_b.size()):
                        result[n_overlaps, 0] = idx_a
                        result[n_overlaps, 1] = open_b.get(i).second
                        n_overlaps += 1
                # open region
                open_a.push(end_a[idx_a] + expand_end_a, idx_a)
                # advance stream and update next coords
                idx_a += 1
                next_start_a = SENTINEL
                while idx_a < len_a:  # skip empty regions
                    next_start_a = start_a[idx_a] - expand_start_a
                    if next_start_a < end_a[idx_a]:
                        break
                    idx_a += 1
                next_end_a = open_a.peek().first                
            else:
                if only_count:  # counting:
                    n_overlaps += open_a.size()
                else:  # getting indices
                    for i in range(open_a.size()):
                        result[n_overlaps, 0] = open_a.get(i).second
                        result[n_overlaps, 1] = idx_b
                        n_overlaps += 1
                # open region
                open_b.push(end_b[idx_b], idx_b)
                # advance stream and update next coords
                idx_b += 1
                next_start_b = SENTINEL
                while idx_b < len_b:  # skip empty regions
                    next_start_b = start_b[idx_b]
                    if next_start_b < end_b[idx_b]:
                        break
                    idx_b += 1
                next_end_b = open_b.peek().first
        else:  #  min(next_start_a, next_start_b) > min(next_end_a, next_end_b)
            # next point is the end of a region -> we close that region
            if next_end_a < next_end_b:  # <, so next_end_a cannot be a sentinel
                open_a.pop()
                next_end_a = SENTINEL if open_a.empty() else open_a.peek().first
            elif next_end_b != SENTINEL:  # next_end_b <= next_end_a, and next_end_b is not a sentinel
                open_b.pop()
                next_end_b = SENTINEL if open_b.empty() else open_b.peek().first
            # else: # a and b are in fact both sentinels

    return n_overlaps


def overlapping_clusters_impl(
    Py_ssize_t[:] result,
    coord_t[:] start not None,
    coord_t[:] end not None,
    coord_t expand_start,
    coord_t expand_end,
    bint is_ascending
) -> Py_ssize_t:
    cdef bint only_count = result is None    
    cdef Py_ssize_t n_ranges = start.shape[0]

    if n_ranges == 0:
        return 0

    cdef Py_ssize_t idx    
    cdef Py_ssize_t d_idx
    cdef Py_ssize_t stop_idx
    if is_ascending:
        idx = 0
        d_idx = 1
        stop_idx = n_ranges
    else:
        idx = n_ranges - 1
        d_idx = -1
        stop_idx = -1

    cdef coord_t SENTINEL = end[stop_idx - d_idx] + max(expand_end, 0) + 1  # sentinel is greater than any other value
    cdef Py_ssize_t n_clusters = 0
    cdef priority_queue[coord_t] open_ranges = priority_queue[coord_t]()    
    cdef coord_t next_start = start[idx]
    cdef coord_t next_end = SENTINEL
        
    while idx != stop_idx:
        if next_start < next_end:  # "=" is important here            
            # next point is the start of a region -> we open that region
            # here there is no way that the stream has ended, bc sentinel is always greater than any other value
            if not only_count:
                result[idx] = n_clusters            
            open_ranges.push(-(end[idx] + expand_end))
            # advance stream and update next coords
            idx += d_idx
            next_start = start[idx] - expand_start if idx != stop_idx else SENTINEL
            next_end = -open_ranges.top()
        else:  # next_start > next_end
            # next point is the end of a region -> we close that region
            open_ranges.pop()
            if open_ranges.empty():
                n_clusters += 1
                next_end = SENTINEL
            else:
                next_end = -open_ranges.top()
                
    return n_ranges


def target_indices_to_permutation_impl(
    cython.integral[:] result not None,
    cython.integral[:] indices_a not None,
    cython.integral[:] indices_b not None
):
    cdef Py_ssize_t n_a = indices_a.shape[0]
    cdef Py_ssize_t i
    for i in range(n_a):
        result[indices_a[i]] = i
    for i in range(indices_b.shape[0]):
        result[indices_b[i]] = i + n_a


def mergesort_ranges_indices_impl(
    Py_ssize_t[:] indices_a not None, Py_ssize_t[:] indices_b not None,
    coord_t[:] start_a not None, coord_t[:] end_a not None,
    coord_t[:] start_b not None, coord_t[:] end_b not None
):
    cdef Py_ssize_t len_a = start_a.shape[0]
    cdef Py_ssize_t len_b = start_b.shape[0]
    cdef Py_ssize_t total_length = len_a + len_b

    cdef coord_t SENTINEL = max(end_a[len_a - 1], end_b[len_b - 1]) + 1 # sentinel is greater than any other value

    cdef Py_ssize_t idx_a = 0
    cdef Py_ssize_t idx_b = 0
    cdef Py_ssize_t idx_res = 0    
    while idx_res < total_length:
        s_a = start_a[idx_a] if idx_a < len_a else SENTINEL
        s_b = start_b[idx_b] if idx_b < len_b else SENTINEL
        if s_a < s_b:  # Take from A
            indices_a[idx_a] = idx_res
            idx_a += 1
        elif s_a > s_b:  # Take from B
            indices_b[idx_b] = idx_res
            idx_b += 1
        else:  # s_a == s_b
            e_a = end_a[idx_a] if idx_a < len_a else SENTINEL
            e_b = end_b[idx_b] if idx_b < len_b else SENTINEL
            if e_a <= e_b:  # Take from A
                indices_a[idx_a] = idx_res
                idx_a += 1
            else:  # Take from B
                indices_b[idx_b] = idx_res
                idx_b += 1
        idx_res += 1


def get_idx_of_cluster_starts_impl(cython.integral[:] result not None, cython.integral[:] cluster_id not None):    
    cdef Py_ssize_t n = cluster_id.shape[0]
    if n == 0:
        return    
    cdef Py_ssize_t curr_cluster = cluster_id[0]
    cdef Py_ssize_t n_clusters = 1
    result[0] = 0
    cdef Py_ssize_t i
    for i in range(1, n):
        if cluster_id[i] != curr_cluster:
            curr_cluster = cluster_id[i]
            result[n_clusters] = i
            n_clusters += 1