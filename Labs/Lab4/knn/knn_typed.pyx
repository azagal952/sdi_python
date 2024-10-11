from libc.math cimport sqrt
import numpy as np

TYPE1 = np.float64
DTYPE = np.int64

def KNN_algorithm(x_train, class_train, x_test, int k):
    cdef Py_ssize_t x_test_max = x_test.shape[0]
    cdef Py_ssize_t x_train_max = x_train.shape[0]
    cdef Py_ssize_t point_size = x_test.shape[1]
    cdef Py_ssize_t point_train_size = x_train.shape[1]

    assert point_size == point_train_size
    assert x_train.dtype == TYPE1
    assert x_test.dtype == TYPE1
    assert class_train.dtype == DTYPE

    class_test_pred = np.zeros(x_test_max, dtype = DTYPE)

    nearest_neighbor_ids = np.zeros(k)
    nearest_neighbor_labels = np.zeros(k)

    cdef Py_ssize_t i, j, d
    cdef double dist, diff
    distances = np.zeros(x_train_max)

    for i in range(x_test_max):
        test_point = x_test[i]
        for j in range(x_train_max):
            dist = 0
            for d in range(point_size):
                diff = x_train[j, d] - test_point[d]
                dist += diff * diff
            distances[j] = sqrt(dist)

        nearest_neighbor_ids = np.argsort(distances)[:k]

        nearest_neighbor_labels = class_train[nearest_neighbor_ids]
        class_test_pred[i] = np.bincount(nearest_neighbor_labels).argmax()

    return class_test_pred