from libc.math cimport sqrt
import numpy as np

DTYPE = np.int_

def KNN_algorithm(double[:,:] x_train, long long[:] class_train, double[:,:] x_test, Py_ssize_t k):
    cdef Py_ssize_t x_test_max = x_test.shape[0]
    cdef Py_ssize_t x_train_max = x_train.shape[0]
    cdef Py_ssize_t point_size = x_test.shape[1]
    cdef Py_ssize_t point_train_size = x_train.shape[1]

    assert point_size == point_train_size

    class_test_pred = np.zeros(x_test_max, dtype = DTYPE)
    cdef long long[:] class_test_pred_view = class_test_pred

    nearest_neighbor_ids = np.zeros(k, dtype = DTYPE)
    nearest_neighbor_labels = np.zeros(k, dtype = DTYPE)
    cdef Py_ssize_t[:] nearest_neighbor_ids_view = nearest_neighbor_ids
    cdef long long[:] nearest_neighbor_labels_view = nearest_neighbor_labels

    cdef Py_ssize_t i, j, d, p
    cdef double dist, diff

    distances = np.zeros(x_train_max)
    cdef double[:] distances_view = distances

    for i in range(x_test_max):
        test_point = x_test[i]
        for j in range(x_train_max):
            dist = 0
            for d in range(point_size):
                diff = x_train[j, d] - test_point[d]
                dist += diff * diff
            distances_view[j] = sqrt(dist)

        nearest_neighbor_ids_view = np.argpartition(distances_view, k)

        for p in range(k):
            nearest_neighbor_labels_view[p] = class_train[nearest_neighbor_ids_view[p]]
        class_test_pred_view[i] = np.bincount(nearest_neighbor_labels).argmax()

    return class_test_pred