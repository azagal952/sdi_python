from numpy import zeros, argsort, bincount
from numpy.linalg import norm

def KNN_algorithm(x_train, class_train, x_test, k):
    """Perform the K-nearest neighbours algorithm for classification

    Args:
        x_train (np.ndarray): matrix of coordinates of the train set
        class_train (np.ndarray): labels of points in train set
        x_test (np.ndarray): matrix of coordinates of the test set
        k (int): Number of nearest neighbours to consider

    Returns:
        np.ndarray: predicted labels for x_test
    """
    class_test_pred = zeros(x_test.shape[0], dtype = class_train.dtype)

    for i, test_point in enumerate(x_test):
        ## For each point of the test set, we compute the distance between each point of x_train and this test point.
        distances = norm(x_train - test_point, axis=1)

        ## We select indexes of the k points of x_train for which the distance with the point of x_test is minimal
        nearest_neighbor_ids = argsort(distances)[:k]

        nearest_neighbor_labels = class_train[nearest_neighbor_ids]
        class_test_pred[i] = bincount(nearest_neighbor_labels).argmax()

    return class_test_pred