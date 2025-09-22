import math

def update_centroid(cluster):
    """
    Calculate the centroid of a cluster by averaging coordinates of all points.

    Args:
        cluster: List of points in the cluster, where each point is a list of coordinates

    Returns:
        tuple: Centroid coordinates as a tuple, or None if cluster is empty
    """
    if not cluster:
        return None
    dim = len(cluster[0])
    k = len(cluster)
    centroid = [0] * dim
    for point in cluster:
        for i, coord in enumerate(point):
            centroid[i] += coord / k
    return tuple(centroid)


def assign_cluster(point, centroids):
    """
    Assign a point to the nearest centroid based on Euclidean distance.

    Args:
        point: List of coordinates representing a data point
        centroids: List of centroid coordinates

    Returns:
        int: Index of the nearest centroid
    """
    min_dist = float("inf")
    assignment = 0
    for i, cent in enumerate(centroids):
        curr_dist = eq_dist(point, cent)
        if curr_dist < min_dist:
            assignment = i
            min_dist = curr_dist
    return assignment


def eq_dist(p, q):
    """
    Calculate Euclidean distance between two points.

    Args:
        p: First point as list of coordinates
        q: Second point as list of coordinates

    Returns:
        float: Euclidean distance between the two points
    """
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def is_converged(centroids, new_centroids, EPS):
    """
    Check if centroids have converged by comparing distances to previous iteration.

    Args:
        centroids: Current centroids as list of coordinate tuples
        new_centroids: New centroids as list of coordinate tuples
        EPS: Convergence threshold

    Returns:
        bool: True if all centroids have moved less than EPS distance, False otherwise
    """
    for i in range(len(centroids)):
        if eq_dist(centroids[i], new_centroids[i]) >= EPS:
            return False
    return True

def k_means(K, iter, points):
    """
    Perform K-means clustering algorithm.

    Args:
        K: Number of clusters
        iter: Maximum number of iterations
        points: List of data points, where each point is a list of coordinates

    Returns:
        list: Final centroids after convergence or maximum iterations
    """
    EPS = 0.001
    centroids = points[:K]

    for i in range(iter):
        clusters = [list() for _ in range(K)]
        for point in points:
            clusters[assign_cluster(point, centroids)].append(point)
        new_centroids = [update_centroid(cluster) for cluster in clusters]
        for i in range(len(new_centroids)):
            if new_centroids[i] is None:
                new_centroids[i] = centroids[i]

        if is_converged(centroids, new_centroids, EPS):
            break
        centroids = new_centroids

    return centroids