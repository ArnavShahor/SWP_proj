import math

def update_centroid(cluster):
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
    min_dist = float("inf")
    assignment = 0
    for i, cent in enumerate(centroids):
        curr_dist = eq_dist(point, cent)
        if curr_dist < min_dist:
            assignment = i
            min_dist = curr_dist
    return assignment


def eq_dist(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


def is_converged(centroids, new_centroids, EPS):
    for i in range(len(centroids)):
        if eq_dist(centroids[i], new_centroids[i]) >= EPS:
            return False
    return True

def k_means(K, iter, points):
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