import math
import sys


def k_means(K, iter, points):
    try:
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

        for cent in centroids:
            if cent is not None:
                print(*tuple(f"{x:.4f}" for x in cent), sep=",")
        return 0

    except:
        print("An Error Has Occurred")
        return 1


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


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("An Error Has Occurred")
        return 1
    try:
        K = float(sys.argv[1])
        if int(K) != K:
            raise ValueError
        else:
            K = int(K)
    except ValueError:
        print("Incorrect number of clusters!")
        return 1
    try:
        iter = float(sys.argv[2]) if len(sys.argv) > 2 else 400
        if int(iter) != iter:
            raise ValueError
        else:
            iter = int(iter)
        if not 1 < iter < 1000:
            raise ValueError
    except ValueError:
        print("Incorrect maximum iteration!")
        return 1
    dim_error = False
    try:
        input_data = sys.stdin.read().split()
        points = list()
        dim = len(tuple(map(float, input_data[0].split(","))))
        for line in input_data:
            point = tuple(map(float, line.split(",")))
            if len(point) != dim:
                dim_error = True
                break
            points.append(point)
        N = len(points)
        if not 1 < K < N:
            print("Incorrect number of clusters!")
            return 1
        if dim_error:
            raise ValueError

        return k_means(K, iter, points)

    except Exception:
        print("An Error Has Occurred")
        return 1


if __name__ == "__main__":
    exit(main())
