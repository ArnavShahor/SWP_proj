import math
import sys
import kmeans.py as kmeans
import pandas as pd
from sklearn import metrics
import symnmfmodule
import symnmf.py as symnmf
import numpy as np



def main():
    K = int(sys.argv[1])
    filename = sys.argv[2]

    data = pd.read_csv(filename, delimiter=',', header=None)
    DEFAULT_ITER = 400
    points = data.values.tolist()
    N = len(points)
    
    centroids = kmeans.k_means(K, DEFAULT_ITER, points)
    kmeans_labeling = [0 for _ in range(N)]
    for i in range(N):
        kmeans_labeling[i] = kmeans.assign_cluster(points[i], centroids)

    kmeans_silhouette_score = metrics.silhouette_score(points, kmeans_labeling, metric='euclidean')

    # running symnmf
    W = symnmfmodule.norm(points)
    H = symnmf.initialize_H(W, N, K)
    H = symnmfmodule.symnmf(H, W)

    # print("checking type")
    # print(type(H[0]))
    # print(type(H))
    # print(H)
    # print("checking type done")

    symnmf_clustering = np.argmax(np.array(H), axis=1)  # computing each datapoint's label
    symnmf_silhouette_score = metrics.silhouette_score(points, symnmf_clustering, metric='euclidean')

    print(f"nmf: %.4f" %symnmf_silhouette_score)
    print(f"kmeans: %.4f" %kmeans_silhouette_score)

    
if __name__ == "__main__":
    exit(main())