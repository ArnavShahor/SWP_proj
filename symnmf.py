import sys
import numpy as np
import symnmfmodule
import pandas as pd

def init_H(W, k):
    """Initialize H matrix with random values."""
    np.random.seed(1234)
    n = len(W)
    m = np.mean(W)
    upper_bound = 2 * np.sqrt(m / k)
    H = np.random.uniform(0, upper_bound, size=(n, k))
    return H.tolist()

def format_output(matrix):
    """Format matrix output with 4 decimal places."""
    for row in matrix:
        formatted_row = ','.join(f"{val:.4f}" for val in row)
        print(formatted_row)

def main():
    if len(sys.argv) != 4:
       raise Exception()

    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]

    # Parse input data
    data = pd.read_csv(filename, delimiter=',', header=None)
    n = data.shape[0]

    # Check if k < n
    if not (1 < k < n):
        raise Exception()
    # Convert to list for C module
    data_list = data.values.tolist()
    if goal == "sym":
        result = symnmfmodule.sym(data_list)
    elif goal == "ddg":
        result = symnmfmodule.ddg(data_list)
    elif goal == "norm":
        result = symnmfmodule.norm(data_list)
    elif goal == "symnmf":
        # First compute W (normalized similarity matrix)
        W = symnmfmodule.norm(data_list)
        # Initialize H
        H = init_H(W, k)
        # Perform SymNMF
        result = symnmfmodule.symnmf(H, W)
    else:
        raise Exception()

    # Format and print output
    format_output(result)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An Error Has Occurred")