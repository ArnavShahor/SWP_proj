import sys
import numpy as np
import symnmfmodule
import pandas as pd

def init_H(W, k):
    """
    Initialize H matrix for SymNMF with random values bounded by theoretical upper limit.

    Args:
        W: Normalized similarity matrix as a 2D list
        k: Number of clusters

    Returns:
        numpy.ndarray: Initialized H matrix of shape (n, k) with random values
    """
    np.random.seed(1234)
    n = len(W)
    W_array = np.array(W)
    m = np.mean(W_array)
    upper_bound = 2 * np.sqrt(m / k)
    H = np.random.uniform(0, upper_bound, size=(n, k))
    return H

def format_output(matrix):
    """
    Format and print matrix with 4 decimal places, comma-separated values.

    Args:
        matrix: 2D list or array to be formatted and printed

    Returns:
        None: Prints formatted matrix to stdout
    """
    for row in matrix:
        formatted_row = ','.join(f"{val:.4f}" for val in row)
        print(formatted_row)

def main():
    """
    Main function to execute SymNMF algorithm based on command line arguments.

    Processes input data and performs one of four operations: sym, ddg, norm, or symnmf.
    Validates input parameters and handles exceptions.

    Args:
        sys.argv[1]: Number of clusters k (integer)
        sys.argv[2]: Goal operation ("sym", "ddg", "norm", or "symnmf")
        sys.argv[3]: Input CSV filename

    Returns:
        None: Prints results to stdout or error message on failure
    """
    if len(sys.argv) != 4:
        raise Exception()

    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]

    # Parse input data
    data = pd.read_csv(filename, delimiter=',', header=None)
    n = data.shape[0]

    # Check inputs
    if not (1 < k < n) or goal not in {"sym", "ddg", "norm", "symnmf"} or filename == "":
        if not (k == 0 and goal != "symnmf") or filename == "":
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
        H = init_H(W, k).tolist()
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