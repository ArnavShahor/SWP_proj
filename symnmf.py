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

def execute_goal(goal, data_list, k):
    """
    Execute the specified goal operation on the data.

    Args:
        goal: Goal operation ("sym", "ddg", "norm", or "symnmf")
        data_list: Input data as list of lists
        k: Number of clusters

    Returns:
        Result matrix from the specified operation

    Raises:
        Exception: If goal is invalid
    """
    if goal == "sym":
        return symnmfmodule.sym(data_list)
    elif goal == "ddg":
        return symnmfmodule.ddg(data_list)
    elif goal == "norm":
        return symnmfmodule.norm(data_list)
    elif goal == "symnmf":
        W = symnmfmodule.norm(data_list)
        H = init_H(W, k).tolist()
        return symnmfmodule.symnmf(H, W)
    else:
        raise Exception()

def main():
    """
    Main function to execute SymNMF algorithm based on command line arguments.

    Returns:
        None: Prints results to stdout or error message on failure
    """
    if len(sys.argv) != 4:
        raise Exception()

    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]

    data = pd.read_csv(filename, delimiter=',', header=None)
    n = data.shape[0]

    # Check inputs
    if not (1 < k < n) or goal not in {"sym", "ddg", "norm", "symnmf"} or filename == "":
        if not (k == 0 and goal != "symnmf") or filename == "":
            raise Exception()

    data_list = data.values.tolist()
    result = execute_goal(goal, data_list, k)
    format_output(result)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An Error Has Occurred")