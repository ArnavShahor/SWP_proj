import sys
import numpy as np
import symnmfmodule
import pandas as pd

def init_H(W, k):
    """Initialize H matrix with random values."""
    print(f"DEBUG: init_H called with k={k}, W.shape={np.array(W).shape}", file=sys.stderr)
    np.random.seed(1234)
    n = len(W)
    m = np.mean(W)
    upper_bound = 2 * np.sqrt(m / k)
    H = np.random.uniform(0, upper_bound, size=(n, k))
    print(f"DEBUG: init_H completed, H.shape={H.shape}", file=sys.stderr)
    return H

def format_output(matrix):
    """Format matrix output with 4 decimal places."""
    print(f"DEBUG: format_output called with matrix shape={np.array(matrix).shape}", file=sys.stderr)
    for row in matrix:
        formatted_row = ','.join(f"{val:.4f}" for val in row)
        print(formatted_row)

def main():
    print(f"DEBUG: main() started with args: {sys.argv}", file=sys.stderr)

    if len(sys.argv) != 4:
        print("DEBUG: Wrong number of arguments", file=sys.stderr)
        raise Exception()

    k = int(sys.argv[1])
    goal = sys.argv[2]
    filename = sys.argv[3]
    print(f"DEBUG: Parsed args - k={k}, goal={goal}, filename={filename}", file=sys.stderr)

    # Parse input data
    try:
        print(f"DEBUG: Reading file {filename}", file=sys.stderr)
        data = pd.read_csv(filename, delimiter=',', header=None)
        n = data.shape[0]
        print(f"DEBUG: Data loaded successfully, shape={data.shape}", file=sys.stderr)
    except Exception as e:
        print(f"DEBUG: Error reading file: {e}", file=sys.stderr)
        raise Exception()

    # Check if k < n
    if not (1 < k < n):
        print(f"DEBUG: Invalid k value: k={k}, n={n}", file=sys.stderr)
        raise Exception()

    # Convert to list for C module
    print(f"DEBUG: Converting data to list", file=sys.stderr)
    data_list = data.values.tolist()
    print(f"DEBUG: Data converted, executing goal: {goal}", file=sys.stderr)

    if goal == "sym":
        print(f"DEBUG: Calling symnmfmodule.sym", file=sys.stderr)
        result = symnmfmodule.sym(data_list)
        print(f"DEBUG: symnmfmodule.sym completed", file=sys.stderr)
    elif goal == "ddg":
        print(f"DEBUG: Calling symnmfmodule.ddg", file=sys.stderr)
        result = symnmfmodule.ddg(data_list)
        print(f"DEBUG: symnmfmodule.ddg completed", file=sys.stderr)
    elif goal == "norm":
        print(f"DEBUG: Calling symnmfmodule.norm", file=sys.stderr)
        result = symnmfmodule.norm(data_list)
        print(f"DEBUG: symnmfmodule.norm completed", file=sys.stderr)
    elif goal == "symnmf":
        print(f"DEBUG: Starting symnmf goal", file=sys.stderr)
        # First compute W (normalized similarity matrix)
        print(f"DEBUG: Computing W matrix", file=sys.stderr)
        W = symnmfmodule.norm(data_list)
        print(f"DEBUG: W matrix computed", file=sys.stderr)
        # Initialize H
        print(f"DEBUG: Initializing H matrix", file=sys.stderr)
        H = init_H(W, k)
        print(f"DEBUG: H matrix initialized", file=sys.stderr)
        # Perform SymNMF
        print(f"DEBUG: Calling symnmfmodule.symnmf", file=sys.stderr)
        result = symnmfmodule.symnmf(H, W)
        print(f"DEBUG: symnmfmodule.symnmf completed", file=sys.stderr)
    else:
        print(f"DEBUG: Invalid goal: {goal}", file=sys.stderr)
        raise Exception()

    print(f"DEBUG: Formatting output", file=sys.stderr)
    # Format and print output
    format_output(result)
    print(f"DEBUG: Program completed successfully", file=sys.stderr)

if __name__ == "__main__":
    try:
        print(f"DEBUG: Starting program", file=sys.stderr)
        main()
    except Exception as e:
        print(f"DEBUG: Exception caught: {e}", file=sys.stderr)
        print("An Error Has Occurred")