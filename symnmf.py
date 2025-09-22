import sys
import numpy as np
import symnmfmodule
import pandas as pd

def init_H(W, k):
    """Initialize H matrix with random values."""
    print(f"DEBUG: init_H called with k={k}")
    np.random.seed(1234)
    n = len(W)
    W_array = np.array(W)
    m = np.mean(W_array)
    upper_bound = 2 * np.sqrt(m / k)
    print(f"DEBUG: init_H - n={n}, mean(W)={m:.6f}, upper_bound={upper_bound:.6f}")
    H = np.random.uniform(0, upper_bound, size=(n, k))
    print(f"DEBUG: init_H - Generated H with shape {H.shape}")
    return H

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
    
    print(f"DEBUG: Starting symnmf with k={k}, goal={goal}, filename={filename}")

    # Parse input data
    data = pd.read_csv(filename, delimiter=',', header=None)
    n = data.shape[0]
    print(f"DEBUG: Loaded data with shape {data.shape} (n={n} datapoints)")

    # Check inputs
    if not (1 < k < n) or goal not in {"sym", "ddg", "norm", "symnmf"} or filename == "":
        if not (k == 0 and goal != "symnmf") or filename == "":
            raise Exception()

    # Convert to list for C module
    data_list = data.values.tolist()
    print(f"DEBUG: Converted data to list format")

    if goal == "sym":
        print("DEBUG: Computing similarity matrix")
        result = symnmfmodule.sym(data_list)
    elif goal == "ddg":
        print("DEBUG: Computing diagonal degree matrix")
        result = symnmfmodule.ddg(data_list)
    elif goal == "norm":
        print("DEBUG: Computing normalized similarity matrix")
        result = symnmfmodule.norm(data_list)
    elif goal == "symnmf":
        print("DEBUG: Starting SymNMF algorithm")
        # First compute W (normalized similarity matrix)
        print("DEBUG: Step 1 - Computing normalized similarity matrix W")
        W = symnmfmodule.norm(data_list)
        W_array = np.array(W)
        print(f"DEBUG: W matrix shape: {W_array.shape}")
        print(f"DEBUG: W matrix stats - min: {W_array.min():.6f}, max: {W_array.max():.6f}, mean: {W_array.mean():.6f}")
        
        # Initialize H
        print("DEBUG: Step 2 - Initializing H matrix")
        H = init_H(W, k)
        H_array = np.array(H)
        print(f"DEBUG: H matrix shape: {H_array.shape}")
        print(f"DEBUG: H matrix stats - min: {H_array.min():.6f}, max: {H_array.max():.6f}, mean: {H_array.mean():.6f}")
        
        # Perform SymNMF
        print("DEBUG: Step 3 - Running SymNMF algorithm")
        result = symnmfmodule.symnmf(H, W)
        result_array = np.array(result)
        print(f"DEBUG: Final result shape: {result_array.shape}")
        print(f"DEBUG: Final result stats - min: {result_array.min():.6f}, max: {result_array.max():.6f}, mean: {result_array.mean():.6f}")
    else:
        raise Exception()

    print("DEBUG: Formatting and printing final output")
    # Format and print output
    format_output(result)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An Error Has Occurred")