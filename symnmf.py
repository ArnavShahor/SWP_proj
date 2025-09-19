import sys
import numpy as np
import symnmfmodule

def parse_data(filename):
    """Parse data from CSV file."""
    try:
        data = np.loadtxt(filename, delimiter=',')
        return data
    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

def initialize_H(W, n, k):
    """Initialize H matrix with random values."""
    np.random.seed(1234)
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
        print("An Error Has Occurred")
        sys.exit(1)

    try:
        k = int(sys.argv[1])
        goal = sys.argv[2]
        filename = sys.argv[3]
    except ValueError:
        print("An Error Has Occurred")
        sys.exit(1)

    # Parse input data
    data = parse_data(filename)
    n = data.shape[0]

    # Convert to list for C module
    data_list = data.tolist()

    try:
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
            H = initialize_H(W, n, k)
            # Perform SymNMF
            result = symnmfmodule.symnmf(H, W)
        else:
            print("An Error Has Occurred")
            sys.exit(1)

        # Format and print output
        format_output(result)

    except Exception:
        print("An Error Has Occurred")
        sys.exit(1)

if __name__ == "__main__":
    main()