import numpy as np
import matplotlib.pyplot as plt
import re

def parse_sparse_matrix(filename):
    entries = []
    max_row = max_col = 0

    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith("row"):
                continue
            row_match = re.match(r"row (\d+):", line)
            if not row_match:
                continue
            row = int(row_match.group(1))
            cols = re.findall(r"\((\d+),\s*([-+eE0-9.]+)\)", line)
            for col_str, val_str in cols:
                col = int(col_str)
                val = float(val_str)
                if abs(val) > 1e-14:  # Filter out near-zero entries
                    entries.append((row, col))
                    max_row = max(max_row, row)
                    max_col = max(max_col, col)

    mat = np.zeros((max_row + 1, max_col + 1), dtype=int)
    for r, c in entries:
        mat[r, c] = 1

    return mat

def save_sparsity_plot(mat, output_file="sparsity_pattern2.png"):
    plt.figure(figsize=(8, 6))
    plt.spy(mat, markersize=5)
    plt.title("Sparsity Pattern")
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Sparsity pattern saved to {output_file}")

if __name__ == "__main__":
    matrix_file = "matrix_output2.txt"
    matrix = parse_sparse_matrix(matrix_file)
    save_sparsity_plot(matrix)
