import numpy as np
import firedrake as fd

def load_vector_from_file(filename):
    values = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip "Process [x]" lines
            if line.startswith("Process"):
                continue
            try:
                values.append(float(line))
            except ValueError:
                pass  # Skip any line that can't be converted to float
    return np.array(values)

# Load both vectors
vec1 = load_vector_from_file("after_solve.txt")
vec2 = load_vector_from_file("after_solve copy.txt")

# Ensure both vectors have the same length
if vec1.shape != vec2.shape:
    print(f"Error: Vector lengths do not match. vec1: {vec1.shape}, vec2: {vec2.shape}")
else:
    # Compute L2 norm
    l2_norm = np.linalg.norm(vec1 - vec2)
    print(f"L2 norm between files: {l2_norm}")

def true_solution(x, y, t):
    return np.cos(np.pi * x) * np.cos(2 * np.pi * y) * np.exp(-5*np.pi*np.pi*t)

x = np.linspace(0, 1, 5)
y = np.linspace(0, 1, 5)
X, Y = np.meshgrid(x, y)
t = 0.001
true_sol = true_solution(X, Y, t)
print("True solution at t=0.001:")
print(true_sol)


