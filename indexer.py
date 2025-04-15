import numpy as np

k = 2
rows = int(2**k)

indices = np.arange(1,rows+1)

index_list = []

for _ in range(k):
    even_indices = indices[::2]
    odd_indices = indices[1::2]
    index_list.append(odd_indices)
    indices = even_indices

print(index_list)
