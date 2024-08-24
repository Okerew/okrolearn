from okrolearn.src.okrolearn.okrolearn import Tensor
# Create a tensor
t = Tensor([3, 1, 4, 1, 5, 9, 2, 6])

# Sort the tensor
sorted_t, indices = t.sort()
print("Sorted tensor:", sorted_t)
print("Sorting indices:", indices)

# Sort in descending order
sorted_t_desc, indices_desc = t.sort(descending=True)
print("Sorted tensor (descending):", sorted_t_desc)
print("Sorting indices (descending):", indices_desc)
