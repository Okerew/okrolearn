from okrolearn.okrolearn import Tensor
from okrolearn.tensor import pd
# Creating a Tensor from a pandas DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
tensor = Tensor.from_pandas(df)
print(tensor)
# Converting a Tensor back to a pandas DataFrame
df_from_tensor = tensor.to_pandas(columns=['A', 'B'])
print(df_from_tensor)