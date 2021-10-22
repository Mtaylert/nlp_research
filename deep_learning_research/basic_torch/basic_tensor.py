import torch

#===============================================#
#             Initializing Tensor               #
#===============================================#

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_tensor = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32,
                         device=device)

print(my_tensor)
print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)

#other common initialization methods
x = torch.empty(size = (3, 3))
