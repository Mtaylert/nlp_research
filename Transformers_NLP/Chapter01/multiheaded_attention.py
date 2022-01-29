import numpy as np
from scipy.special import softmax

print("Step 1: Input : 3 inputs, d_model=4")
x =np.array([[1.0, 0.0, 1.0, 0.0], # Input 1
 [0.0, 2.0, 0.0, 2.0], # Input 2
 [1.0, 1.0, 1.0, 1.0]]) # Input 3
print("Step 2: weights 3 dimensions x d_model=4")
print("w_query")
w_query =np.array([[1, 0, 1],
 [1, 0, 0],
 [0, 0, 1],
 [0, 1, 1]])



print("w_key")
w_key =np.array([[0, 0, 1],
 [1, 1, 0],
 [0, 1, 0],
 [1, 1, 0]])


print("w_value")
w_value = np.array([[0, 2, 0],
 [0, 3, 0],
 [1, 0, 3],
 [1, 1, 0]])

Q=np.matmul(x,w_query)
K=np.matmul(x,w_key)
V=np.matmul(x,w_value)
k_d = 1
attention_scores = (Q @ K.transpose())/k_d
print(attention_scores)