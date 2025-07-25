import math ,copy
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)  # reduced display precision on numpy arrays
x_train=np.array([[2104,5,1,45],[1416,3,2,40],[852,2,1,35]])
y_train=np.array([460,232,178])
# data is stored in numpy array/matrix
print(f"X Shape: {x_train.shape} ,X Type: {type(x_train)}")
print(x_train)
print(f"Y Shape: {y_train.shape}, Y Type: {type(y_train)}")
print(y_train)
b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")