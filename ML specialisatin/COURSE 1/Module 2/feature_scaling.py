#Featre scaling and learning rate (Multi-Variable)
import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import load_house_data,run_gradient_descent
from lab_utils_multi import norm_plot,plt_equal_scale,plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

#load the dataset
x_train,y_train=load_house_data()
x_features=['size(sqft)','bedrooms','floors','age']

#let us seee the data set and its features by plotting each feature vs price
fig,ax=plt.subplots(1,4, figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(x_train[:,1],y_train)
    ax[i].set_xlabel(x_features[i])
ax[0].set_ylabel("Price (1000's)")
plt.show()