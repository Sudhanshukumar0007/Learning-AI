# #GOAL Learn to implement the model  𝑓𝑤,𝑏
#   for linear regression with one variable
import numpy as np
import matplotlib.pyplot as plt
#x_train is the input variable(size in 1000 square feet)
#y_train is the target(price in 1000s dollors)
x_train=np.array([1.0,2.0])
y_train=np.array([300.0,500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# number of training examles m
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")
m=len(x_train)
print(f"Number of training examples is: {m}")
#x_i,y_i Training Examples
i=0
x_i=x_train[i]
y_i=y_train[i]
print(f"(x^({i}),y^({i})) = ({x_i} , {y_i})")
#plotting the data
#plot the data points
plt.scatter(x_train, y_train,marker="x",c='r')
#set the title
plt.title("Housing Prices")
#Set the y-axis Label
plt.ylabel('Price(in 1000s of dollars)')
#Set the x-axis Label
plt.xlabel('Size(1000 sqft)')
plt.show()
# W and b are the parameters let us start with a value of w=100 and b=100
w=100
b=100
print(f"w: {w}")
print(f"b: {b}")
# For a large number of data points, this can get unwieldy and repetitive. So instead, you can calculate the function output in a for loop as shown in the compute_model_output function below.

# Note: The argument description (ndarray (m,)) describes a Numpy n-dimensional array of shape (m,). (scalar) describes an argument without dimensions, just a magnitude.
# Note: np.zero(n) will return a one-dimensional numpy array with  𝑛
#   entries
#function of the liner regression is below
#F(x)=wx+b
def compute_model_output(x, w, b):
        """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
        m=x.shape[0]
        f_wb =np.zeros(m)
        for i in range(m):
                f_wb[i]=w * x[i] + b
        return f_wb
tmp_f_wb = compute_model_output(x_train,w,b,) 
#plot our Model prediction
plt.plot(x_train,tmp_f_wb,c='b',label='Our prediction')
#Plot the data points
plt.scatter(x_train,y_train,marker='x',c='r',label='Actual value')
#set the title
plt.title("Housing prices")
#Set the y axis label
plt.ylabel('Price (in 1000s of dollars )')
#Set the x axis
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()
#prediction
w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")