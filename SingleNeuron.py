import numpy as np

#Ste 1 : Define Input features ie X
#                 [x1 ,  x2 ,  x3]
input = np.array([2.0 , 3.0 , 4.0])
print("X : ",input)

#Step 2 : Define weights ie W
#                   [w1 ,  w2 ,  w3]
weights = np.array([0.5 , 0.3 , 0.2])
print("W : ",weights)

#Step 3 : Define Bias ie b 
#       b
bias = 1.0
print("b : ",bias)

#Step 4 : Calculate weighted sum ie Z
# z = x1w1 + x2w2 + x3w3 + b
# z = (2.0*0.5) +   + b

z = np.dot(input , weights) + bias
print("Z : ",z)

#Step 5 : Activetion function (ReLU)
def ReLU(X):
    return max(0,X)

#Step 6 : Final output (Y)
Y = ReLU(z)
print("Y : ",Y)

