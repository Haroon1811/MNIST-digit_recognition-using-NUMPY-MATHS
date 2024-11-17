# defining the functions for every preocess
import numpy as np
import pandas as pd

def init_params():
    W1 = np.random.rand(100,784) - 0.5
    b1 = np.random.rand(100,1) - 0.5
    W2 = np.random.rand(100,100) - 0.5
    b2 = np.random.rand(100,1) - 0.5
    W3 = np.random.rand(10,100) - 0.5
    b3 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2, W3, b3

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    return np.exp(Z)/sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

def ReLU_deriv(Z):
    return Z > 0                         # logic behind this is the derivative of linear(positive part of ReLU is postive) and derivative of negative part is 0(as a straight line)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))         # Y is labels from 0 to 9 so 9+1 =10 the total number of classes and teh number of columns ,Y.size =1 so shape is 1*10
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def back_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)                             # one hot the labels as to convert it into array of numbers
    """
     Trying L2 loss in this network
     L2 loss = 1/m sum(y_true - y_pred)**2
     derivative of the loss here is 2/m(y_true - y_pred)
      * did not work as L2 loss is mainly used for regression 
      * for classification we acn use Cross Entropy loss : -1/m(sum(sum(y_true*log(y_pred))))
        and its derivative is y_true - y_pred 
     """
    dZ3 = A3 - one_hot_Y
    dW3 = 1/m * (dZ3.dot(A2.T))
    db3 = 1/m * (np.sum(dZ3))
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1/m * (dZ2.dot(A1.T))
    db2 = 1/m * (np.sum(dZ2))
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1/m * (dZ1.dot(X.T))
    db1 = 1/m * (np.sum(dZ1))
    return dW1, db1, dW2, db2, dW3, db3

def update_params(dW1, db1, dW2, db2, dW3, db3, W1, b1, W2, b2, W3, b3, alpha):
    W1 = W1 - alpha*dW1
    b1 = b1 - alpha*db1
    W2 = W2 - alpha*dW2
    b2 = b2 - alpha*db2
    W3 = W3 - alpha*dW3
    b3 = b3 - alpha*db3
    return W1, b1, W2, b2, W3, b3

    
    
    

    
