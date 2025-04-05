import numpy as np
import math


def load_data(filename):
    data = np.genfromtxt('shopping.csv', delimiter=',', skip_header=1, dtype=str)
    return data

def sigmoid(x):
    clipped_x = np.clip(x, -500, 500)
    #return (1.0 / (1.0 + (math.e ** -z)))
    return 1 / (1 + np.exp(-clipped_x))

def compute_cost(X, y, w, b):
    m = len(y)
    z = np.dot(X, w) + b
    predict_val = sigmoid(z)
    epsilon = 1e-10
    cost_val = -y * np.log(predict_val + epsilon) - (1.0 - y) * np.log(1.0 - predict_val + epsilon)
    total_cost = (1.0 / m) * np.sum(cost_val)
    return total_cost



def compute_gradient(X, y, w, b):
    m = len(y)
    z_wb = np.dot(X, w) + b
    f_wb = sigmoid(z_wb)
    
    dj_db = np.sum(f_wb - y) / m
    dj_dw = np.dot(X.T, (f_wb - y)) / m
    
    return dj_db, dj_dw


def predict(X, w, b):
    z = np.dot(X, w) + b
    return sigmoid(z)

#Regularization functions
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    m, n = X.shape

    cost_without_reg = compute_cost(X, y, w, b) 

    reg_cost = 0.

    for i in range (n):
        reg_cost += w[i] ** 2
    reg_cost *= (lambda_/(2*m))

    total_cost = cost_without_reg + reg_cost

    return total_cost


def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)
 
    regularization = (lambda_/m) * w
    
    dj_dw += regularization

    return dj_db, dj_dw