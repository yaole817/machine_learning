import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from scipy.io import loadmat

data = loadmat('data/ex3data1.mat')


def sigmoid(z):
	return 1/(1+np.exp(-z))

def cost(theta,X,y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)
	first = np.multiply(-y,np.log(sigmoid(X*theta.T)))
	second = np.multiply((1-y),np.log(1-sigmoid(X*theta.T)))
	reg = (learningRate / 2 *len(X))*np.sum(np.power(theta[:,1:theta.shape[1]],2))
	return np.sum(first - second)/(len(X)) + reg

def gradient_with_loop(theta, X, y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	grad = np.zeros(parameters)

	error = sigmoid(X*theta.T) - y
	for i in range(parameters):
		term = np.multiply(error,X[:,j])
		if(i==0):
			grad[i] = np.sum(term)/len(X)
		else:
			grad[i] = (np.sum(term)/len(X))+ learningRate/len(X)*theta[:,i]
	return grad 


def gradient(theta,X,y,learningRate):
	theta = np.matrix(theta)
	X = np.matrix(X)
	y = np.matrix(y)

	parameters = int(theta.ravel().shape[1])
	error = sigmoid(X * theta.T) - y
	grad = ((X.T * error)/len(X)).T + (learningRate/len(X))*theta

	grad[0,0] = np.sum(np.multiply(error,X[:,0]))/len(X)

	return np.array(grad).ravel()

from scipy.optimize import minimize 

def one_vs_all(X,y,num_labels,learning_rate):
	rows = X.shape[0]
	params = X.shape[1]

	all_theta = np.zeros((num_labels,params+1))

	X = np.insert(X,0,values = np.ones(rows),axis=1)

	for i in range(1, num_labels +1):
		theta = np.zeros(params + 1)
		y_i = np.array([1 if label == i else 0 for label in y])
		y_i = np.reshape(y_i,(rows,1))

		fmin = minimize(fun = cost,x0 = theta,args=(X,y_i,learning_rate),method = 'TNC',jac= gradient)
		all_theta[i-1,:] = fmin.x
	return all_theta

rows = data['X'].shape[0]
params = data['X'].shape[1]
all_theta = np.zeros((10,params + 1))

X = np.insert(data['X'],0,values = np.ones(rows),axis = 1)
theta = np.zeros(params + 1)

y_0 = np.array([1 if label == 0 else 0 for label in data['y']])
y_0 = np.reshape(y_0,(rows,1))

all_theta = one_vs_all(data['X'],data['y'],10,1)

def predict_all(X,all_theta):
	rows = X.shape[0]
	params = X.shape[1]
	num_labels = all_theta.shape[0]

	X = np.insert(X,0,values = np.ones(rows),axis =1)
	X = np.matrix(X)
	all_theta = np.matrix(all_theta)

	h = sigmoid(X*all_theta.T)
	h_argmax = np.argmax(h,axis = 1)
	h_argmax = h_argmax + 1

	return h_argmax

y_pred = predict_all(data['X'],all_theta)
correct = [1 if a == b else 0 for (a,b) in zip(y_pred,data['y'])]
accuracy = (sum(map(int,correct))/float(len(correct)))
print("accuracy = {0}%".format(accuracy*100))
a