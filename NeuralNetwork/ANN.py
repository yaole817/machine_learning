import numpy as np
from sklearn import datasets,linear_model
import matplotlib.pyplot as plt

class Config:
	AnnInputDim = 2
	AnnOutputDim = 2
	epsilon = 0.01
	regLambda = 0.01


def generateData():
	np.random.seed(0)
	X,y = datasets.make_moons(200,noise= 0.2)
	# plt.scatter(X[:,0],X[:,1],s=40,c=y,cmap = plt.cm.Spectral)
	# plt.show()
	return X,y


def plotDecisionBoundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

def logicRegression(X,y):
	clf = linear_model.LogisticRegressionCV()
	clf.fit(X,y)
	plotDecisionBoundary(lambda x:clf.predict(x),X,y)


def calculateLoss(model,X,y):
	num_examples = len(X)
	W1,b1,W2,b2 = model['W1'],model['b1'],model['W2'], model['b2']

	z1 = X.dot(W1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(W2)+b2 
	exp_scores = np.exp(z2)
	probs = exp_scores / np.sum(exp_scores,axis = 1, keepdims = True)
	corectLogprobs = -np.log(probs[range(num_examples),y])
	data_loss = np.sum(corectLogprobs)

	data_loss += Config.regLambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))

	return 1./num_examples * data_loss


def buildModule(X,y,nn_hdim,num_passes=20000,print_loss=False):
	num_examples = len(X)
	np.random.seed(0)
	W1 = np.random.randn(Config.AnnInputDim,nn_hdim) / np.sqrt(Config.AnnInputDim)
	b1 = np.zeros((1,nn_hdim))
	W2 = np.random.randn(nn_hdim,Config.AnnOutputDim) / np.sqrt(nn_hdim)
	b2 = np.zeros((1,Config.AnnOutputDim))

	model = {}
	for i in range(0,num_passes):
		z1 = X.dot(W1) + b1
		a1 = np.tanh(z1)
		z2 = a1.dot(W2)+b2
		exp_scores = np.exp(z2)
		probs = exp_scores / np.sum(exp_scores,axis =1 ,keepdims = True)

		delta3 = probs
		delta3[range(num_examples),y] -= 1
		dW2 = (a1.T).dot(delta3)
		db2 = np.sum(delta3,axis=0,keepdims=True)
		delta2 = delta3.dot(W2.T)*(1 - np.power(a1,2))
		dW1 = np.dot(X.T,delta2)
		db1 = np.sum(delta2,axis=0)

		dW2 += Config.regLambda * W2
		dW1 += Config.regLambda * W1

		W1 += -Config.epsilon*dW1
		b1 += -Config.epsilon*db1
		W2 += -Config.epsilon*dW2
		b2 += -Config.epsilon*db2

		model = {'W1':W1, 'b1':b1, 'W2':W2,'b2':b2}

		if print_loss and i%1000 == 0:
			print ("loss after iteration %i : %f"%(i,calculateLoss(model,X,y)))

	return model



def main():
	X,y = generateData()
	#logicRegression(X,y)
	model = buildModule(X, y, 10, print_loss=True)
	for key in model:
		print key, model[key]
	plotDecisionBoundary(lambda x:predict(model,x), X, y)
	

if __name__ == '__main__':
	main()
