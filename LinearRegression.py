import numpy as np 
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.random.rand(500,30), np.random.rand(50,30), np.random.rand(500), np.random.rand(50)

class LinearRegression:
    
    def __init__(self):
        self.weights = None
        self.num_examples = None
        self.num_features = None
        self.X = None
        self.y = None
    
    
    def _standardizeData(self, X):
        
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/ np.std(X[:,i])
        
        return X
    
    def _computeCost(self):
    
        return np.dot((self.y - np.dot(self.X, self.weights.T)).T, (self.y - np.dot(self.X, self.weights.T)))/(2*self.num_examples)
    
    def _computeGradients(self):
        
        return np.dot((np.dot(self.X, self.weights.T) - self.y).T, self.X)/(1*self.num_examples)
    
    
    def fit(self, X, y, lr = 0.001, max_iter = 100):
        
        self.X = self._standardizeData(X)
        
        self.y = y.reshape((self.X.shape[0], 1))
        
        self.num_examples, self.num_features = self.X.shape
        
        bias_array = np.ones((self.num_examples,1))
        
        self.X = np.concatenate((bias_array, self.X), axis = 1)
        
        print(self.X.shape)
        
        self.weights = np.random.randn(self.num_features)
        self.weights = np.concatenate((np.ones(1), self.weights))
        self.weights = self.weights.reshape((1, self.num_features+1))
        
        costs_list = []
        
        for i in range(max_iter):
            
            gradients = self._computeGradients()
            self.weights = self.weights - lr*gradients
            
            costs_list.append(self._computeCost()[0])
        
        
        return (costs_list, max_iter)

    def predict(self, X_test):
        
        X_test = self._standardizeData(X_test)
        bias_array = np.ones((self.num_examples,1))
        X_test = np.concatenate((bias_array, X_test), axis = 1)
        
        predictions = np.dot(X_test, self.weights.T)
        
        return predictions

LinearRegObj = LinearRegression()
cost_list, max_iter = LinearRegObj.fit(X_train, y_train, max_iter = 1000)

plt.plot(cost_list, range(max_iter))