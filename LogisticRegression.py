import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.random.rand(1000, 30), np.random.rand(500,30), np.random.choice([0,1], 1000), np.random.choice([0,1], 500)

class LogisticRegression:
    
    def __init__(self):
        self.weights = None
        self.X = None
        self.y = None
        self.num_examples = None
        self.num_features = None
    
    def _sigmoid(self, Z):
        
        return 1/(1+np.exp(-Z))
    
    def _standardizeData(self, X):
        
        for i in range(X.shape[1]):
            
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
        
        return X
    
    def _computeCost(self):
        
        cost_0 = np.dot(self.y.T, np.log(self._sigmoid(np.dot(self.X, self.weights.T))))
        cost_1 = np.dot((1- self.y).T, np.log(1 - self._sigmoid(np.dot(self.X, self.weights.T))))
        
        return -(cost_0 + cost_1)/self.num_examples
    
    def _computeGradients(self):
        
        return -1 * np.dot((self.y - self._sigmoid(np.dot(self.X, self.weights.T))).T, self.X)/self.num_examples
    
    def fit(self, X_train, y_train, lr = 0.01, max_iter = 100):
        
        X_train = self._standardizeData(X_train)
        self.num_examples, self.num_features = X_train.shape
        bias_array = np.ones((self.num_examples, 1))
        self.X = np.concatenate((bias_array, X_train), axis = 1)
        self.y = y_train.reshape((self.num_examples, 1))
        
        self.weights = np.random.randn(self.num_features)
        self.weights = np.concatenate((np.ones(1), self.weights)).reshape((1, self.num_features+1))
        
        cost_list = []
        
        for i in range(max_iter):
            
            gradients = self._computeGradients()
            
            self.weights = self.weights - lr * gradients
            
            cost_list.append(self._computeCost()[0])
        
        return (cost_list, max_iter)
    
    def predict(self, X_test):
        
        X_test = self._standardizeData(X_test)
        bias_array = np.ones((X_test.shape[0], 1))
        X_test = np.concatenate((bias_array, X_test), axis = 1)
        
        predictions = self._sigmoid(np.dot(X_test, self.weights.T))
        
        predictions[predictions<=0.5] = 0
        predictions[predictions>0.5] = 1
        
        return predictions 

LogRegObj = LogisticRegression()
cost_list, max_iter = LogRegObj.fit(X_train, y_train, max_iter = 5000)

plt.plot(cost_list, range(max_iter))

y_pred = LogRegObj.predict(X_test)

def precision(y_pred, y_test):
    
    tp = 0
    fp = 0
    
    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i][0] == 1:
            fp += 1
        elif y_test[i] == 1 and y_pred[i][0] == 1:
            tp += 1
    
    
    return tp /(tp + fp)

precision(y_pred, y_test)