import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

X, y = np.random.rand(1000, 20), np.random.choice([0,1], 1000)

class KNN:
    
    def __init__(self, k):
        self.X = None
        self.y = None
        self.k = k
        self.num_examples = None
        self.num_features = None
    
    
    def _L2Distance(self, x1, x2):
        
        return np.sqrt(np.sum((x1-x2)**2, axis=1))
    
    
    def _standardizeData(self, X):
        
        for i in range(X.shape[1]):
            X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
        
        return X
    
    def fit(self, X, y):
        
        self.X = self._standardizeData(X)
        self.y = y
        self.num_examples, self.num_features = self.X.shape
        
    def predict(self, X_test):
        
        X_test = self._standardizeData(X_test)
        
        distances = np.zeros((self.X.shape[0], X_test.shape[0]))
        predictions = np.zeros(X_test.shape[0])
        
        for i, point in enumerate(X_test):
            distances[:,i] = self._L2Distance(self.X, point)
        
        for i in range(X_test.shape[0]):
            topK_closest_neig = np.argsort(distances[:,i])[0:self.k]
            topK_labels = self.y[topK_closest_neig]
            predictions[i] = Counter(topK_labels).most_common(1)[0][0]
        
        return predictions

KNNObj = KNN(10)
KNNObj.fit(X,y)

X_test = np.random.rand(100, 20)

KNNObj.predict(X_test)

