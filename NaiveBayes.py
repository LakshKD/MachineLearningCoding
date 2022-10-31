import numpy as np
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.random.rand(1000, 20), np.random.rand(500, 20), np.random.choice([0,1], 1000), np.random.choice([0,1], 500)

class NB:
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self._mean = None
        self._var = None
        self.priors = None
    
    
    def fit(self):
        
        self._mean = np.zeros((self.n_classes, self.X.shape[1]), dtype=np.float64)
        self._var = np.zeros((self.n_classes, self.X.shape[1]), dtype=np.float64)
        self._priors = np.zeros(self.n_classes, dtype=np.float64)
        
        for c in self.classes:
            X_c = self.X[self.y==c]
            
            self._mean[c,:] = np.mean(X_c)
            self._var[c,:] = np.var(X_c)
            
            self._priors[c] = len(X_c)/self.X.shape[0]
        
    
    def predict(self, X_test):
        
        predictions = [self._predict(x) for x in X_test]
        
        return np.array(predictions)
        
    def _predict(self, x):
        
        posterior = []
        
        for i, c in enumerate(self.classes):
            class_prior = np.log(self._priors[i])
            class_conditional = np.sum(np.log(self._pdf(x, c)))
            class_posterior = class_conditional + class_prior
            posterior.append(class_posterior)
        
        arg_max_class = np.argmax(posterior)
        return self.classes[arg_max_class]
    
    def _pdf(self, x, c):
        
        mean = self._mean[c,:]
        var = self._var[c,:]
        
        class_conditional_prob = (np.exp((-1*(x-mean)**2)/(2*var)))/np.sqrt(2*np.pi*var)
        
        return class_conditional_prob

NBObj = NB(X_train, y_train)
NBObj.fit()

y_pred = NBObj.predict(X_test)

len(y_pred[y_pred == y_test]) / len(y_test)