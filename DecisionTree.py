import numpy as np 
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = np.random.rand(1000, 20), np.random.rand(500, 20), np.random.choice([0,1,2], 1000), np.random.choice([0,1,2], 500)

class Node:
    
    def __init__(self, feature = None, threshold=None, leftNode=None, rightNode=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.value = value
    
    #Non-leaf nodes has value as None
    def is_leaf(self): 
        
        return self.value is not None

class DecisionTree:
    
    def __init__(self, max_depth = None, min_samples_per_split = 2):
        self.X = None
        self.y = None
        self.max_depth = max_depth
        self.min_samples_per_split = min_samples_per_split
        self.root= None
    
    def is_finished(self, depth, samples, classes):
        if depth >= self.max_depth or samples <= self.min_samples_per_split or classes == 1:
            return True
        
        return False
    
    def _entropy(self, y):
        proportions = np.bincount(y)/len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
        
        return entropy
    
    def _informationGain(self, X, y, threshold):
        
        entropy_parent = self._entropy(y)
        left_idx, right_idx = np.argwhere(X <= threshold).flatten(), np.argwhere(X > threshold).flatten()
        n, left_size, right_size = len(y), len(left_idx), len(right_idx)
        
        if left_size == 0 or right_size == 0:
            return 0
        
        return entropy_parent - ((left_size/n)*self._entropy(y[left_idx]) + (right_size/n)*self._entropy(y[right_idx]))
    
    def _createSplit(self, X, threshold):
        
        left_idx, right_idx = np.argwhere(X<=threshold).flatten(), np.argwhere(X>threshold).flatten()
        
        return left_idx, right_idx
    
    def _bestSplit(self, X, y, features):
        
        score_dict = {"score":float("-inf"), "feature":None, "threshold":None}
        
        for f in features:
            thresholds = np.unique(X[:,f])
            for t in thresholds:
                score_temp = self._informationGain(X[:,f], y, t)
                
                if score_temp > score_dict["score"]:
                    score_dict["score"] = score_temp
                    score_dict["feature"] = f
                    score_dict["threshold"] = t
        
        
        return score_dict["feature"], score_dict["threshold"]
    
    def _buildTree(self, X, y, depth):
        
        num_samples, num_features = X.shape
        classes = len(np.unique(y))
        
        if self.is_finished(depth, num_samples, classes):
            most_common_value = np.argmax(np.bincount(y))
            return Node(value = most_common_value)
        
        
        rnd_features = np.random.choice(num_features, num_features, replace=False)
        best_feature, best_threshold = self._bestSplit(X, y, rnd_features)
        node = Node(feature = best_feature, threshold = best_threshold) #Non leaf Node
        left_idx, right_idx = self._createSplit(X[:,best_feature], best_threshold)
        
        node.leftNode = self._buildTree(X[left_idx,:], y[left_idx], depth+1)
        node.rightNode = self._buildTree(X[right_idx, :], y[right_idx], depth+1)
        
        return node
    
    
    def fit(self, X_train, y_train):
        
        self.root = self._buildTree(X_train, y_train, 0)
    
    
    def predict(self, X_test):
        
        root = self.root
        predictions = [self._traverse(root, x) for x in X_test]
        
        return np.array(predictions)
    
    def _traverse(self, root, x):
        
        if root.is_leaf():
            return root.value
        
        else:
            
            if x[root.feature]<=root.threshold:
                return self._traverse(root.leftNode, x)
            else:
                return self._traverse(root.rightNode, x)

DTObj = DecisionTree(max_depth = 30)

DTObj.fit(X_train, y_train)

y_pred = DTObj.predict(X_test)
len(y_pred[y_pred == y_test])/len(y_test)

