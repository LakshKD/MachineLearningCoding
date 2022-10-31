import numpy as np

X = np.random.rand(105, 2)
y = np.random.randint(0,2, 105)

class KFold:
    
    def __init__(self, k=None, shuffle=False, seed=None):
        self.k = k
        self.shuffle = shuffle
        self.seed = seed
    
    
    def split(self, X):
        num_examples = X.shape[0]
        indices = np.array(range(num_examples))
        
        if self.seed:
            np.random.seed(self.seed)
            if self.shuffle:
                np.random.shuffle(indices)
        
        num_samples = (num_examples//self.k)*self.k #Will help us to get exact samples 
        indices_to_split, left_over_indices = indices[:num_samples], indices[num_samples:]
        
        all_splits = np.split(indices_to_split, self.k)
        
        all_splits_full = []
        
        for index in range(self.k):
            if index < len(left_over_indices):
                element = left_over_indices[index]
                temp = all_splits[index]
            else:
                temp = all_splits[index]
            
            all_splits_full.append(temp)
        
        
        for split in all_splits_full:
            mask = np.array([True if index in set(split) else False for index in indices])
            data = [indices[~mask], indices[mask]]
            yield data


k_fold = KFold(k=5, shuffle=True, seed= 42)

for train_indices, test_indices in k_fold.split(X):
    print(test_indices)
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = y[train_indices]
    Y_test = y[test_indices]
    
    print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)