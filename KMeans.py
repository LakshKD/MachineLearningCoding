import numpy as np
import matplotlib.pyplot as plt

X = np.random.rand(100,2)

plt.scatter(X[:,0], X[:,1])

X = (X - np.mean(X, axis = 0))/ np.std(X, axis = 0) #Standardize the data

class KMeans:
     
    def __init__(self, k, max_iter = 100, random_state = None, verbose = False):
        self.X = None
        self.k = k
        self.num_examples = None
        self.num_features = None
        self.centroids = None
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.clusters = None
    
    
    def _L2Distance(self, X, centroid):
        
        return np.sqrt(np.sum((X-centroid)**2, axis=1))
    
    
    def _initialize(self):
        
        if self.random_state:
            np.random.seed(self.random_state)
        
        random_points = np.random.choice(self.num_examples, self.k, replace=False)
        
        return self.X[random_points]
    
    
    
    def _initializeWithK_Plus_Plus(self, X):
        
        
        self.X = X
        centroids = []
        centroids.append(self.X[np.random.randint(self.X.shape[0]), :])
        
        for c_id in range(self.k-1):
            
            distances  = np.zeros((self.X.shape[0], len(centroids)))
            min_distances = np.zeros((self.X.shape[0], 1))
            
            for i,c in enumerate(centroids):
                
                distances[:,i] = self._L2Distance(self.X, c)
            
            min_distances = np.min(distances, axis = 1)
            next_centroid = self.X[np.argmax(min_distances)]
            centroids.append(next_centroid)
            
        
        return centroids
            
    
    def fitWithBestK(self, X, max_iter = 100):
        
        self.X = self._standardize(X)
        self.num_examples, self.num_features = X.shape
        K = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        inertia_values = []
        centroidsList = []
        
        for cluster_count in K:
            
            self.k = cluster_count
            centroids = self._initialize()
            distances = np.zeros((self.num_examples, cluster_count))
            cluster_predictions = np.zeros((self.num_examples))


            for j in range(max_iter): #Training Loop 

                for i,c in enumerate(centroids):
                    distances[:, i] = self._L2Distance(self.X, c)


                cluster_predictions = np.argmin(distances, axis = 1)

                for i,c in enumerate(centroids):

                    centroids[i] = np.mean(X[cluster_predictions == i])
            
            
            #Calculate Inertia after training
            total_inertia = 0
            for i, c in enumerate(centroids):
                distances[:, i] = self._L2Distance(self.X, c)
            
            cluster_predictions = np.argmin(distances, axis = 1)
            
            for i, c in enumerate(centroids): 
                total_inertia += np.sum(self._L2Distance(self.X[cluster_predictions==i], c))
            
            inertia_values.append(total_inertia)
            centroidsList.append(centroids)
        

        return (K, inertia_values, centroidsList)
            
        
    def fit(self, X):
        
        self.X = X
        self.num_examples, self.num_features = X.shape
        
        
        #centroids = self._initialize()
        centroids = self._initializeWithK_Plus_Plus(X)
        distances = np.zeros((self.num_examples, self.k))
        cluster_predictions = np.zeros((self.num_examples))
        previous_cluster_predictions = np.zeros((self.num_examples))
        
        
        for iter_num in range(max_iter):
        
            for i,c in enumerate(centroids):
                distances[:, i] = self._L2Distance(self.X, c)


            cluster_predictions = np.argmin(distances, axis = 1)
            
            for i,c in enumerate(centroids):

                centroids[i] = np.mean(X[cluster_predictions == i])
            
            
            if np.all(previous_cluster_predictions == cluster_predictions):
                if self.verbose:
                    print("Finishing early in " + str(iter_num) + "steps")
                    break
            
            previous_cluster_predictions = cluster_predictions.copy()
        
        self.clusters = cluster_predictions
        self.centroids = centroids
        
        return self.centroids, self.clusters
    
    def predict(self, X_test):
        
        distances = np.zeros((X_test.shape[0], self.k))
        
        for i,c in enumerate(self.centroids):
            distances[:,i] = self._L2Distance(X_test, c)
        
        cluster_predictions = np.argmin(distances, axis=1)
        
        return cluster_predictions
    
    
    def sse(self, X):
        
        c_mean = np.mean(X, axis = 0)
        total_sse = np.sum(self._L2Distance(X,c_mean))
        
        sse_numerator = 0
        for i,c in enumerate(self.centroids):
            X_c = X[self.clusters == i]
            ss = np.sum(self._L2Distance(X_c, c))
            sse_numerator += ss
        
        return sse_numerator / total_sse



KMeansObj = KMeans(5, 1000, 42, True)
KMeansObj._initializeWithK_Plus_Plus(X)
centroids, clusters = KMeansObj.fit(X)

KMeansObj.predict(np.random.rand(20, 2))
K, inertia_values, centroidList = KMeansObj.fitWithBestK(X, 1000)

plt.scatter(K, inertia_values)
X_test = np.random.rand(100, 20)
predictions = KMeansObj.predict(X_test)
plt.scatter(X_test[:,1], predictions)