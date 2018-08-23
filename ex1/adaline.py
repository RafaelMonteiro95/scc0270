import numpy as np

# this function helps opening the data files
def readFile(filename):
    with open(filename) as f:
        return np.array([int(num) for num in f.read().split()])


class Adaline:

    # threshold is used on activation function
    def Adaline(self, threshold):
        self.weights = none
        self.threshold = threshold
        
    def signal(self, net, threshold = 0.5):
        return ( 1 if net > threshold else -1 )
        
    # X is the training dataset (an array with arrays), Y is the training dataset labels (an array with -1 or 1)
    # this function trains the ADALINE
    def train(self, X, Y, eta = 0.5, min_error=10e-3, max_iter=1000):
        # Using X first instance to determine weights length 
        size = X[0].shape[0]
        # initializing weights with length (size + 1) to accomodate theta
        self.weights = np.random.rand(size + 1)
        # initializing error
        square_error = min_error * 2
        
        # trains while max_iter is not exceeded and error is above minimum
        counter = 0
        while counter < max_iter and square_error >= min_error:
            
            # for each instance in X with label Y:
            for instance, label in zip(X,Y):
                
                # inserting theta in instance
                xi = np.append(instance, [1])
                
                # calculating net function
                net = np.sum(xi * self.weights)
                
                # calculating an estimated label
                est = self.signal(net)
                
                # calculating error
                error = label - est
                square_error = square_error + (error ** 2)
                
                # calculating gradient
                grad = -2 * error * xi
                
                # adjusting weights
                self.weights = self.weights - eta * grad
                
            
            # adjusting current error
            square_error /= len(X)
            counter += 1
                
                
    
    # predicts if target belongs to class 1 or -1
    def predict(self, sample):
        xi = np.append(sample, 1)
        net = np.sum(xi * self.weights)
        return self.signal(net)