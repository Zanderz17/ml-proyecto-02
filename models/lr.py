import numpy as np


class SoftmaxRegression:
    def __init__(self):
        np.random.seed(2024)
        self.epsilon = 1e-8

    def h(self, x, w):
        return np.dot(x, w)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def loss(self, y, y_approx):
        y_approx = np.clip(y_approx, self.epsilon, 1 - self.epsilon)
        return -np.mean(np.sum(y * np.log(y_approx), axis=1))
    
    def derivatives(self, x, y, y_approx):
        return np.dot(x.T, (y_approx - y)) / x.shape[0]
    
    def change_parameters(self, w, derivatives, alpha):
        return w - alpha * derivatives
    
    def one_hot_encode(self, y, num_classes):
        y = y.astype(int)  # Ensure y contains integer values
        if np.any(y >= num_classes) or np.any(y < 0):
            raise ValueError("Labels in y are out of bounds for the number of classes")
        return np.eye(num_classes)[y]
    
    def training(self, x, y, epochs, alpha):
        num_classes = len(np.unique(y))
        y_encoded = self.one_hot_encode(y, num_classes)
        n_features = x.shape[1]
        w = np.random.rand(n_features, num_classes)
        loss_values = []
        
        for _ in range(epochs):
            z = self.h(x, w)
            y_approx = self.softmax(z)
            loss = self.loss(y_encoded, y_approx)
            dw = self.derivatives(x, y_encoded, y_approx)
            w = self.change_parameters(w, dw, alpha)
            loss_values.append(loss)
        
        return loss_values, w

    def predict(self, x, w):
        z = self.h(x, w)
        y_approx = self.softmax(z)
        return np.argmax(y_approx, axis=1)
