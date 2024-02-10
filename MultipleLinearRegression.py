import numpy as np

class MultipleLinearRegression:
    def __init__(self, init_w, init_b):
        self.w = np.array(init_w)
        self.b = init_b
        self.costs = []

    def fit(self, X, y, lr, epochs):
        m = X.shape[0]
        for i in range(epochs):
            dw, db = self.gradient_descent(X, y, self.w, self.b)
            cost = self.cost_function(X, y, self.w, self.b)
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
            self.costs.append(cost)
            if i % 100 == 0:
                print(f' {i} Cost: {cost:.6f} | w: {self.w} | b: {self.b}')

    def vectorized_fit(self, X, y, lr, epochs):
        m = X.shape[0]
        for i in range(epochs):
            dw, db = self.vectorized_gradient_descent(X, y, self.w, self.b)
            cost = self.cost_function(X, y, self.w, self.b)
            self.w = self.w - lr * dw
            self.b = self.b - lr * db
            self.costs.append(cost)
            if i % 100 == 0:
                print(f' {i} Cost: {cost:.6f} | w: {self.w} | b: {self.b}')

    def vectorized_gradient_descent(self, X, y, w, b):
        m,n = X.shape
        err = np.dot(X, w) + b - y
        dw = np.dot(X.T, err) / m
        db = np.sum(err) / m
        return dw, db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def cost_function(self, X, y, w, b):
        m = X.shape[0]
        cost = 0
        for i in range(m):
            cost = cost + ((np.dot(X[i], w) + b) - y[i]) ** 2
        cost = cost / (2 * m )
        return cost
    
    def gradient_descent(self, X, y, w, b):
        m,n = X.shape
        dj_dw = np.zeros(n)
        dj_db = 0
        for i in range(m):
            err = (np.dot(X[i], w) + b) - y[i] 
            dj_db =  dj_db + err
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err * X[i][j]

        dw = dj_dw / m
        db = dj_db / m
        return dw, db
