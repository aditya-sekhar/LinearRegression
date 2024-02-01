class SimpleLinearRegression:

    def __init__(self, init_w, init_b):
        self.w = init_w
        self.b = init_b
    
    def fit(self, X, y, lr, epochs):
        costs = []
        for i in range(epochs):
            dldw, dldb = self.gradient_descent(X, y, lr)
            self.w = self.w - lr * dldw
            self.b = self.w - lr * dldb
            cost = self.cost_function(X, y)
            costs.append(cost)
            print(f' {i} Cost: {cost:.6f} | dldw: {dldw:.6f} | dldb : {dldb:.6f}  w: {self.w:.6f} | b: {self.b:.6f}')
        return costs

    def predict(self, X):
        return self.w * X + self.b
    
    def cost_function(self, X, y):
        m = len(X)
        y_pred = self.predict(X)
        cost = 0
        for i in range(m):
            cost += (y_pred[i] - y[i]) ** 2
        return cost / m
    
    def gradient_descent(self, X, y, lr):
        m = len(X)
        y_pred = self.predict(X)
        dldw = 0
        dldb = 0

        for i in range(m):
            dldw += 2 * X[i] * (y_pred[i] - y[i])
            dldb += 2 * (y_pred[i] - y[i])

        return dldw, dldb