class SimpleLinearRegression:
    """
    Simple Linear Regression model.

    Parameters:
    - init_w: Initial value for the weight parameter.
    - init_b: Initial value for the bias parameter.
    """

    def __init__(self, init_w=0, init_b=0):
        self.w = init_w
        self.b = init_b
        self.costs = []
    
    def fit(self, x, y, lr, epochs):
        """
        Fit the linear regression model to the training data.

        Parameters:
        - x: Input features.
        - y: Target values.
        - lr: Learning rate.
        - epochs: Number of training epochs.

        """
        for i in range(epochs):
            dldw, dldb = compute_gradient(x, y,self.w, self.b)
            cost = cost_function(x, y, self.w, self.b)
            self.costs.append(cost)
            if i % 100 == 0:
                print(f' {i} Cost: {cost:.6f} | dldw: {dldw:.6f} | dldb : {dldb:.6f}  w: {self.w:.6f} | b: {self.b:.6f}')
            self.w = self.w - lr * dldw
            self.b = self.b - lr * dldb

    def predict(self, x):
        """
        Predict the target values for the given input features.

        Parameters:
        - x: Input features.

        Returns:
        - y_pred: Predicted target values.
        """
        return self.w * x + self.b
    
def cost_function( x, y, w, b):
    """
    Calculate the cost function value.

    Parameters:
    - x: Input features.
    - y: Target values.

    Returns:
    - cost: Cost function value.
    """
    m = len(x)
    cost_sum = 0
    for i in range(m):
        y_pred = w * x[i] + b
        cost = (y_pred - y[i]) ** 2
        cost_sum = cost_sum + cost
    total_cost = (1 / (2 * m)) * cost_sum  

    return total_cost

def compute_gradient( x, y, w, b):
    m = len(x)
    dldw = 0
    dldb = 0

    for i in range(m):
        y_pred = w * x[i] + b
        dldw_i = (y_pred - y[i]) * x[i]
        dldb_i = y_pred - y[i]
        dldw = dldw + dldw_i
        dldb = dldb + dldb_i

    dldw = dldw / m
    dldb = dldb / m
    return dldw, dldb