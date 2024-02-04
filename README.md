# Linear Regression Implementation

This repository contains implementations of Linear Regression algorithms in Python, demonstrating the core concepts of this fundamental machine learning technique with a focus on housing price prediction based on house area. The implementations are divided into two primary classes:

## 1. `SimpleLinearRegression`
This class implements Linear Regression using a single feature (House area in square feet). It is designed to illustrate the basic principles of Linear Regression, including fitting a model, making predictions, and understanding the underlying mathematics. Key features:

- **No dependencies on external libraries**: Pure Python implementation.
- **Fit Method**: `fit(x, y, lr)` where `x` is the house area, `y` is the house price, and `lr` is the learning rate.
- **Predict Method**: `predict(x)` used for making predictions on new house areas.
- **Cost Function**: Evaluation of the model performance.
- **Gradient Descent**: Optimization of the model.

## 2. `MultipleLinearRegression`
Extends the concepts to handle multiple features (like house area, number of rooms, etc.). It includes both vectorized and standard implementations. Features:

- **Vectorized Implementation**: Efficient handling of multiple features.
- **Standard Implementation**: Traditional approach, ideal for understanding the operations.
- **Fit Method**: Adapted for multiple features.
- **Predict Method**: Extended to handle predictions with multiple features.
- **Cost Function and Gradient Descent**: Adapted for multiple features.

## Visualizations

1. **Linear Regression Line**:
   ![Linear Regression Line](linear_regression.png)
   - *Description*: The plot shows the relationship between house area (in square feet) and house price (in INR). The green line represents the best fit line obtained through the linear regression model.

2. **Cost Function and Gradient Descent Visualization**:
   ![Cost Function and Gradient Descent](gradient_descent.png)
   - *Description*: This plot illustrates the 'cost bowl' curve of the cost function and the progression of the gradient descent algorithm towards the minimum cost. It gives an intuitive understanding of how the model learns.

