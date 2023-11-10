import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
import joblib
import seaborn as sns

# this loss function is normally used just for calculating the loss and analysis purpose. (since our main goal is to minimize this function we will train the model using the gradient_descent function)
def mean_squared_error(m, b, x, y):
    """
        Calculate the Mean Squared Error (MSE) for a linear regression model.

        Parameters:
            m (float): The actual slope of the linear model.
            b (float): The actual y-intercept of the linear model.
            x (pd.Series): The independent variable column from a DataFrame.
            y (pd.Series): The actual dependent variable (target) column from a DataFrame.

        Returns:
            float: The Mean Squared Error (MSE) between the actual data points and the predictions
            made by the linear model with given `actual_m` and `actual_b`.
    """

    predicted_y = (m * x) + b

    squared_error = (y - predicted_y) ** 2

    mse = squared_error.mean()

    return(mse)

def gradient_descent(current_m, current_b, x, y, learning_rate):
    """
        Perform one step of gradient descent to update the model parameters.
        
        Parameters:
            current_m(float): The current slope of the linear model.
            current_b(float): The current y-intercept of the linear model.
            x (pd.Series): The independent variable column from a DataFrame.
            y (pd.Series): The actual dependent variable (target) column from a DataFrame.
            learning_rate (float): The learning rate, controlling the step size in the gradient descent.
            
        Returns:
            (float, float): Updated slope(m) and y-intercept (b) after one step of gradient descent.
    """



    n = len(x)

    m_gradient = 0
    b_gradient = 0

    for j in range(n):
        y_pred = (current_m * x[j]) + current_b
        m_gradient += ((y_pred - y[j]) * x[j])
        b_gradient += (y_pred - y[j])
    



    updated_m = current_m - (m_gradient * 2/n) * learning_rate
    updated_b = current_b - (b_gradient * 2/n) * learning_rate

    return(updated_m, updated_b)


def train(x, y, epochs=10000, learning_rate=0.01):
    """
        Train a linear regression model using gradient descent.

        Parameters:
        x (pd.Series): The independent variable column from a DataFrame.
        y (pd.Series): The actual dependent variable (target) column from a DataFrame.
        epochs (int): The number of training epochs (iterations).
        learning_rate (float): The learning rate, controlling the step size in gradient descent.

        Returns:
        (float, float): The trained slope (m) and y-intercept (b) of the linear regression model.
    """
    m = 0
    b = 0
    for i in range(epochs):
        m, b = gradient_descent(m, b, x, y, learning_rate)
    return m,b


def main():
    data = pd.read_csv("../data.csv")

    # data seems to have high variations in both columns, and needs to be standarized (z-score standarization)
    min_km = data['km'].min()
    max_km = data['km'].max()
    min_price = data['price'].min()
    max_price = data['price'].max()
    scaled_km = (data['km'] - min_km) / (max_km - min_km)
    scaled_price = (data['price'] - min_price) / (max_price - min_price)
    # print(data.describe())
    # print(scaled_km.describe())
    # print(scaled_price.describe())


    # train the model to get the most optimum m, b
    m, b = train(scaled_km, scaled_price)


    # writing the model slope and y-intercept in a file
    with open('../model_params.csv', 'w', newline='') as model_file:
        writer = csv.writer(model_file)
        writer.writerow(['m', 'b', 'min_km', 'max_km', 'min_price', 'max_price'])
        writer.writerow([m, b, min_km, max_km, min_price, max_price])


if __name__ == '__main__':
    main()
