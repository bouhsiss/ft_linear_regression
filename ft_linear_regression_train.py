import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
import joblib 

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

    n = float(len(x))

    y_pred = (current_m * x) + current_b

    m_gradient = (-2/n) * sum(x * (y - y_pred)) # derivative of mse wrt m
    b_gradient = (-2/n) * sum(y - y_pred) # derivative of mse wrt b

    updated_m = current_m - (learning_rate * m_gradient)
    updated_b = current_b - (learning_rate * b_gradient)
    return(updated_m, updated_b)

def train(x, y, epochs=500, learning_rate=0.01):
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
    data = pd.read_csv("./data.csv")

    # data seems to have high variations in both columns, and needs to be standarized (z-score standarization)
    mean = data.mean()
    std_dev = data.std()
    scaled_data = (data - mean)/std_dev
    # print(data.describe())
    # print(scaled_data.describe())


    # train the model to get the most optimum m, b
    m, b = train(scaled_data['km'], scaled_data['price'])

    # plotting the data points
    plt.scatter(data['km'], data['price'])

    # plotting the linear regression model
    x_values = scaled_data['km']
    y_values = (m * x_values) + b
    x_values = (x_values * std_dev['km']) + mean['km']
    y_values = (y_values * std_dev['price']) + mean['price']
    plt.plot(x_values, y_values, color='red')
    plt.show()

    # calculating the mse
    print("the mean squared error for this model (precision of the algorithm) : " + str(mean_squared_error(m,b,scaled_data['km'], scaled_data['price'])))

    # writing the model slope and y-intercept in a file
    with open('model_params.csv', 'w', newline='') as model_file:
        writer = csv.writer(model_file)
        writer.writerow(['m', 'b', 'mean_km', 'std_dev_km', 'mean_price', 'std_dev_price'])
        writer.writerow([m, b, mean['km'], std_dev['km'], mean['price'], std_dev['price']])


if __name__ == '__main__':
    main()
