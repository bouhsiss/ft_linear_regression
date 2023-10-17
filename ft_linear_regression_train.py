import pandas as pd
import matplotlib.pyplot as plt

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
			current_m (float): The current slope of the linear model.
			current_b (float): The current y-intercept of the linear model.
			x (pd.Series): The independent variable column from a DataFrame.
			y (pd.Series): The actual dependent variable (target) column from a DataFrame.
			learning_rate (float): The learning rate, controlling the step size in the gradient descent.

		Returns:
			(float, float): Updated slope (m) and y-intercept (b) after one step of gradient descent.
    """

    n = len(x)


    m_gradient = (-2/n) * sum(x * (y - (current_m * x + current_b)))
    b_gradient = (-2/n) * sum(y - (current_m * x + current_b))

    updated_m = current_m - learning_rate * m_gradient
    updated_b = current_b - learning_rate * b_gradient

    return updated_m, updated_b

def train(x, y, epochs, learning_rate=0.001):
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
	for i in range(len(x)):
		m, b = gradient_descent(m, b, x, y, learning_rate)
	return m,b



def main():
	data = pd.read_csv("./data.csv")
	m, b = train(data['km'], data['price'], epochs=100)
	plt.scatter(data['km'], data['price'])
	x_values = data['km']
	y_values =(m * x_values) + b
	plt.plot(x_values, y_values, color='red')
	plt.show()


if __name__ == '__main__':
	main()