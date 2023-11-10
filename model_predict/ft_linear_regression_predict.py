import csv
import sys
import pandas as pd

class ModelParameters:
    def __init__(self, theta1, theta0, min_km, max_km, min_price, max_price):
        self.theta1 = theta1
        self.theta0 = theta0
        self.min_km = min_km
        self.max_km = max_km
        self.min_price = min_price
        self.max_price = max_price

params = None

# Load 'm' and 'b' values from the model_params.csv file
def load_model_params(file_path):
    global params
    try:
        with open(file_path, 'r') as file:
            reader = csv.DictReader(file)
            row = next(reader)
            theta1 = float(row['theta1'])
            theta0 = float(row['theta0'])
            min_km = float(row['min_km'])
            max_km = float(row['max_km'])
            min_price = float(row['min_price'])
            max_price = float(row['max_price'])

            params = ModelParameters(theta1, theta0, min_km, max_km, min_price, max_price)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

# Function to make predictions
def predict(x, theta1, theta0):
    return (theta1 * x )+ theta0

def scale_down_x(x):
    scaled_down_x = (x - params.min_km) / (params.max_km - params.min_km)
    return((x - params.min_km) / (params.max_km - params.min_km))

def scale_up_y(y):
    scaled_up_y = (y * (params.max_price - params.min_price)) + params.min_price
    return((y * (params.max_price - params.min_price)) + params.min_price)

def main():
    load_model_params('../model_params.csv')
    # Prompt for user input and make a prediction
    try:
        x_input = float(input("Enter the value of mileage for prediction: "))
        if(x_input < 0):
            print("Negative mileage? You've entered the Twilight Zone of car predictions. Please provide a positive mileage value and try again.")
        else :
            predicted_y = scale_up_y(predict(scale_down_x(x_input), params.theta1, params.theta0))
            if(predicted_y < 0):
                print("Are you sure this isn't a spaceship?")
            else :
                print(f"For input a mileage = {x_input}, the predicted car price is: {predicted_y}")
    except ValueError:
        print("Invalid input. Please enter a valid numeric value for mileage.")


if __name__ == '__main__':
    main()