import csv
import sys
import pandas as pd

class ModelParameters:
    def __init__(self, m, b, min_km, max_km, min_price, max_price):
        self.m = m
        self.b = b
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
            m = float(row['m'])
            b = float(row['b'])
            min_km = float(row['min_km'])
            max_km = float(row['max_km'])
            min_price = float(row['min_price'])
            max_price = float(row['max_price'])

            params = ModelParameters(m, b, min_km, max_km, min_price, max_price)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

# Function to make predictions
def predict(x, m, b):
    print("m " + str(m) + " b " + str(b))
    return (m * x )+ b

def scale_down_x(x):
    scaled_down_x = (x - params.min_km) / (params.max_km - params.min_km)
    print("scaled down x " + str(x) + " : " + str(scaled_down_x))
    return((x - params.min_km) / (params.max_km - params.min_km))

def scale_up_y(y):
    scaled_up_y = (y * (params.max_price - params.min_price)) + params.min_price
    print("scaled up y " + str(y) + " : " + str(scaled_up_y))
    return((y * (params.max_price - params.min_price)) + params.min_price)

def main():
    load_model_params('model_params.csv')
    # Prompt for user input and make a prediction
    try:
        x_input = float(input("Enter the value of mileage for prediction: "))
        predicted_y = scale_up_y(predict(scale_down_x(x_input), params.m, params.b))
        print(f"For input a mileage = {x_input}, the predicted car price is: {predicted_y}")
    except ValueError:
        print("Invalid input. Please enter a valid numeric value for mileage.")


if __name__ == '__main__':
    main()