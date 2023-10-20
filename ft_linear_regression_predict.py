import csv
import sys
import pandas as pd

class ModelParameters:
    def __init__(self, m, b, mean_km, std_dev_km, mean_price, std_dev_price):
        self.m = m
        self.b = b
        self.mean_km = mean_km
        self.std_dev_km = std_dev_km
        self.mean_price = mean_price
        self.std_dev_price = std_dev_price

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
            mean_km = float(row['mean_km'])
            std_dev_km = float(row['std_dev_km'])
            mean_price = float(row['mean_price'])
            std_dev_price = float(row['std_dev_price'])

            params = ModelParameters(m, b, mean_km, std_dev_km, mean_price, std_dev_price)
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

# Function to make predictions
def predict(x, m, b):
    return (m * x )+ b

def scale_down_x(x):
    
    return((x - params.mean_km)/ params.std_dev_km)

def scale_up_y(y):
    return((y * params.std_dev_price) + params.mean_price)

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