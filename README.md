# Car Price Prediction using Linear Regression

## Project Overview

This project is part of the 42 network cursus. We will create a program to predict the price of a car using a 
linear function trained with a gradient descent algorithm. The project consists of two main programs: 
one for predicting the price of a car based on its mileage and another for training the model using a dataset.

## Project Structure

```
.
|-- bonus
| |-- ft_linear_regression_bonus.py
|-- model_predict
| |-- ft_linear_regression_predict.py
|-- model_train
| |-- ft_linear_regression_train.py
|-- model_params.csv
|-- data.csv
```

### 1. Model Prediction
The first program predicts the price of a car for a given mileage.

- **File**: `model_predict/ft_linear_regression_predict.py`
- **Functionality**: Prompts the user for a mileage and returns the estimated price using the hypothesis:
```
estimatePrice(mileage) = θ0 + (θ1 * mileage)
```
- **Initial Conditions**: Before training, θ0 and θ1 are set to 0.

### 2. Model Training
The second program trains the linear regression model using the provided dataset.

- **File**: `model_train/ft_linear_regression_train.py`
- **Functionality**: Reads the dataset from `data.csv` and performs a linear regression. Updates θ0 and θ1 using gradient descent and saves them to `model_params.csv`.

## Formulas

- Update rules for gradient descent:
```
tmpθ0 = learningRate * (1/m) * Σ (estimatePrice(mileage[i]) - price[i])
tmpθ1 = learningRate * (1/m) * Σ (estimatePrice(mileage[i]) - price[i]) * mileage[i]
```

Note: `m` is the number of data points.

## Bonus

### 1. Plotting Data and Regression Line
- **File**: `bonus/ft_linear_regression_bonus.py`
- **Functionality**: Plots the data and the resulting regression line to visualize the results.

### 2. Calculating Algorithm Precision
- **File**: `bonus/ft_linear_regression_bonus.py`
- **Functionality**: Calculates and displays the precision of the algorithm.

## Dataset

- **File**: `data.csv`
- **Description**: Contains the dataset with car mileage and corresponding prices.

## Parameters

- **File**: `model_params.csv`
- **Description**: Stores the trained parameters θ0 and θ1.

## Usage

1. **Train the Model**:
```
python3 model_train/ft_linear_regression_train.py
```
1. **Predict Car Price**:
```
python3 model_train/ft_linear_regression_predict.py
```

For additional functionalities, use the bonus script:
```
python3 bonus/ft_linear_regression_bonus.py
```

## Requirements
- Python 3.x
- matplotlib (for plotting, if using the bonus script)

## Conclusion

This project provides a foundational understanding of linear regression and gradient descent in machine learning. The model can be extended and applied to other datasets with similar characteristics.
