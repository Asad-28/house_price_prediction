# House Price Prediction

This repository contains code for a machine learning model that predicts house prices using linear regression on the Boston housing dataset.

## Code Description

The code performs the following tasks:

- Imports the necessary libraries for exploring, visualizing, and creating models from the dataset.
- Loads the Boston house dataset from the sklearn datasets.
- Creates a dataframe using the dataset and performs preprocessing on the dataframe.
- Explores the data by displaying information such as the number of rows, columns, observations, missing values, duplicated records, and outliers.
- Visualizes the data through scatter plots and bivariate plots.
- Implements the linear regression model on the dataset.
- Splits the dataset into training and testing sets.
- Fits the linear regression model to the training data and predicts house prices for the testing data.
- Evaluates the model by calculating metrics such as mean absolute error (MAE), mean squared error (MSE), square root of mean squared error (SRMSE), and R-squared score.
- Plots the predicted house prices against the Nitric Oxide content.
- Displays a pair plot of the selected features against the house prices.

## Prerequisites

To run the code, make sure you have the following dependencies installed:

- numpy
- matplotlib
- sklearn
- pandas
- seaborn
- scipy

## Usage

1. Open the Jupyter notebook `House_Price_Prediction.ipynb` in a Python environment or a Jupyter notebook environment.

2. Run the cells in the notebook sequentially to execute the code.

## Acknowledgments

- The code is based on the Boston housing dataset provided by sklearn.
- This project is for educational purposes and serves as an example for implementing a machine learning model for house price prediction.
