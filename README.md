# Housing-Price-Analysis
Analysing the Boston housing data to predict the price of a house using various features and factors.

**Project Overview**
-This project involves analyzing the Boston Housing dataset to predict house prices based on various features.
-The dataset contatins information collected by the US census service concerning housing in the area of Boston.
-The goal of this project is to build machine learning model to predict house price (median value i.e., medv) based on provided features.


**Table of Contents**

-Project Overview
-Installation
-Dataset
-Exploratory Data Analysis
-Modellling
-Evaluation
-Results

**Installation**
To run this project, you'll need Python and the following libraries installed
 -Pandas
 -Numpy
 -Matplotlib
 -Seaborn
 -scikit-learn

You can install these packages using: (in Linux)
       pip install pandas numpy matplotlib seaborn scikit-learn

**Dataset**

The dataset can be downloaded from [here](https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv).



**Features**
  -CRIM: Per capita crime rate by town
  -ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
  -INDUS: Proportion of non-retail business acres per town
  -CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
  -NOX: Nitric oxides concentration (parts per 10 million)
  -RM: Average number of rooms per dwelling
  -AGE: Proportion of owner-occupied units built prior to 1940
  -DIS: Weighted distances to five Boston employment centers
  -RAD: Index of accessibility to radial highways
  -TAX: Full-value property tax rate per $10,000
  -PTRATIO: Pupil-teacher ratio by town
  -B: 1000(Bk - 0.63)^2 where Bk is the proportion of Black people by town
  -LSTAT: Percentage of lower status of the population
  -MEDV: Median value of owner-occupied homes in $1000's (Target Variable)

**Exploratory Data Analysis**

In this step, the data is visualized and explored to understand relationships between the features and the target variable. Key insights include:

   -Correlation between different features and the target variable.
   -Distribution of the features.
   -Handling missing values and outliers if any.


**Modelling**

A Linear Regression model is used to predict the target variable (MEDV). The dataset is split into training and testing sets to evaluate the model's performance.
Steps:

  -Data Preprocessing: Feature scaling, handling missing values, etc.
  -Model Training: Training the Linear Regression model on the training data.
  -Predictions: Using the trained model to make predictions on the test data.

**Evaluation**

The model's performance is evaluated using the following metrics:

  -Mean Squared Error (MSE)
  -R-squared (R²) Score


**Results**

The model was able to predict the house prices with an R² score of 0.66 on the test data.








