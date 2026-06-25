# oasis-infobyte-internship-project1-level2
# House Price Prediction

Predicting house prices based on various features like crime rate, number of rooms, 
tax rate, and more using Linear Regression.

## About the Dataset

Used the Boston Housing dataset which contains 506 samples and 13 features related 
to housing in Boston. Features include crime rate per town, average number of rooms, 
pupil-teacher ratio, and others. Target variable is the median house price.

## What I Did

Started with loading and exploring the dataset — checked shape, missing values, and 
basic statistics. Used all 13 features for training. Split the data 80-20 for 
training and testing, then trained a Linear Regression model.

Evaluated the model using MSE, RMSE, and R2 score. Also plotted Actual vs Predicted 
prices to visually check how well the model performed, and a Residuals plot to check 
for any patterns in errors.

## Results

- Model showed a good R2 score indicating decent predictive performance
- Residuals were mostly scattered around zero — no major pattern found
- Linear Regression worked reasonably well for this dataset

## Libraries Used

pandas, numpy, matplotlib, scikit-learn

## How to Run
1. Place `housing.csv` in the project folder
2. Run:
   python house_pricing.py

## Internship

Oasis Infobyte — Data Analytics Internship | Level 2, Project 1


1. Place `housing.csv` in the project folder
2. Run:
