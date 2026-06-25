# =============================================
# Project 1: House Price Prediction
# Dataset: Boston Housing (Kaggle)
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------------------------
# STEP 1: Load Dataset
# -----------------------------------------------
columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
           'DIS','RAD','TAX','PTRATIO','B','LSTAT','PRICE']

df = pd.read_csv("C:/Users/hp/OneDrive/Desktop/datasets/housing.csv", header=None, names=columns, sep='\s+')

print("Dataset loaded!")
print("Shape:", df.shape)
print(df.head())

# -----------------------------------------------
# STEP 2: Data Exploration & Cleaning
# -----------------------------------------------
print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Basic Stats ---")
print(df.describe())

# -----------------------------------------------
# STEP 3: Feature Selection
# -----------------------------------------------
X = df.drop('PRICE', axis=1)
y = df['PRICE']

print("\nFeatures used:", list(X.columns))

# -----------------------------------------------
# STEP 4: Model Training
# -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel trained successfully!")

# -----------------------------------------------
# STEP 5: Model Evaluation
# -----------------------------------------------
y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print("\n--- Results ---")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.2f}")

# -----------------------------------------------
# STEP 6: Visualization
# -----------------------------------------------

# Plot 1: Actual vs Predicted
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred, color='steelblue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.tight_layout()
plt.savefig("plot1_actual_vs_predicted.png")
plt.show()

# Plot 2: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(6,5))
plt.scatter(y_pred, residuals, color='orange', alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.tight_layout()
plt.savefig("plot2_residuals.png")
plt.show()

print("\nDone! Plots saved.")