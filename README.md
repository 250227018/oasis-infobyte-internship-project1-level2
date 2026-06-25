# oasis-infobyte-internship-project1-level2
# Wine Quality Prediction

Predicting whether a red wine is good or not based on its chemical properties like acidity, alcohol content, density, and more.

## About the Dataset

Used the Red Wine Quality dataset from UCI Machine Learning Repository. It has 1599 samples with 11 chemical features. The quality score (0–10) was converted into a binary label — wines scoring 6 or above are considered "good".

## What I Did

Started with basic exploration — checked the shape, missing values, and how quality scores are distributed. Plotted a heatmap to see which features are most correlated with quality. Alcohol turned out to have the strongest positive correlation.

Then trained three classifier models:
- **Random Forest** — ensemble of decision trees, handled the data well
- **SGD Classifier** — gradient descent based, fast but slightly inconsistent
- **SVC** — support vector machine with RBF kernel, gave solid results

## Results

Random Forest came out on top with the highest accuracy. SVC was close behind. SGD was the weakest of the three on this dataset.

## Libraries Used

pandas, numpy, matplotlib, seaborn, scikit-learn

## How to Run

1. Place `winequality-red.csv` in the project folder
2. Run:
```
python wine_quality.py
```
