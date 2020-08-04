# Used Car Price Prediction using Machine Learning
![image](https://miro.medium.com/max/647/1*ZOcUPrSXLYucFxppoI-dYg.png)

> # Content
## 1)Data Cleaning (Identifying null values, filling missing values and removing outliers)<br>
## 2)Data Preprocessing (Standardization or Normalization)<br>
## 3)ML Models: Linear Regression, Ridge Regression, Lasso, KNN, Random Forest Regressor, Bagging Regressor, Adaboost Regressor, and XGBoost.<br>
## 4)Comparison of the performance of the models.<br>
## 5)Some insights from data.<br>

# Why price feature is scaled by log transformation?
In the regression model, for any fixed value of X, Y is normally distributed in this problem data-target value (Price ) not normally distributed, it is right Skewed.<br>
To solve this problem, the log transformation on the target variable is applied when it has skewed distribution and we need to apply an inverse function on the predicted values to get the actual predicted target value.<br>
Due to this, for evaluating the model, the RMSLE is calculated to check the error and R2 Score is also calculated to evaluate the accuracy of the model.
