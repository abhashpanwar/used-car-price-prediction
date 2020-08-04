<h1 align='center'>Used Car Price Prediction using Machine Learning</h1>
![image](https://miro.medium.com/max/647/1*ZOcUPrSXLYucFxppoI-dYg.png)

> # Content
**1)Data Cleaning (Identifying null values, filling missing values and removing outliers)** <br>
**2)Data Preprocessing (Standardization or Normalization)**<br>
**3)ML Models: Linear Regression, Ridge Regression, Lasso, KNN, Random Forest Regressor, Bagging Regressor, Adaboost Regressor, and XGBoost.**<br>
**4)Comparison of the performance of the models.**<br>
**5)Some insights from data.**<br>

# Why price feature is scaled by log transformation?
In the regression model, for any fixed value of X, Y is normally distributed in this problem data-target value (Price ) not normally distributed, it is right Skewed.<br>
To solve this problem, the log transformation on the target variable is applied when it has skewed distribution and we need to apply an inverse function on the predicted values to get the actual predicted target value.<br>
Due to this, for evaluating the model, the RMSLE is calculated to check the error and R2 Score is also calculated to evaluate the accuracy of the model.

# Some Key Concepts:
* **Learning Rate:** Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the slower we travel along the downward slope. While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge — especially if we get stuck on a plateau region.
* **n_estimators:** This is the number of trees you want to build before taking the maximum voting or averages of predictions. A higher number of trees give you better performance but make your code slower.
* **R² Score:** It is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. 0% indicates that the model explains none of the variability of the response data around its mean.


