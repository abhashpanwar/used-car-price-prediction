# Used Car Price Prediction using Machine Learning
### (Data Cleaning, Data Preprocessing, 8 Different ML Models and Some Insights from Data)

You can reach all Python scripts relative to this on my github page. If you are interested, you can also find the scripts used for data cleaning and data visualization for this study in the same repository.
Content
    1. Data Cleaning (Identifying null values, filling missing values and removing outliers)
    2. Data Preprocessing (Standardization or Normalization)
    3. ML Models: Linear Regression, Ridge Regression, Lasso, KNN, Random Forest Regressor, Bagging Regressor, Adaboost Regressor, and XGBoost
    4. Comparison of the performance of the models
    5. Some insights from data
Why price feature is scaled by log transformation?
In the regression model, for any fixed value of X, Y is normally distributed in this problem data-target value (Price ) not normally distributed, it is right Skewed.
To solve this problem, the log transformation on the target variable is applied when it has skewed distribution and we need to apply an inverse function on the predicted values to get the actual predicted target value.
Due to this, for evaluating the model, the RMSLE is calculated to check the error and R2 Score is also calculated to evaluate the accuracy of the model.
Some Key Concepts:
    • Learning Rate: Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the slower we travel along the downward slope. While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge — especially if we get stuck on a plateau region.
    • n_estimators: This is the number of trees you want to build before taking the maximum voting or averages of predictions. A higher number of trees give you better performance but make your code slower.
    • R² Score: It is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression. 0% indicates that the model explains none of the variability of the response data around its mean.
1. The Data:
The dataset used in this project was downloaded from kaggle.
2. Data Cleaning:
The first step is to remove irrelevant/useless features like ‘url’, ’region_url’, ’vin’, ’image_url’, ’description’, ’county’, ’state’ from the dataset.
As a next step, check missing values for each feature.

Next, now missing values were filled with appropriate values by an appropriate method.
To fill the missing values, IterativeImputer method is used and different estimators are implemented then calculated MSE of each estimator using cross_val_score
    1. Mean and Median
    2. BayesianRidge Estimator
    3. DecisionTreeRegressor Estimator
    4. ExtraTreesRegressor Estimator
    5. KNeighborsRegressor Estimator

From the above figure, we can conclude that the ExtraTreesRegressor estimator will be better for the imputation method to fill the missing value.

At last, after dealing with missing values there zero null values.
Outliers: InterQuartile Range (IQR) method is used to remove the outliers from the data.

    • From figure 1, the prices whose log is below 6.55 and above 11.55 are the outliers
    • From figure 2, it is impossible to conclude something so IQR is calculated to find outliers i.e odometer values below 6.55 and above 11.55 are the outliers.
    • From figure 3, the year below 1995 and above 2020 are the outliers.
At last, Shape of dataset before process= (435849, 25) and after process= (374136, 18). Total 61713 rows and 7 cols removed.
3. Data preprocessing:
Label Encoder: In our dataset, 12 features are categorical variables and 4 numerical variables (price column excluded). To apply the ML models, we need to transform these categorical variables into numerical variables. And sklearn library LabelEncoder is used to solve this problem.
Normalization: The dataset is not normally distributed. All the features have different ranges. Without normalization, the ML model will try to disregard coefficients of features that have low values because their impact will be so small compared to the big value. Hence to normalized, sklearn library i.e. MinMaxScaler is used.
Train the data. In this process, 90% of the data was split for the train data and 10% of the data was taken as test data.
4. ML Models:
In this section, different machine learning algorithms are used to predict price/target-variable.
The dataset is supervised dataset, so the models are applied in a given order:
    1. Linear Regression
    2. Ridge Regression
    3. Lasso Regression
    4. K-Neighbors Regressor
    5. Random Forest Regressor
    6. Bagging Regressor
    7. Adaboost Regressor
    8. XGBoost
1) Linear Regression:
In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). In linear regression, the relationships are modeled using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models. More Details
Coefficients: The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.
    • A positive sign indicates that as the predictor variable increases, the response variable also increases.
    • A negative sign indicates that as the predictor variable increases, the response variable decreases.

Considering this figure, linear regression suggests that year, cylinder, transmission, fuel and odometer these five variables are the most important.


2) Ridge Regression:
Ridge Regression is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they may be far from the true value.
To find the best alpha value in ridge regression, yellowbrick library AlphaSelection was applied.
From the figure, the best value of alpha to fit the dataset is 20.336.
Note: The value of alpha is not constant it varies every time.
Using this value of alpha, Ridgeregressor is implemented.



Considering this figure, Lasso regression suggests that year, cylinder, transmission, fuel and odometer these five variables are the most important.


The performance of ridge regression is almost the same as Linear Regression.
3)Lasso Regression:
Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters).
Why Lasso regression is used?
The goal of lasso regression is to obtain the subset of predictors that minimizes prediction error for a quantitative response variable. The lasso does this by imposing a constraint on the model parameters that causes regression coefficients for some variables to shrink toward zero.

But for this dataset, there is no need for lasso regression as there no much difference in error.
4)KNeighbors Regressor: Regression-based on k-nearest neighbors.
The target is predicted by local interpolation of the targets associated with the nearest neighbors in the training set.
k-NN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation. Read More

From the above figure, for k=5 KNN give the least error. So dataset is trained using n_neighbors=5 and metric=’euclidean’.

The performance KNN is better and error is decreasing with increased accuracy.
5) Random Forest:
The random forest is a classification algorithm consisting of many decision trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is more accurate than that of any individual tree. Read More
In our model, 180 decisions are created with max_features 0.5

This is the simple bar plot which illustrates that year is most important feature of a car and then odometer variable and then others.


The performance of Random forest is better and accuracy is increased by approx 10% which is good. Since the random forest is using bagging when building each individual tree so next Bagging Regressor will be performed.
6) Bagging Regressor:
A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it. Read More
In our model, DecisionTreeRegressor is used as the estimator with max_depth=20 which creates 50 decision trees and results show below.

The performance of Random Forest is much better than the Bagging regressor.
The key difference between Random forest and Bagging: The fundamental difference is that in Random forests, only a subset of features are selected at random out of the total and the best split feature from the subset is used to split each node in a tree, unlike in bagging where all features are considered for splitting a node.
7) Adaboost regressor:
AdaBoost can be used to boost the performance of any machine learning algorithm. Adaboost helps you combine multiple “weak classifiers” into a single “strong classifier”. Library used: AdaBoostRegressor & Read More

This is the simple bar plot which illustrates that year is the most important feature of a car and then odometer variable and then model, etc.
In our model, DecisionTreeRegressor is used as an estimator with 24 max_depth and creates 200 trees & learning the model with 0.6 learning_rate and result shown below.

8) XGBoost: XGBoost stands for eXtreme Gradient Boosting
XGBoost is an ensemble learning method.XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. The beauty of this powerful algorithm lies in its scalability, which drives fast learning through parallel and distributed computing and offers efficient memory usage. Read More

This is the simple bar plot in descending of importance which illustrates that which feature/variable is an important feature of a car is more important.
According to XGBoost, Odometer is an important feature whereas from the previous models year is an important feature.
In this model,200 decision trees are created of 24 max depth and the model is learning the parameter with a 0.4 learning rate.

4)Comparison of the performance of the models:

From the above figures, we can conclude that XGBoost regressor with 89.662% accuracy is performing better than other models.












5) Some insights from the dataset:
1From the pair plot, we can’t conclude anything. There is no correlation between the variables.

2From the distplot, we can conclude that initially, the price is increasing rapidly but after a particular point, the price starts decreasing.

3From figure 1, we analyze that the car price of the diesel variant is high then the price of the electric variant comes. Hybrid variant cars has lowest price. 

4 From figure 2, we analyze that the car price of the respective fuel also depends upon the condition of the car.





5From figure 3, we analyze that car prices are increasing per year after 1995, and from figure 4, the number of cars also increasing per year, and at some point i.e in 2012yr, the number of cars is nearly the same. 

6From figure 5, we can analyze that the price of the cars also depends upon the condition of the car, and from figure 6, price varies with the condition of the cars with there size also.







7From figure 7-8, we analyze that price of the cars also various each transmission of a car. People are ready to buy the car having “other transmission” and the price of the cars having “manual transmission” is low.
























8 Below there are similar graphs with same insight but different features.


Conclusion:
By performing different ML model, our aim is to get a better result or less error with max accuracy. Our purpose was to predict the price of the used cars having 25 predictors and 509577 data entries.
Initially, data cleaning is performed to remove the null values and outliers from the dataset then ML models are implemented to predict the price of cars.
Next, with the help of data visualization features were explored deeply. The relation between the features is examined.
From the below table, it can be concluded that XGBoost is the best model for the prediction for used car prices. XGBoost as a regression model gave the best MSLE and RMSLE values.
