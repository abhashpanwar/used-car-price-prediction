<div>
    <h1 align='center'>Used Car Price Prediction using Machine Learning</h1>
    <img src='https://miro.medium.com/max/647/1*ZOcUPrSXLYucFxppoI-dYg.png'/>
    <h1 id="39eb" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">Content</h1>
    <ol class="">
        <li id="4a97" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl ln lo lp ap" data-selectable-paragraph="">Data Cleaning (Identifying null values, filling missing values and removing outliers)</li>
        <li id="fd34" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">Data Preprocessing (Standardization or Normalization)</li>
        <li id="660b" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">ML Models: Linear Regression, Ridge Regression, Lasso, KNN, Random Forest Regressor, Bagging Regressor, Adaboost Regressor, and XGBoost</li>
        <li id="9def" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">Comparison of the performance of the models</li>
        <li id="76fe" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">Some insights from data</li>
    </ol>
    <h2 id="eb32" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">Why price feature is scaled by log transformation?</h2>
    <p id="b92f" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">In the regression model, for any fixed value of X, Y is normally distributed in this problem data-target value (Price ) not normally distributed, it is right Skewed.</p>
    <p id="4404" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap"
        data-selectable-paragraph="">To solve this problem, the log transformation on the target variable is applied when it has skewed distribution and we need to apply an inverse function on the predicted values to get the actual predicted target value.</p>
    <p id="b2d0" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap"
        data-selectable-paragraph="">Due to this, for evaluating the model, the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_log_error.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">RMSLE</em></a> is calculated
        to check the error and <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">R2 Score</em></a> is also calculated to evaluate
        the accuracy of the model.</p>
    <h1 id="8475" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">Some Key Concepts:</h1>
    <ul class="">
        <li id="a531" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl mh lo lp ap" data-selectable-paragraph=""><strong class="ka mi">Learning Rate: </strong>Learning rate is a hyper-parameter that controls how much we are adjusting the weights of our network with respect to the loss gradient. The lower the value, the slower we travel along the downward
            slope. While this might be a good idea (using a low learning rate) in terms of making sure that we do not miss any local minima, it could also mean that we’ll be taking a long time to converge — especially if we get stuck on a plateau region.</li>
        <li
            id="7ff7" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl mh lo lp ap" data-selectable-paragraph=""><strong class="ka mi">n_estimators</strong>: This is the number of trees you want to build before taking the maximum voting or averages of predictions. A higher number of trees give you better performance but make your code slower.</li>
            <li id="6d88"
                class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl mh lo lp ap" data-selectable-paragraph=""><strong class="ka mi">R² Score: </strong>It is a statistical measure of how close the data are to the fitted regression line. It is also known as the coefficient of determination, or the coefficient of multiple determination for multiple regression.
                0% indicates that the model explains none of the variability of the response data around its mean.</li>
    </ul>
    <h1 id="dc96" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph=""><strong class="bk">1. The Data:</strong></h1>
    <p id="4bba" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">The dataset used in this project was downloaded from <a href="https://www.kaggle.com/austinreese/craigslist-carstrucks-data" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">kaggle</a>.</p>
    <h1 id="92bf" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap"
        data-selectable-paragraph=""><strong class="bk">2. Data Cleaning:</strong></h1>
    <p id="587d" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">The first step is to remove irrelevant/useless features like ‘url’, ’region_url’, ’vin’, ’image_url’, ’description’, ’county’, ’state’ from the dataset.</p>
    <p id="64fa" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">As a next step, check missing values for each feature.</p>
    <img src='https://miro.medium.com/max/565/1*2EPrbZHIVWGSz6xAqUYIIA.png' />
    <p id="1e6e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Next, now missing values were filled with appropriate values by an appropriate method.</p>
    <p id="772f" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">To fill the missing values, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">IterativeImputer</em></a> method is used
        and different estimators are implemented then calculated <a href="https://en.wikipedia.org/wiki/Mean_squared_error" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">MSE</em></a> of each estimator using <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html"
            class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">cross_val_score</em></a></p>
    <ol class="">
        <li id="8b67" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl ln lo lp ap" data-selectable-paragraph="">Mean and Median</li>
        <li id="a90b" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">BayesianRidge Estimator</li>
        <li id="a986" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">DecisionTreeRegressor Estimator</li>
        <li id="e451" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">ExtraTreesRegressor Estimator</li>
        <li id="5c07" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap" data-selectable-paragraph="">KNeighborsRegressor Estimator</li>
    </ol>
    <figure class="je jf jg jh ji jj ge gf paragraph-image">
        <div class="mm mn dc mo ai">
            <div class="ge gf ml">
                <div class="jn r dc fs">
                    <div class="mp jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*5Mp3PscLpgg1Zmck6TqQnw.png?q=20" width="864" height="432"></div><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1728/1*5Mp3PscLpgg1Zmck6TqQnw.png" width="864" height="432" srcSet="https://miro.medium.com/max/552/1*5Mp3PscLpgg1Zmck6TqQnw.png 276w, https://miro.medium.com/max/1104/1*5Mp3PscLpgg1Zmck6TqQnw.png 552w, https://miro.medium.com/max/1280/1*5Mp3PscLpgg1Zmck6TqQnw.png 640w, https://miro.medium.com/max/1400/1*5Mp3PscLpgg1Zmck6TqQnw.png 700w" sizes="700px"/></noscript></div>
                </div>
            </div>
        </div>
    </figure>
    <p id="e605" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">From the above figure, we can conclude that the <em class="jz">ExtraTreesRegressor </em>estimator will be better for the imputation method to fill the missing value.</p>
    <figure class="je jf jg jh ji jj ge gf paragraph-image">
        <div class="ge gf mq">
            <div class="jn r dc fs">
                <div class="mr jp r">
                    <div class="db jk s t u eg ai av jl jm"><img alt="zero-null-value" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/58/1*7h8JugocaValwthZeXIo0A.png?q=20" width="419" height="430"></div><noscript><img alt="zero-null-value" class="s t u eg ai" src="https://miro.medium.com/max/838/1*7h8JugocaValwthZeXIo0A.png" width="419" height="430" srcSet="https://miro.medium.com/max/552/1*7h8JugocaValwthZeXIo0A.png 276w, https://miro.medium.com/max/838/1*7h8JugocaValwthZeXIo0A.png 419w" sizes="419px"/></noscript></div>
            </div>
        </div>
    </figure>
    <p id="09db" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">At last, after dealing with missing values there zero null values.</p>
    <p id="677c" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Outliers: </strong>InterQuartile Range (IQR) method is used to remove the outliers from the data.</p>
    
    
    
    <ul class="">
        <li id="fd20" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl mh lo lp ap" data-selectable-paragraph="">From figure 1, the prices whose log is below 6.55 and above 11.55 are the outliers</li>
        <li id="0d88" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl mh lo lp ap" data-selectable-paragraph="">From figure 2, it is impossible to conclude something so IQR is calculated to find outliers i.e odometer values below 6.55 and above 11.55 are the outliers.</li>
        <li id="bb0d" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl mh lo lp ap"
            data-selectable-paragraph="">From figure 3, the year below 1995 and above 2020 are the outliers.</li>
    </ul>
    <p id="06d9" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">At last, Shape of dataset before process= (435849, 25) and after process= (374136, 18). Total 61713 rows and 7 cols removed.</p>
    <h1 id="aa21" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">3. Data preprocessing:</h1>
    <p id="7fa9" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Label Encoder:</strong> In our dataset, 12 features are categorical variables and 4 numerical variables (price column excluded). To apply the ML models, we need to transform these categorical variables into numerical variables.
        And sklearn library <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">LabelEncoder</em></a> is used to solve this
        problem.</p>
    <p id="edb1" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Normalization</strong>: The dataset is not normally distributed. All the features have different ranges. Without normalization, the ML model will try to disregard coefficients of features that have low values because their impact
        will be so small compared to the big value. Hence to normalized, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">sklearn library i.e. MinMaxScaler</em></a>        is used.</p>
    <p id="95c4" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Train the data.</strong> In this process, 90% of the data was split for the train data and 10% of the data was taken as test data.</p>
    <h1 id="7105" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">4. ML Models:</h1>
    <p id="8d35" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">In this section, different machine learning algorithms are used to predict price/target-variable.</p>
    <p id="aa53" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The dataset is supervised dataset, so the models are applied in a given order:</p>
    <ol class="">
        <li id="8d44" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl ln lo lp ap" data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Linear Regression</a></li>
        <li id="e430" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ridge_regression.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Ridge Regression</a></li>
        <li id="474a" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Lasso Regression</a></li>
        <li id="39e4" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">K-Neighbors Regressor</a></li>
        <li id="70aa" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Random Forest Regressor</a></li>
        <li id="740b" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Bagging Regressor</a></li>
        <li id="544d" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">Adaboost Regressor</a></li>
        <li id="8c70" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl ln lo lp ap"
            data-selectable-paragraph=""><a href="https://xgboost.readthedocs.io/en/latest/" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">XGBoost</a></li>
    </ol>
    <h2 id="82a1" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph=""><strong class="bk">1) Linear Regression:</strong></h2>
    <p id="d611" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">In statistics, linear regression is a linear approach to modeling the relationship between a scalar response (or dependent variable) and one or more explanatory variables (or independent variables). In linear regression, the relationships are modeled
        using linear predictor functions whose unknown model parameters are estimated from the data. Such models are called linear models. <a href="https://en.wikipedia.org/wiki/Linear_regression" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">More Details</em></a></p>
    <p
        id="c61b" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Coefficients: The sign of each coefficient indicates the direction of the relationship between a predictor variable and the response variable.</p>
        <ul class="">
            <li id="df89" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl mh lo lp ap" data-selectable-paragraph="">A positive sign indicates that as the predictor variable increases, the response variable also increases.</li>
            <li id="afe4" class="jx jy bx ka b hs lq kc hv lr ke kf ls ia kh lt id kj lu ig kl mh lo lp ap" data-selectable-paragraph="">A negative sign indicates that as the predictor variable increases, the response variable decreases.</li>
        </ul>
        <figure class="je jf jg jh ji jj ge gf paragraph-image">
            <div class="mm mn dc mo ai">
                <div class="ge gf mw">
                    <div class="jn r dc fs">
                        <div class="mp jp r">
                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*2xeokQnLn7xbZ2Ipw5tLDw.png?q=20" width="720" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="720"
                                height="360" src="https://miro.medium.com/max/720/1*2xeokQnLn7xbZ2Ipw5tLDw.png" srcset="https://miro.medium.com/max/276/1*2xeokQnLn7xbZ2Ipw5tLDw.png 276w, https://miro.medium.com/max/552/1*2xeokQnLn7xbZ2Ipw5tLDw.png 552w, https://miro.medium.com/max/640/1*2xeokQnLn7xbZ2Ipw5tLDw.png 640w, https://miro.medium.com/max/700/1*2xeokQnLn7xbZ2Ipw5tLDw.png 700w"
                                sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1440/1*2xeokQnLn7xbZ2Ipw5tLDw.png" width="720" height="360" srcSet="https://miro.medium.com/max/552/1*2xeokQnLn7xbZ2Ipw5tLDw.png 276w, https://miro.medium.com/max/1104/1*2xeokQnLn7xbZ2Ipw5tLDw.png 552w, https://miro.medium.com/max/1280/1*2xeokQnLn7xbZ2Ipw5tLDw.png 640w, https://miro.medium.com/max/1400/1*2xeokQnLn7xbZ2Ipw5tLDw.png 700w" sizes="700px"/></noscript></div>
                    </div>
                </div>
            </div>
        </figure>
        <figure class="mx jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
            <div class="ge gf ms">
                <div class="jn r dc fs">
                    <div class="ni jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*Pi93nxezaDaeqFnf-mbtXA.jpeg?q=20" width="432" height="432"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="432"
                            height="432" src="https://miro.medium.com/max/432/1*Pi93nxezaDaeqFnf-mbtXA.jpeg" srcset="https://miro.medium.com/max/276/1*Pi93nxezaDaeqFnf-mbtXA.jpeg 276w, https://miro.medium.com/max/432/1*Pi93nxezaDaeqFnf-mbtXA.jpeg 432w" sizes="432px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/864/1*Pi93nxezaDaeqFnf-mbtXA.jpeg" width="432" height="432" srcSet="https://miro.medium.com/max/552/1*Pi93nxezaDaeqFnf-mbtXA.jpeg 276w, https://miro.medium.com/max/864/1*Pi93nxezaDaeqFnf-mbtXA.jpeg 432w" sizes="432px"/></noscript></div>
                </div>
            </div>
        </figure>
        <p id="ffa9" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Considering this figure, linear regression suggests that <strong class="ka mi"><em class="jz">year, cylinder, transmission, fuel and odometer</em></strong> these five variables are the most important.</p>
        <figure class="je jf jg jh ji jj ge gf paragraph-image">
            <div class="ge gf nj">
                <div class="jn r dc fs">
                    <div class="nk jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*8BEVfmm_vEGtHw1tdhkEOQ.png?q=20" width="559" height="128"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="559"
                            height="128" src="https://miro.medium.com/max/559/1*8BEVfmm_vEGtHw1tdhkEOQ.png" srcset="https://miro.medium.com/max/276/1*8BEVfmm_vEGtHw1tdhkEOQ.png 276w, https://miro.medium.com/max/552/1*8BEVfmm_vEGtHw1tdhkEOQ.png 552w, https://miro.medium.com/max/559/1*8BEVfmm_vEGtHw1tdhkEOQ.png 559w"
                            sizes="559px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1118/1*8BEVfmm_vEGtHw1tdhkEOQ.png" width="559" height="128" srcSet="https://miro.medium.com/max/552/1*8BEVfmm_vEGtHw1tdhkEOQ.png 276w, https://miro.medium.com/max/1104/1*8BEVfmm_vEGtHw1tdhkEOQ.png 552w, https://miro.medium.com/max/1118/1*8BEVfmm_vEGtHw1tdhkEOQ.png 559w" sizes="559px"/></noscript></div>
                </div>
            </div>
        </figure>
        <h2 id="11c7" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph=""><strong class="bk">2) Ridge Regression:</strong></h2>
        <p id="1005" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Ridge Regression</strong> is a technique for analyzing multiple regression data that suffer from multicollinearity. When multicollinearity occurs, least squares estimates are unbiased, but their variances are large so they
            may be far from the true value.</p>
        <p id="84cf" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">To find the best alpha value in ridge regression, yellowbrick library <a href="https://www.scikit-yb.org/en/latest/api/regressor/alphas.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><em class="jz">AlphaSelection</em></a>            was applied.</p>
        <figure class="je jf jg jh ji jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
            <div class="ge gf nl">
                <div class="jn r dc fs">
                    <div class="nm jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*8HDeCd_6C6UPXg28gqLwBA.png?q=20" width="331" height="329"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="331"
                            height="329" src="https://miro.medium.com/max/331/1*8HDeCd_6C6UPXg28gqLwBA.png" srcset="https://miro.medium.com/max/276/1*8HDeCd_6C6UPXg28gqLwBA.png 276w, https://miro.medium.com/max/331/1*8HDeCd_6C6UPXg28gqLwBA.png 331w" sizes="331px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/662/1*8HDeCd_6C6UPXg28gqLwBA.png" width="331" height="329" srcSet="https://miro.medium.com/max/552/1*8HDeCd_6C6UPXg28gqLwBA.png 276w, https://miro.medium.com/max/662/1*8HDeCd_6C6UPXg28gqLwBA.png 331w" sizes="331px"/></noscript></div>
                </div>
            </div>
        </figure>
        <p id="fb0c" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">From the figure, the best value of alpha to fit the dataset is 20.336.</p>
        <p id="73cd" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Note: The value of alpha is not constant it varies every time.</p>
        <p id="b597" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Using this value of alpha, Ridgeregressor is implemented.</p>
        <figure class="je jf jg jh ji jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
            <div class="ge gf ms">
                <div class="jn r dc fs">
                    <div class="ni jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*JCldwUXZFuxhUFzXXmaHGg.jpeg?q=20" width="432" height="432"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="432"
                            height="432" src="https://miro.medium.com/max/432/1*JCldwUXZFuxhUFzXXmaHGg.jpeg" srcset="https://miro.medium.com/max/276/1*JCldwUXZFuxhUFzXXmaHGg.jpeg 276w, https://miro.medium.com/max/432/1*JCldwUXZFuxhUFzXXmaHGg.jpeg 432w" sizes="432px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/864/1*JCldwUXZFuxhUFzXXmaHGg.jpeg" width="432" height="432" srcSet="https://miro.medium.com/max/552/1*JCldwUXZFuxhUFzXXmaHGg.jpeg 276w, https://miro.medium.com/max/864/1*JCldwUXZFuxhUFzXXmaHGg.jpeg 432w" sizes="432px"/></noscript></div>
                </div>
            </div>
        </figure>
        <p id="c90b" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">Considering this figure, Lasso regression suggests that <strong class="ka mi"><em class="jz">year, cylinder, transmission, fuel and odometer</em></strong> these five variables are the most important.</p>
        <figure class="je jf jg jh ji jj ge gf paragraph-image">
            <div class="ge gf nn">
                <div class="jn r dc fs">
                    <div class="no jp r">
                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*uEVvsEUyBM1fdv6kRErZZg.png?q=20" width="560" height="140"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="560"
                            height="140" src="https://miro.medium.com/max/560/1*uEVvsEUyBM1fdv6kRErZZg.png" srcset="https://miro.medium.com/max/276/1*uEVvsEUyBM1fdv6kRErZZg.png 276w, https://miro.medium.com/max/552/1*uEVvsEUyBM1fdv6kRErZZg.png 552w, https://miro.medium.com/max/560/1*uEVvsEUyBM1fdv6kRErZZg.png 560w"
                            sizes="560px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1120/1*uEVvsEUyBM1fdv6kRErZZg.png" width="560" height="140" srcSet="https://miro.medium.com/max/552/1*uEVvsEUyBM1fdv6kRErZZg.png 276w, https://miro.medium.com/max/1104/1*uEVvsEUyBM1fdv6kRErZZg.png 552w, https://miro.medium.com/max/1120/1*uEVvsEUyBM1fdv6kRErZZg.png 560w" sizes="560px"/></noscript></div>
                </div>
            </div>
        </figure>
        <p id="e925" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The performance of ridge regression is almost the same as Linear Regression.</p>
        <h2 id="5255" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">3)Lasso Regression:</h2>
        <p id="c90c" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">Lasso regression is a type of linear regression that uses shrinkage. Shrinkage is where data values are shrunk towards a central point, like the mean. The lasso procedure encourages simple, sparse models (i.e. models with fewer parameters).</p>
        <p
            id="f7a4" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">Why Lasso regression is used?</strong></p>
            <p id="be9b" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The goal of lasso regression is to obtain the subset of predictors that minimizes prediction error for a quantitative response variable. The lasso does this by imposing a constraint on the model parameters that causes regression coefficients
                for some variables to shrink toward zero.</p>
            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                <div class="ge gf np">
                    <div class="jn r dc fs">
                        <div class="nq jp r">
                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*6RYwCGUqNXZrKReGji1TUA.png?q=20" width="556" height="125"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="556"
                                height="125" src="https://miro.medium.com/max/556/1*6RYwCGUqNXZrKReGji1TUA.png" srcset="https://miro.medium.com/max/276/1*6RYwCGUqNXZrKReGji1TUA.png 276w, https://miro.medium.com/max/552/1*6RYwCGUqNXZrKReGji1TUA.png 552w, https://miro.medium.com/max/556/1*6RYwCGUqNXZrKReGji1TUA.png 556w"
                                sizes="556px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1112/1*6RYwCGUqNXZrKReGji1TUA.png" width="556" height="125" srcSet="https://miro.medium.com/max/552/1*6RYwCGUqNXZrKReGji1TUA.png 276w, https://miro.medium.com/max/1104/1*6RYwCGUqNXZrKReGji1TUA.png 552w, https://miro.medium.com/max/1112/1*6RYwCGUqNXZrKReGji1TUA.png 556w" sizes="556px"/></noscript></div>
                    </div>
                </div>
            </figure>
            <p id="8008" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">But for this dataset, there is no need for lasso regression as there no much difference in error.</p>
            <h2 id="c564" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">4)KNeighbors Regressor: Regression-based on k-nearest neighbors.</h2>
            <p id="f080" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">The target is predicted by local interpolation of the targets associated with the nearest neighbors in the training set.</p>
            <p id="1632" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><em class="jz">k</em>-NN is a type of <a href="https://en.wikipedia.org/wiki/Instance-based_learning" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">instance-based learning</a>, or <a href="https://en.wikipedia.org/wiki/Lazy_learning"
                    class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">lazy learning</a>, where the function is only approximated locally and all computation is deferred until function evaluation. <a href="https://www.kite.com/python/docs/sklearn.neighbors.KNeighborsRegressor"
                    class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><strong class="ka mi"><em class="jz">Read More</em></strong></a></p>
            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                <div class="ge gf nr">
                    <div class="jn r dc fs">
                        <div class="ns jp r">
                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*ozTH4P3Hi0Cw4_rKQPSGtA.png?q=20" width="665" height="222"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="665"
                                height="222" src="https://miro.medium.com/max/665/1*ozTH4P3Hi0Cw4_rKQPSGtA.png" srcset="https://miro.medium.com/max/276/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 276w, https://miro.medium.com/max/552/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 552w, https://miro.medium.com/max/640/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 640w, https://miro.medium.com/max/665/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 665w"
                                sizes="665px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1330/1*ozTH4P3Hi0Cw4_rKQPSGtA.png" width="665" height="222" srcSet="https://miro.medium.com/max/552/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 276w, https://miro.medium.com/max/1104/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 552w, https://miro.medium.com/max/1280/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 640w, https://miro.medium.com/max/1330/1*ozTH4P3Hi0Cw4_rKQPSGtA.png 665w" sizes="665px"/></noscript></div>
                    </div>
                </div>
            </figure>
            <p id="4156" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">From the above figure, for k=5 KNN give the least error. So dataset is trained using n_neighbors=5 and metric=’euclidean’.</p>
            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                <div class="ge gf nt">
                    <div class="jn r dc fs">
                        <div class="nu jp r">
                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*emXX4dcH_AWq7xbnrFokSA.png?q=20" width="550" height="125"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="550"
                                height="125" src="https://miro.medium.com/max/550/1*emXX4dcH_AWq7xbnrFokSA.png" srcset="https://miro.medium.com/max/276/1*emXX4dcH_AWq7xbnrFokSA.png 276w, https://miro.medium.com/max/550/1*emXX4dcH_AWq7xbnrFokSA.png 550w" sizes="550px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1100/1*emXX4dcH_AWq7xbnrFokSA.png" width="550" height="125" srcSet="https://miro.medium.com/max/552/1*emXX4dcH_AWq7xbnrFokSA.png 276w, https://miro.medium.com/max/1100/1*emXX4dcH_AWq7xbnrFokSA.png 550w" sizes="550px"/></noscript></div>
                    </div>
                </div>
            </figure>
            <p id="18ee" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The performance KNN is better and error is decreasing with increased accuracy.</p>
            <h2 id="ad9f" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">5) Random Forest:</h2>
            <p id="4414" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">The random forest is a classification algorithm consisting of many decision trees. It uses bagging and feature randomness when building each individual tree to try to create an uncorrelated forest of trees whose prediction by committee is
                more accurate than that of any individual tree. <a href="https://en.wikipedia.org/wiki/Random_forest" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><strong class="ka mi"><em class="jz">Read More</em></strong></a></p>
            <p
                id="832e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">In our model, 180 decisions are created with max_features 0.5</p>
                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                    <div class="mm mn dc mo ai">
                        <div class="ge gf mw">
                            <div class="jn r dc fs">
                                <div class="mp jp r">
                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*zkcpztR7BL97M_gJKnbJ4w.jpeg?q=20" width="720" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                        width="720" height="360" src="https://miro.medium.com/max/720/1*zkcpztR7BL97M_gJKnbJ4w.jpeg" srcset="https://miro.medium.com/max/276/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 276w, https://miro.medium.com/max/552/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 552w, https://miro.medium.com/max/640/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 640w, https://miro.medium.com/max/700/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 700w"
                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1440/1*zkcpztR7BL97M_gJKnbJ4w.jpeg" width="720" height="360" srcSet="https://miro.medium.com/max/552/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 276w, https://miro.medium.com/max/1104/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 552w, https://miro.medium.com/max/1280/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 640w, https://miro.medium.com/max/1400/1*zkcpztR7BL97M_gJKnbJ4w.jpeg 700w" sizes="700px"/></noscript></div>
                            </div>
                        </div>
                    </div>
                </figure>
                <figure class="mx jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
                    <div class="mm mn dc mo ai">
                        <div class="ge gf nv">
                            <div class="jn r dc fs">
                                <div class="ni jp r">
                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*etX_i8kHXxAA2ZV1relcBw.jpeg?q=20" width="3600" height="3600"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                        width="3600" height="3600" src="https://miro.medium.com/max/3600/1*etX_i8kHXxAA2ZV1relcBw.jpeg" srcset="https://miro.medium.com/max/276/1*etX_i8kHXxAA2ZV1relcBw.jpeg 276w, https://miro.medium.com/max/500/1*etX_i8kHXxAA2ZV1relcBw.jpeg 500w"
                                        sizes="500px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/7200/1*etX_i8kHXxAA2ZV1relcBw.jpeg" width="3600" height="3600" srcSet="https://miro.medium.com/max/552/1*etX_i8kHXxAA2ZV1relcBw.jpeg 276w, https://miro.medium.com/max/1000/1*etX_i8kHXxAA2ZV1relcBw.jpeg 500w" sizes="500px"/></noscript></div>
                            </div>
                        </div>
                    </div>
                </figure>
                <p id="f768" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">This is the simple bar plot which illustrates that <strong class="ka mi"><em class="jz">year</em></strong> is most important feature of a car and then <strong class="ka mi"><em class="jz">odometer</em></strong> variable and then others.</p>
                <figure
                    class="je jf jg jh ji jj ge gf paragraph-image">
                    <div class="ge gf nw">
                        <div class="jn r dc fs">
                            <div class="nx jp r">
                                <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*j_A8fK0kDW4XamxL7Fch3A.png?q=20" width="557" height="124"></div><img alt="Image for post" class="eu ve s t u eg ai jt" width="557"
                                    height="124" src="https://miro.medium.com/max/557/1*j_A8fK0kDW4XamxL7Fch3A.png" srcset="https://miro.medium.com/max/276/1*j_A8fK0kDW4XamxL7Fch3A.png 276w, https://miro.medium.com/max/552/1*j_A8fK0kDW4XamxL7Fch3A.png 552w, https://miro.medium.com/max/557/1*j_A8fK0kDW4XamxL7Fch3A.png 557w"
                                    sizes="557px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1114/1*j_A8fK0kDW4XamxL7Fch3A.png" width="557" height="124" srcSet="https://miro.medium.com/max/552/1*j_A8fK0kDW4XamxL7Fch3A.png 276w, https://miro.medium.com/max/1104/1*j_A8fK0kDW4XamxL7Fch3A.png 552w, https://miro.medium.com/max/1114/1*j_A8fK0kDW4XamxL7Fch3A.png 557w" sizes="557px"/></noscript></div>
                        </div>
                    </div>
                    </figure>
                    <p id="35f6" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The performance of Random forest is better and accuracy is increased by approx 10% which is good. Since the random forest is using bagging when building each individual tree so next Bagging Regressor will be performed.</p>
                    <h2 id="7a4b"
                        class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">6) Bagging Regressor:</h2>
                    <p id="cdc5" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">A Bagging regressor is an ensemble meta-estimator that fits base regressors each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction.
                        Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.
                        <a
                            href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><strong class="ka mi"><em class="jz">Read More</em></strong></a>
                    </p>
                    <p id="2a2e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">In our model, DecisionTreeRegressor is used as the estimator with max_depth=20 which creates 50 decision trees and results show below.</p>
                    <figure class="je jf jg jh ji jj ge gf paragraph-image">
                        <div class="ge gf ny">
                            <div class="jn r dc fs">
                                <div class="nz jp r">
                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*cAW2QHkrcotgdGiBUgZn_Q.png?q=20" width="547" height="120"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                        width="547" height="120" src="https://miro.medium.com/max/547/1*cAW2QHkrcotgdGiBUgZn_Q.png" srcset="https://miro.medium.com/max/276/1*cAW2QHkrcotgdGiBUgZn_Q.png 276w, https://miro.medium.com/max/547/1*cAW2QHkrcotgdGiBUgZn_Q.png 547w"
                                        sizes="547px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1094/1*cAW2QHkrcotgdGiBUgZn_Q.png" width="547" height="120" srcSet="https://miro.medium.com/max/552/1*cAW2QHkrcotgdGiBUgZn_Q.png 276w, https://miro.medium.com/max/1094/1*cAW2QHkrcotgdGiBUgZn_Q.png 547w" sizes="547px"/></noscript></div>
                            </div>
                        </div>
                    </figure>
                    <p id="929f" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">The performance of Random Forest is much better than the Bagging regressor.</p>
                    <p id="401e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph=""><strong class="ka mi">The key difference between Random forest and Bagging: </strong>The fundamental difference is that in <strong class="ka mi">Random forests</strong>, only a subset of features are selected at random out of the total
                        <strong class="ka mi">and</strong> the best split feature from the subset is used to split each node in a tree, unlike in <strong class="ka mi">bagging</strong> where all features are considered for splitting a node.</p>
                    <h2 id="132e"
                        class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">7) Adaboost regressor:</h2>
                    <p id="1037" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">AdaBoost can be used to boost the performance of any machine learning algorithm. Adaboost helps you combine multiple “weak classifiers” into a single “strong classifier”. <strong class="ka mi">Library used: </strong><a href="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html"
                            class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">AdaBoostRegressor</a> &amp; <a href="https://en.wikipedia.org/wiki/AdaBoost" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><strong class="ka mi"><em class="jz">Read More</em></strong></a></p>
                    <figure
                        class="je jf jg jh ji jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
                        <div class="ge gf oa">
                            <div class="jn r dc fs">
                                <div class="ni jp r">
                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg?q=20" width="360" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                        width="360" height="360" src="https://miro.medium.com/max/360/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg" srcset="https://miro.medium.com/max/276/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg 276w, https://miro.medium.com/max/360/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg 360w"
                                        sizes="360px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/720/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg" width="360" height="360" srcSet="https://miro.medium.com/max/552/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg 276w, https://miro.medium.com/max/720/1*jKBeeEm8CJSV2JBccKcxXQ.jpeg 360w" sizes="360px"/></noscript></div>
                            </div>
                        </div>
                        </figure>
                        <p id="965a" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">This is the simple bar plot which illustrates that <strong class="ka mi"><em class="jz">year</em></strong> is the most important feature of a car and then <strong class="ka mi"><em class="jz">odometer</em></strong> variable and
                            then model, etc.</p>
                        <p id="78d7" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">In our model, DecisionTreeRegressor is used as an estimator with 24 max_depth and creates 200 trees &amp; learning the model with 0.6 learning_rate and result shown below.</p>
                        <figure class="je jf jg jh ji jj ge gf paragraph-image">
                            <div class="ge gf ob">
                                <div class="jn r dc fs">
                                    <div class="oc jp r">
                                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*YFhcbflIh3SsMSMRsg_cZw.png?q=20" width="552" height="122"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                            width="552" height="122" src="https://miro.medium.com/max/552/1*YFhcbflIh3SsMSMRsg_cZw.png" srcset="https://miro.medium.com/max/276/1*YFhcbflIh3SsMSMRsg_cZw.png 276w, https://miro.medium.com/max/552/1*YFhcbflIh3SsMSMRsg_cZw.png 552w"
                                            sizes="552px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1104/1*YFhcbflIh3SsMSMRsg_cZw.png" width="552" height="122" srcSet="https://miro.medium.com/max/552/1*YFhcbflIh3SsMSMRsg_cZw.png 276w, https://miro.medium.com/max/1104/1*YFhcbflIh3SsMSMRsg_cZw.png 552w" sizes="552px"/></noscript></div>
                                </div>
                            </div>
                        </figure>
                        <h2 id="76c0" class="lv kr bx bw fh lw lx kc ly lz ke ma mb ia mc md id me mf ig mg ap" data-selectable-paragraph="">8) XGBoost: XGBoost stands for e<strong class="bk">X</strong>treme <strong class="bk">G</strong>radient <strong class="bk">B</strong>oosting</h2>
                        <p id="939c" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap"
                            data-selectable-paragraph="">XGBoost is an <a href="https://en.wikipedia.org/wiki/Ensemble_learning" class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow">ensemble learning </a>method.XGBoost is an implementation of gradient boosted decision trees
                            designed for speed and performance. The beauty of this powerful algorithm lies in its scalability, which drives fast learning through parallel and distributed computing and offers efficient memory usage. <a href="https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/"
                                class="cl dy km kn ko kp" target="_blank" rel="noopener nofollow"><strong class="ka mi"><em class="jz">Read More</em></strong></a></p>
                        <figure class="je jf jg jh ji jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
                            <div class="ge gf oa">
                                <div class="jn r dc fs">
                                    <div class="ni jp r">
                                        <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg?q=20" width="360" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                            width="360" height="360" src="https://miro.medium.com/max/360/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg" srcset="https://miro.medium.com/max/276/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg 276w, https://miro.medium.com/max/360/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg 360w"
                                            sizes="360px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/720/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg" width="360" height="360" srcSet="https://miro.medium.com/max/552/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg 276w, https://miro.medium.com/max/720/1*yXftMIVgQZLnjMJ92Z1w0A.jpeg 360w" sizes="360px"/></noscript></div>
                                </div>
                            </div>
                        </figure>
                        <p id="1bfe" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">This is the simple bar plot in descending of importance which illustrates that which<strong class="ka mi"><em class="jz"> </em>feature/variable<em class="jz"> </em></strong>is an important feature of a car is more important.</p>
                        <p
                            id="e99a" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">According to XGBoost, <strong class="ka mi">Odometer</strong> is an important feature whereas from the previous models <strong class="ka mi">year</strong> is an important feature.</p>
                            <p id="3d70" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap"
                                data-selectable-paragraph="">In this model,200 decision trees are created of 24 max depth and the model is learning the parameter with a 0.4 learning rate.</p>
                            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                <div class="ge gf np">
                                    <div class="jn r dc fs">
                                        <div class="od jp r">
                                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*QKrWdcpHiulRLLQuw8pPxw.png?q=20" width="556" height="123"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                width="556" height="123" src="https://miro.medium.com/max/556/1*QKrWdcpHiulRLLQuw8pPxw.png" srcset="https://miro.medium.com/max/276/1*QKrWdcpHiulRLLQuw8pPxw.png 276w, https://miro.medium.com/max/552/1*QKrWdcpHiulRLLQuw8pPxw.png 552w, https://miro.medium.com/max/556/1*QKrWdcpHiulRLLQuw8pPxw.png 556w"
                                                sizes="556px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1112/1*QKrWdcpHiulRLLQuw8pPxw.png" width="556" height="123" srcSet="https://miro.medium.com/max/552/1*QKrWdcpHiulRLLQuw8pPxw.png 276w, https://miro.medium.com/max/1104/1*QKrWdcpHiulRLLQuw8pPxw.png 552w, https://miro.medium.com/max/1112/1*QKrWdcpHiulRLLQuw8pPxw.png 556w" sizes="556px"/></noscript></div>
                                    </div>
                                </div>
                            </figure>
                            <h1 id="279a" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">4)Comparison of the performance of the models:</h1>
                            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                <div class="mm mn dc mo ai">
                                    <div class="ge gf oe">
                                        <div class="jn r dc fs">
                                            <div class="of jp r">
                                                <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*8eZXofAxqfRwtT-i0dtFKg.jpeg?q=20" width="12000" height="3600"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                    width="12000" height="3600" src="https://miro.medium.com/max/12000/1*8eZXofAxqfRwtT-i0dtFKg.jpeg" srcset="https://miro.medium.com/max/276/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 276w, https://miro.medium.com/max/552/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 552w, https://miro.medium.com/max/640/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 640w, https://miro.medium.com/max/700/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 700w"
                                                    sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/24000/1*8eZXofAxqfRwtT-i0dtFKg.jpeg" width="12000" height="3600" srcSet="https://miro.medium.com/max/552/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 276w, https://miro.medium.com/max/1104/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 552w, https://miro.medium.com/max/1280/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 640w, https://miro.medium.com/max/1400/1*8eZXofAxqfRwtT-i0dtFKg.jpeg 700w" sizes="700px"/></noscript></div>
                                        </div>
                                    </div>
                                </div>
                            </figure>
                            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                <div class="ge gf og">
                                    <div class="jn r dc fs">
                                        <div class="oh jp r">
                                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*z81UsHSR2SkVrRsDnzyWrw.png?q=20" width="554" height="330"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                width="554" height="330" src="https://miro.medium.com/max/554/1*z81UsHSR2SkVrRsDnzyWrw.png" srcset="https://miro.medium.com/max/276/1*z81UsHSR2SkVrRsDnzyWrw.png 276w, https://miro.medium.com/max/552/1*z81UsHSR2SkVrRsDnzyWrw.png 552w, https://miro.medium.com/max/554/1*z81UsHSR2SkVrRsDnzyWrw.png 554w"
                                                sizes="554px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1108/1*z81UsHSR2SkVrRsDnzyWrw.png" width="554" height="330" srcSet="https://miro.medium.com/max/552/1*z81UsHSR2SkVrRsDnzyWrw.png 276w, https://miro.medium.com/max/1104/1*z81UsHSR2SkVrRsDnzyWrw.png 552w, https://miro.medium.com/max/1108/1*z81UsHSR2SkVrRsDnzyWrw.png 554w" sizes="554px"/></noscript></div>
                                    </div>
                                </div>
                            </figure>
                            <p id="e8c6" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">From the above figures, we can conclude that XGBoost regressor with 89.662% accuracy is performing better than other models.</p>
                            <h1 id="f6bb" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">5) Some insights from the dataset:</h1>
                            <p id="fde9" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">1</span>From the pair plot, we can’t conclude anything. There is no correlation between the variables.</p>
                            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                <div class="ge gf or">
                                    <div class="jn r dc fs">
                                        <div class="ni jp r">
                                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*4prrHWaa8gMCRoivyabCJQ.png?q=20" width="540" height="540"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                width="540" height="540" src="https://miro.medium.com/max/540/1*4prrHWaa8gMCRoivyabCJQ.png" srcset="https://miro.medium.com/max/276/1*4prrHWaa8gMCRoivyabCJQ.png 276w, https://miro.medium.com/max/540/1*4prrHWaa8gMCRoivyabCJQ.png 540w"
                                                sizes="540px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1080/1*4prrHWaa8gMCRoivyabCJQ.png" width="540" height="540" srcSet="https://miro.medium.com/max/552/1*4prrHWaa8gMCRoivyabCJQ.png 276w, https://miro.medium.com/max/1080/1*4prrHWaa8gMCRoivyabCJQ.png 540w" sizes="540px"/></noscript></div>
                                    </div>
                                </div>
                            </figure>
                            <p id="3ec4" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">2</span>From the distplot, we can conclude that initially, the price is increasing rapidly but after a particular point, the price starts decreasing.</p>
                            <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                <div class="ge gf ms">
                                    <div class="jn r dc fs">
                                        <div class="mt jp r">
                                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*DTsSYuGAmFfWgva4LssFSw.png?q=20" width="432" height="288"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                width="432" height="288" src="https://miro.medium.com/max/432/1*DTsSYuGAmFfWgva4LssFSw.png" srcset="https://miro.medium.com/max/276/1*DTsSYuGAmFfWgva4LssFSw.png 276w, https://miro.medium.com/max/432/1*DTsSYuGAmFfWgva4LssFSw.png 432w"
                                                sizes="432px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/864/1*DTsSYuGAmFfWgva4LssFSw.png" width="432" height="288" srcSet="https://miro.medium.com/max/552/1*DTsSYuGAmFfWgva4LssFSw.png 276w, https://miro.medium.com/max/864/1*DTsSYuGAmFfWgva4LssFSw.png 432w" sizes="432px"/></noscript></div>
                                    </div>
                                </div>
                            </figure>
                            <p id="9e2e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">3</span>From figure 1, we analyze that the car price of the diesel variant is high then the price of the electric variant comes. Hybrid variant cars has lowest price.</p>
                            <figure
                                class="je jf jg jh ji jj jt my dw mz na nb nc nd bl ch ne nf ng nh cs paragraph-image">
                                <div class="ge gf oa">
                                    <div class="jn r dc fs">
                                        <div class="ni jp r">
                                            <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*a5diIhr--h_-wTlt6lQKNA.png?q=20" width="360" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                width="360" height="360" src="https://miro.medium.com/max/360/1*a5diIhr--h_-wTlt6lQKNA.png" srcset="https://miro.medium.com/max/276/1*a5diIhr--h_-wTlt6lQKNA.png 276w, https://miro.medium.com/max/360/1*a5diIhr--h_-wTlt6lQKNA.png 360w"
                                                sizes="360px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/720/1*a5diIhr--h_-wTlt6lQKNA.png" width="360" height="360" srcSet="https://miro.medium.com/max/552/1*a5diIhr--h_-wTlt6lQKNA.png 276w, https://miro.medium.com/max/720/1*a5diIhr--h_-wTlt6lQKNA.png 360w" sizes="360px"/></noscript></div>
                                    </div>
                                </div>
                                </figure>
                                <p id="a3b3" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">4</span> From figure 2, we analyze that the car price of the respective fuel also depends upon the condition of the car.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf mw">
                                            <div class="jn r dc fs">
                                                <div class="mp jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*PD2pzladq6HRqM7_I_j_Rw.png?q=20" width="720" height="360"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="720" height="360" src="https://miro.medium.com/max/720/1*PD2pzladq6HRqM7_I_j_Rw.png" srcset="https://miro.medium.com/max/276/1*PD2pzladq6HRqM7_I_j_Rw.png 276w, https://miro.medium.com/max/552/1*PD2pzladq6HRqM7_I_j_Rw.png 552w, https://miro.medium.com/max/640/1*PD2pzladq6HRqM7_I_j_Rw.png 640w, https://miro.medium.com/max/700/1*PD2pzladq6HRqM7_I_j_Rw.png 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1440/1*PD2pzladq6HRqM7_I_j_Rw.png" width="720" height="360" srcSet="https://miro.medium.com/max/552/1*PD2pzladq6HRqM7_I_j_Rw.png 276w, https://miro.medium.com/max/1104/1*PD2pzladq6HRqM7_I_j_Rw.png 552w, https://miro.medium.com/max/1280/1*PD2pzladq6HRqM7_I_j_Rw.png 640w, https://miro.medium.com/max/1400/1*PD2pzladq6HRqM7_I_j_Rw.png 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <p id="be5b" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">5</span>From figure 3, we analyze that car prices are increasing per year after 1995, and from figure 4, the number of cars also increasing per year, and at some point i.e in
                                    2012yr, the number of cars is nearly the same.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf oe">
                                            <div class="jn r dc fs">
                                                <div class="mp jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg?q=20" width="12000" height="6000"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="12000" height="6000" src="https://miro.medium.com/max/12000/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg" srcset="https://miro.medium.com/max/276/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 276w, https://miro.medium.com/max/552/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 552w, https://miro.medium.com/max/640/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 640w, https://miro.medium.com/max/700/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/24000/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg" width="12000" height="6000" srcSet="https://miro.medium.com/max/552/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 276w, https://miro.medium.com/max/1104/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 552w, https://miro.medium.com/max/1280/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 640w, https://miro.medium.com/max/1400/1*JBIWEiWC0nr4gr8P4vHGGA.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <p id="1ae9" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">6</span>From figure 5, we can analyze that the price of the cars also depends upon the condition of the car, and from figure 6, price varies with the condition of the cars with
                                    there size also.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf os">
                                            <div class="jn r dc fs">
                                                <div class="ot jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*MJrzsqYvonlwCKQfyICkKg.jpeg?q=20" width="7200" height="2400"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="7200" height="2400" src="https://miro.medium.com/max/7200/1*MJrzsqYvonlwCKQfyICkKg.jpeg" srcset="https://miro.medium.com/max/276/1*MJrzsqYvonlwCKQfyICkKg.jpeg 276w, https://miro.medium.com/max/552/1*MJrzsqYvonlwCKQfyICkKg.jpeg 552w, https://miro.medium.com/max/640/1*MJrzsqYvonlwCKQfyICkKg.jpeg 640w, https://miro.medium.com/max/700/1*MJrzsqYvonlwCKQfyICkKg.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/14400/1*MJrzsqYvonlwCKQfyICkKg.jpeg" width="7200" height="2400" srcSet="https://miro.medium.com/max/552/1*MJrzsqYvonlwCKQfyICkKg.jpeg 276w, https://miro.medium.com/max/1104/1*MJrzsqYvonlwCKQfyICkKg.jpeg 552w, https://miro.medium.com/max/1280/1*MJrzsqYvonlwCKQfyICkKg.jpeg 640w, https://miro.medium.com/max/1400/1*MJrzsqYvonlwCKQfyICkKg.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <p id="3951" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">7</span>From figure 7-8, we analyze that price of the cars also various each <strong class="ka mi">transmission</strong> of a car. People are ready to buy the car having “other
                                    transmission” and the price of the cars having “manual transmission” is low.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf os">
                                            <div class="jn r dc fs">
                                                <div class="ot jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg?q=20" width="7200" height="2400"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="7200" height="2400" src="https://miro.medium.com/max/7200/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg" srcset="https://miro.medium.com/max/276/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 276w, https://miro.medium.com/max/552/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 552w, https://miro.medium.com/max/640/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 640w, https://miro.medium.com/max/700/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/14400/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg" width="7200" height="2400" srcSet="https://miro.medium.com/max/552/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 276w, https://miro.medium.com/max/1104/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 552w, https://miro.medium.com/max/1280/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 640w, https://miro.medium.com/max/1400/1*C_9omSXvuIkgC8EBGN-A9Q.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <p id="e3a8" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap oi" data-selectable-paragraph=""><span class="r na oj ok ol om on oo op oq dc">8</span> Below there are similar graphs with same insight but different features.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf ou">
                                            <div class="jn r dc fs">
                                                <div class="ov jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*xpF9jB6QVtjN7625o7mSAw.jpeg?q=20" width="9000" height="4800"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="9000" height="4800" src="https://miro.medium.com/max/9000/1*xpF9jB6QVtjN7625o7mSAw.jpeg" srcset="https://miro.medium.com/max/276/1*xpF9jB6QVtjN7625o7mSAw.jpeg 276w, https://miro.medium.com/max/552/1*xpF9jB6QVtjN7625o7mSAw.jpeg 552w, https://miro.medium.com/max/640/1*xpF9jB6QVtjN7625o7mSAw.jpeg 640w, https://miro.medium.com/max/700/1*xpF9jB6QVtjN7625o7mSAw.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/18000/1*xpF9jB6QVtjN7625o7mSAw.jpeg" width="9000" height="4800" srcSet="https://miro.medium.com/max/552/1*xpF9jB6QVtjN7625o7mSAw.jpeg 276w, https://miro.medium.com/max/1104/1*xpF9jB6QVtjN7625o7mSAw.jpeg 552w, https://miro.medium.com/max/1280/1*xpF9jB6QVtjN7625o7mSAw.jpeg 640w, https://miro.medium.com/max/1400/1*xpF9jB6QVtjN7625o7mSAw.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf os">
                                            <div class="jn r dc fs">
                                                <div class="ot jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg?q=20" width="7200" height="2400"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="7200" height="2400" src="https://miro.medium.com/max/7200/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg" srcset="https://miro.medium.com/max/276/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 276w, https://miro.medium.com/max/552/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 552w, https://miro.medium.com/max/640/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 640w, https://miro.medium.com/max/700/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/14400/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg" width="7200" height="2400" srcSet="https://miro.medium.com/max/552/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 276w, https://miro.medium.com/max/1104/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 552w, https://miro.medium.com/max/1280/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 640w, https://miro.medium.com/max/1400/1*gxm7kSHaGzG5o3u3LKiGCw.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="mm mn dc mo ai">
                                        <div class="ge gf os">
                                            <div class="jn r dc fs">
                                                <div class="ot jp r">
                                                    <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*vxVVzH4mRffiV_1qA1ynMA.jpeg?q=20" width="7200" height="2400"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                        width="7200" height="2400" src="https://miro.medium.com/max/7200/1*vxVVzH4mRffiV_1qA1ynMA.jpeg" srcset="https://miro.medium.com/max/276/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 276w, https://miro.medium.com/max/552/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 552w, https://miro.medium.com/max/640/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 640w, https://miro.medium.com/max/700/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 700w"
                                                        sizes="700px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/14400/1*vxVVzH4mRffiV_1qA1ynMA.jpeg" width="7200" height="2400" srcSet="https://miro.medium.com/max/552/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 276w, https://miro.medium.com/max/1104/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 552w, https://miro.medium.com/max/1280/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 640w, https://miro.medium.com/max/1400/1*vxVVzH4mRffiV_1qA1ynMA.jpeg 700w" sizes="700px"/></noscript></div>
                                            </div>
                                        </div>
                                    </div>
                                </figure>
                                <h1 id="080c" class="kq kr bx bw fh ks kt ku kv kw kx ky kz la lb lc ld le lf lg lh ap" data-selectable-paragraph="">Conclusion:</h1>
                                <p id="8b45" class="jx jy bx ka b hs li kc hv lj ke kf lk ia kh ll id kj lm ig kl gs ap" data-selectable-paragraph="">By performing different ML model, our aim is to get a better result or less error with max accuracy. Our purpose was to predict the price of the used cars having 25 predictors and 509577 data entries.</p>
                                <p id="4e6e" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap"
                                    data-selectable-paragraph="">Initially, data cleaning is performed to remove the null values and outliers from the dataset then ML models are implemented to predict the price of cars.</p>
                                <p id="2077" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap"
                                    data-selectable-paragraph="">Next, with the help of data visualization features were explored deeply. The relation between the features is examined.</p>
                                <p id="f60c" class="jx jy bx ka b hs kb kc hv kd ke kf kg ia kh ki id kj kk ig kl gs ap" data-selectable-paragraph="">From the below table, it can be concluded that XGBoost is the best model for the prediction for used car prices. XGBoost as a regression model gave the best MSLE and RMSLE values.</p>
                                <figure class="je jf jg jh ji jj ge gf paragraph-image">
                                    <div class="ge gf og">
                                        <div class="jn r dc fs">
                                            <div class="oh jp r">
                                                <div class="db jk s t u eg ai av jl jm"><img alt="Image for post" class="s t u eg ai jq jr bc vu" src="https://miro.medium.com/max/60/1*z81UsHSR2SkVrRsDnzyWrw.png?q=20" width="554" height="330"></div><img alt="Image for post" class="eu ve s t u eg ai jt"
                                                    width="554" height="330" src="https://miro.medium.com/max/554/1*z81UsHSR2SkVrRsDnzyWrw.png" srcset="https://miro.medium.com/max/276/1*z81UsHSR2SkVrRsDnzyWrw.png 276w, https://miro.medium.com/max/552/1*z81UsHSR2SkVrRsDnzyWrw.png 552w, https://miro.medium.com/max/554/1*z81UsHSR2SkVrRsDnzyWrw.png 554w"
                                                    sizes="554px"><noscript><img alt="Image for post" class="s t u eg ai" src="https://miro.medium.com/max/1108/1*z81UsHSR2SkVrRsDnzyWrw.png" width="554" height="330" srcSet="https://miro.medium.com/max/552/1*z81UsHSR2SkVrRsDnzyWrw.png 276w, https://miro.medium.com/max/1104/1*z81UsHSR2SkVrRsDnzyWrw.png 552w, https://miro.medium.com/max/1108/1*z81UsHSR2SkVrRsDnzyWrw.png 554w" sizes="554px"/></noscript></div>
                                        </div>
                                    </div>
                                </figure>
                                </div>
