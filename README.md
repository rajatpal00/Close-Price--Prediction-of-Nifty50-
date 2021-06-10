# Close-Price-Prediction-of-Nifty50-
 


## OBJECTIVE : 

To perform time series analysis on the NIFTY stock price and forecasting using univariate ARIMA and ARIMAX modeling techniques. 
We further simplified the problem  to  predict the direction of Close price  movements in the next N days.This is represented as a classification task where there are two possible outcomes (either the index went up in the next day or it went down).
 
The Nifty 50 is a stock market index comprised of the shares of 50 of India's most well-diversified firms, covering various economic sectors such as financial services, engineering, pharmaceuticals, and information technology. The weighting of these 50 stocks is determined by their free float market capitalization.

This is a time-series activity which we come across in our daily lives.










## DATA Description

The Data of Nifty50 is collected from the National Stock Exchange price website. https://www1.nseindia.com/products/content/equities/indices/historical_index_data.htm 

One can also get the past 15 years of data from yahoo finance website https://in.finance.yahoo.com/quote/%5ENSEI/history/. 

We colected the data for 20 years.

Description of columns in the file:
Date — Date of trade
Open — Opening stock value.
High — Highest stock value in that day.
Low — Lowest stock value of the day.
Close — Closing stock value.


# WORKING WITH DIFFERENT MODELS

## Auto-regression model
* ## ARIMA 
* ## ARIMAX
* ## Facebook Prophet









# Machine Learning Approach :
*  ## Light GBM

      LightGBM (Light Gradient Boosting Machine) is a free and open source distributed gradient boosting platform for machine learning that was created by Microsoft. It is used for    ranking, classification, and other machine learning tasks and is based on decision tree algorithms. Quality and scalability are at the forefront of the production process.

      Trees are grown leaf-by-leaf by LightGBM (best-first). It will expand the leaf with the greatest delta loss. Leaf-wise algorithms have a lower loss than level-wise          algorithms when leaf is fixed.

      When data is small, leaf-wise can trigger over-fitting, so LightGBM includes the max depth parameter to restrict tree depth. Even when max depth is defined, trees continue to grow leaf-wise.

      Resource : https://lightgbm.readthedocs.io/en/latest/Quick-Start.html 
       
      One can get complete documentation in the above link.



* ## XG-Boost 
     XGBoost is a distributed gradient boosting library that has been optimised for performance, flexibility, and portability. It uses the Gradient Boosting paradigm to implement machine learning algorithms. XGBoost is a parallel tree boosting (also known as GBDT, GBM) algorithm that solves a variety of data science problems quickly and accurately.
https://xgboost.readthedocs.io/en/latest/python/index.html


### Extreme Gradient Boosting (XGBoost) is just an extension of gradient boosting with the following added advantages:
* Regularization: Standard GBM implementation has no regularization like XGBoost, therefore it also helps to reduce overfitting. In fact, XGBoost is also known as ‘regularized boosting‘ technique.
* Parallel Processing: XGBoost implements parallel processing and is blazingly faster as compared to GBM. But hang on, we know that boosting is a sequential process so how can it be parallelized? We know that each tree can be built only after the previous one, but to make a tree it uses all the cores of the system. XGBoost also supports implementation on Hadoop.
* High Flexibility: XGBoost allows users to define custom optimization objectives and evaluation criteria. This adds a whole new dimension to the model and there is no limit to what we can do.
* Handling Missing Values: XGBoost has an in-built routine to handle missing values. User is required to supply a different value than other observations and pass that as a parameter. XGBoost tries different things as it encounters a missing value on each node and learns which path to take for missing values in future.
* Tree Pruning: A GBM would stop splitting a node when it encounters a negative loss in the split. Thus it is more of a greedy algorithm. XGBoost on the other hand makes splits up to the max_depth specified and then starts pruning the tree backwards and removes splits beyond which there is no positive gain. Another advantage is that sometimes a split of negative loss say -2 may be followed by a split of positive loss +10. GBM would stop as it encounters -2. But XGBoost will go deeper and it will see a combined effect of +8 of the split and keep both.
* Built-in Cross-Validation: XGBoost allows users to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run. This is unlike GBM where we have to run a grid-search and only a limited value can be tested.
* Continue on Existing Model: Users can start training an XGBoost model from its last iteration of previous run. This can be of significant advantage in certain specific applications. GBM implementation of sklearn also has this feature so they are even on this point.


# CONCLUSION :

While using regression models, residuals are quite high.
Difficult to predict numerical target value.
While using classification models, we have a higher chance of predicting the outcome.
