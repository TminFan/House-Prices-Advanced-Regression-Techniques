# House Price Prediction
Data is from the Kaggle competition: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).<br>
The data includes a training set and a test set, as well as files with a description of the data and a sample submission. The training set has 81 columns (80 features + 1 target), and the test set has 80 columns (only features), thus the test set is not usable for training but is to be used as the submission input dataset.

## Data Cleaning
**N/A values**<br>
The training set has 19 columns with *N/A* values and the test set has 33 columns with *N/A* values. Some of these *N/A* values should be considered as a category. For example, *N/A* values in the column *BsmtQual* are categorised as *No Basement*. Thus, *N/A* values of some object type categorical data columns are replaced with actual corresponding category names. For some numerical type columns, each *N/A* value is replaced with the median value calculated along the column. Some numerical type columns that are associated with other categorical columns have *N/A* values replaced with 0. For example, the column *BsmtFinSF1* (Type 1 finished square feet) is associated with the column *BsmtCond* (condition of the basement). The values of *BsmtFinSF1* are missing because the corresponding rows of *BsmtCond* are coded as *N/A*, but should instead be categorised as *No Basement*, so the *NA* values of *BsmtFinSF1* should be replaced by 0.

## Feature Engineering
**Encoding object type categorical data columns**<br>
Object type categorical data columns are encoded to numerical type categorical columns using the Sklearn OrdinalEncoder. The OrdinalEncoder’s *categories* parameter is set to *auto* by default. The default setting treats input categories as nominal data. The OrdinalEncoder can also encode ordinal data by setting a list of ordered categories in the *categories* parameter. Models are trained with a dataset using the default setting, as well as a dataset specifying ordinal data. Using the default setting achieved a slightly better result.

**Scaling numerical type columns**<br>
Numerical columns are scaled using logarithm (log) transformation with the natural number e as the base. Log transformation can make data become less skewed. The models are also trained with a dataset where numerical columns are scaled using the Sklearn StandardScaler. StandardScaler removes the data mean and scales to unit variance. Results show that the dataset using log transform performs better than the dataset using StandardScaler.

**Removing outliers**<br>
Outliers are detected using the Scikit-learn (Sklearn) Local Outlier Factor algorithm. This is described in the Scikit documentation 2.7. *Novelty and Outlier Detection*:<br>
>>*The neighbors.LocalOutlierFactor (LOF) algorithm computes score (called local outlier factor) reflecting the degree oabnormality of the observations. It measures the locadensity deviation of a given data point with respect to itneighbors. The idea is to detect the samples that have substantially lower density than their neighbors*.[1]

Models are trained with a dataset that has target column outliers removed, as well as a dataset with feature column outliers removed. Results show that these two datasets perform very similarly.

**Feature Selection**
As mentioned in the Data Cleaning section, some columns are associated with each other. The input feature dataset may have collinearity or multicollinearity problems. Collinearity is when two columns have a linear relationship, and multicollinearity is when a column has a linear relationship with two or more columns. When modelling Multiple Linear Regression (MLR), if there exists linear relationships between features in the input dataset, the estimated coefficients will become unstable due to the huge variance of coefficients. Therefore, it is preferred to have less non-independent feature columns in the input data set. The following is a mathematical explanation of why variance of estimated coefficients will become huge when features in a dataset are not independent of each other:<br>
The estimated coefficients in a multiple linear regression is $ \hat{\beta} = (x^{T}x)^{-1}x^{T}y$<br>
To derive the variance of estimated coefficients:<br>
Let $ (x^{T}x)^{-1}x^{T} $ be A and A is a constant matrix.<br>
$ Var(\hat{\beta}) $ = $ Var((x^{T}x)^{-1}x^{T}y) $<br>
= $ Var(AY) = E[((AY)^{2})] - (E[AY])^{2} $
= $ E[A^{2}Y^{2}] - (E[AY])^{2} $
= $ A^{2}E[Y^{2}] - A^{2}(E[Y])^{2} $
= $ A^{2}Var(Y) $
= $ A Var(Y) A^{T} $<br>
$ Var(\hat{\beta}) = \sigma^{2}(x^{T}x)^{-1} $<br>
The inverse of $ (x^{T}x) $ is $ \frac{1}{det|x^{T}x|}adj(x^{T}x) $.<br>
If $ x $ is not full rank, then $ (x^{T}x) $ is also a singular matrix,
so $ (x^{T}x) $ is not inversible.
$ {det|x^{T}x|} $ will be zero, so the $ Var(\hat{\beta}) $ will blow up.<br>
If $ x $ is closed to full rank, then $ (x^{T}x) $ is closed to being non-invertible.
The $ Var(\hat{\beta}) $ will become huge.[2] (Cosma, S., 2015).<br>
The following methods are used to analyse the collinearity and multicollinearity of the dataset:
1. Three tests analysing collinearity:
>>(1) Between continuous data columns using the Pearson correlation method.<br>
(2) Between categorical data columns using the Chi-Square test.<br>
(3) Between each categorical data column and each continuous data column using one-way Welch’s ANOVA analysis.

2. Variance Inflation Factors (VIF) for measuring multicollinearity: The PennSate STAT 501 course explains that VIF quantifies how much the variance of the estimated coefficients are inflated when multicollinearity exists.[3] (Laura, S., Robert, H., Andrew, W., Derek, Y., Iain, P., 2022).

## Models and Dataset Comparison
The following eight regression models are used to predict house prices in house_price_prediction.ipynb:
1. Linear Regression
2. KNeighbor Regression
3. Epsilon-Support Vector Regression (SVR)
4. ElasticNet Regression
5. Random Forest Regression
6. Gradient Boosting Regression
7. Extreme Gradient Boosting Regression (XGBoost)
8. Stacked Generalisation Regression<br>

Each model is trained using the following five datasets. These datasets are generated from the original training set, but use different data preprocessing.
1. No Treatment dataset: the data preprocessing includes encoding object type categorical feature columns, and scaling numerical feature columns using e base logarithm transformation.
2. Y_log_transformed dataset: the data preprocessing includes the preprocessing of the No Treatment dataset, and also has the target column scaled using e base logarithm transformation.
3. Remove Outliers dataset: the data preprocessing includes the preprocessing of the No Treatment dataset, and also has outliers of feature columns removed.
4. Collinearity Features Reduction with Outliers Removed dataset: The data preprocessing includes the preprocessing of the Remove Outliers dataset, and also has some of the feature columns removed using collinearity feature selection.
5. Multicollinearity Features Reduction with Outliers Removed dataset: The data preprocessing includes the preprocessing of the Remove Outliers dataset, and also some of the feature columns are removed using multicollinearity feature selection.

## Results
Models are evaluated using root-mean-square error (RMSE) with each training dataset. The lowest RMSE is achieved by the XGBoost algorithm with hyperparameter tuning and the Y_log_transformed dataset. The results also show that solving collinearity or multicollinearity by reducing non-independent features does not improve MLR performance. Instead, it decreases the prediction accuracy.

## Files
**data_preprocessing.py**<br>
This Python file is a data preprocessing class that includes data cleaning, encoding, and scaling.<br>
**prediction_models.py**<br>
This Python file is a prediction model class that includes eight regression models.<br>
**house_price_prediction.ipynb**<br>
This Jupyter Notebook demonstrates data preprocessing, and comparison of models and datasets.<br>

## References
[1] 2.7.3.3. Local Outlier Factor, 2.7. Novelty and Outlier Detection, Unsupervised learning, Scikit-learn User Guide. Retrieved from https://scikit-learn.org/stable/modules/outlier_detection.html#id1

[2] Cosma, S. (2015). Lecture 17: Multicollinearity[Course notes]. Carnegie Mellon University Modern Regression. Retrieved from https://www.stat.cmu.edu/~cshalizi/mreg/15/

[3] Laura, S., Robert, H., Andrew, W., Derek, Y., Iain, P. (2022). STAT 501 Lesson 12.4 - Detecting Multicollinearity Using Variance Inflation Factors[Course notes]. The Pennsylvania State University Regression Methods. Retrieved from https://online.stat.psu.edu/stat501/lesson/12/12.4