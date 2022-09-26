"""
This module is to create models for predicting continuous dependent variables,
as well as outputing comparison result of using different input datasets in each model.

Class:

    RegressionModels

Functions:

    linear_regression
    kneighbor_regression
    svr_regression
    elasticnet_regression
    random_forest_regression
    gradientboosting_regression
    xgboosting_regression
    stacking_regression
"""
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.ensemble import StackingRegressor

class RegressionModels:
    """
    This class is to create models for predicting continuous dependent variables
    and outputing comparison result of using different input datasets for each model.
    log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
    """
    def __init__(self) -> None:
        pass

    def linear_regression(self, dataset, log_tranform_y=False, output_predict=False):
        """
        Create three linear models.
        Output root mean absolute error of three models.
        Output prediction is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: default is False.
                        When it is True,
                        the function will output prediction dataset's result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Ordinary least squares linear regression
        ord_reg = LinearRegression()
        ord_reg.fit(x_train, y_train)
        ord_test_pred = np.log(ord_reg.predict(x_test))
        ord_pred = ord_reg.predict(x_predict)

        # Linear model with L1 regularizer
        # Increase alpha to 3 and max_inter to 2000 is for dealing with convergence problem
        # due to poor input data quality.
        lasso_reg = Lasso(alpha=3, max_iter=3000, random_state=10)
        lasso_reg.fit(x_train, y_train)
        lasso_test_pred = np.log(lasso_reg.predict(x_test))
        lasso_pred = lasso_reg.predict(x_predict)

        # Linear model with L2 regularizer
        # Increase alpha to 3 and max_inter to 2000 is for dealing with convergence problem
        # due to poor input data quality.
        ridge_reg = Ridge(alpha=3, max_iter=3000, random_state=11)
        ridge_reg.fit(x_train, y_train)
        ridge_test_pred = np.log(ridge_reg.predict(x_test))
        ridge_pred = ridge_reg.predict(x_predict)

        y_test_loged = np.log(y_test)

        if log_tranform_y == True:
            ord_test_pred = ord_reg.predict(x_test)
            lasso_test_pred = lasso_reg.predict(x_test)
            ridge_test_pred = ridge_reg.predict(x_test)
            ord_pred = np.exp(ord_pred)
            lasso_pred = np.exp(lasso_pred)
            ridge_pred = np.exp(ridge_pred)
    
        ord_rmse = np.sqrt(mean_squared_error(y_test_loged, ord_test_pred))
        lasso_rmse = np.sqrt(mean_squared_error(y_test_loged, lasso_test_pred))
        ridge_rmse = np.sqrt(mean_squared_error(y_test_loged, ridge_test_pred))

        rmse_results = [ord_rmse, lasso_rmse, ridge_rmse]
        pred_results = [ord_pred, lasso_pred, ridge_pred]

        if output_predict == True:

            return rmse_results, pred_results
        else:

            return rmse_results

    def knn_regression(self, dataset, log_transform_y=False, output_predict=False):
        """
        Create a kneighbor regreassion.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        kn_reg = KNeighborsRegressor(n_neighbors=3)
        kn_reg.fit(x_train, y_train)
        kn_test_pred = np.log(kn_reg.predict(x_test))
        kn_pred = kn_reg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            kn_test_pred = kn_reg.predict(x_test)
            kn_pred = np.exp(kn_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, kn_test_pred))

        if output_predict == True:
            return rmse, kn_pred
        else:
            return rmse

    def gridsearch_svm_regression(
        self, dataset, svr_parameters, log_transform_y=False, output_predict=False
    ):
        """
        Create a svr regression model.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        svr_reg = svm.SVR()
        gridsearch_svr_reg = GridSearchCV(svr_reg, svr_parameters)
        gridsearch_svr_reg.fit(x_train, y_train)
        gridsearch_svr_test_pred = np.log(gridsearch_svr_reg.predict(x_test))
        gridsearch_svr_pred = gridsearch_svr_reg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            gridsearch_svr_test_pred = gridsearch_svr_reg.predict(x_test)
            gridsearch_svr_pred = np.exp(gridsearch_svr_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, gridsearch_svr_test_pred))
        best_param = gridsearch_svr_reg.best_estimator_

        if output_predict == True:
            return rmse, gridsearch_svr_pred, best_param
        else:
            return rmse

    def elasticnet_regression(self, dataset, log_transform_y=False, output_predict=False):
        """
        Create a ElasticNet regression model.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        elanet_reg = ElasticNet()
        elanet_reg.fit(x_train, y_train)
        elanet_test_pred = np.log(elanet_reg.predict(x_test))
        elanet_pred = elanet_reg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            elanet_test_pred = elanet_reg.predict(x_test)
            elanet_pred = np.exp(elanet_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, elanet_test_pred))

        if output_predict == True:
            return rmse, elanet_pred
        else:
            return rmse

    def gridsearch_random_forest_regression(
        self, dataset, randomforest_parameters, log_transform_y=False, output_predict=False
    ):
        """
        Create a random forest regreassion model.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        rf_reg = RandomForestRegressor(random_state=15)
        gridsearch_rf_reg = GridSearchCV(rf_reg, randomforest_parameters)
        gridsearch_rf_reg.fit(x_train, y_train)
        gridsearch_rf_test_pred = np.log(gridsearch_rf_reg.predict(x_test))
        gridsearch_rf_pred = gridsearch_rf_reg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            gridsearch_rf_test_pred = gridsearch_rf_reg.predict(x_test)
            gridsearch_rf_pred = np.exp(gridsearch_rf_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, gridsearch_rf_test_pred))
        best_param = gridsearch_rf_reg.best_estimator_

        if output_predict == True:
            return rmse, gridsearch_rf_pred, best_param
        else:
            return rmse

    def gridsearch_gradientbosting_regression(
        self, dataset, gradientboosting_parameters, log_transform_y=False, output_predict=False
    ):
        """
        Create a gradient boosting regreassion with grid search hyperparameter tuning to obtain
        the optimal result and output the rmse score.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        gradient_parameters: a dictionary of tuning hyperparameters.
                    e.g. {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]}
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input in gradient boosting model requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        grad_reg = GradientBoostingRegressor(random_state=12)
        gridsearch_grad = GridSearchCV(grad_reg, gradientboosting_parameters)
        gridsearch_grad.fit(x_train, y_train)
        gridsearch_grad_test_pred = np.log(gridsearch_grad.predict(x_test))
        gridsearch_grad_pred = gridsearch_grad.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            gridsearch_grad_test_pred = gridsearch_grad.predict(x_test)
            gridsearch_grad_pred = np.exp(gridsearch_grad_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, gridsearch_grad_test_pred))
        best_param = gridsearch_grad.best_estimator_

        if output_predict == True:
            return rmse, gridsearch_grad_pred, best_param
        else:
            return rmse

    def gridsearch_xgboosting_regression(
        self, dataset, xgboosting_parameters=None, log_transform_y=False, output_predict=False
    ):
        """
        Create a extrem gradient boosting regreassion mdoel
        with grid search hyperparameter tuning to obtain
        the optimal result and output the rmse score.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        xgboosting_parameters: a dictionary of tuning xgboosting hyperparameters.
                    e.g. {"eta": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]}
                    All datasets use the same parameters for tuning.
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset

        xg_reg = xgb.XGBRegressor()
        gridsearch_xg = GridSearchCV(xg_reg, xgboosting_parameters)
        gridsearch_xg.fit(x_train, y_train)
        gridsearch_xg_test_pred = np.log(gridsearch_xg.predict(x_test))
        gridsearch_xg_pred = gridsearch_xg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            gridsearch_xg_test_pred = gridsearch_xg.predict(x_test)
            gridsearch_xg_pred = np.exp(gridsearch_xg.predict(x_predict))

        rmse = np.sqrt(mean_squared_error(y_test_log, gridsearch_xg_test_pred))
        best_params = gridsearch_xg.best_estimator_.get_params()

        if output_predict == True:
            return rmse, gridsearch_xg_pred, best_params
        else:
            return rmse

    def stacking_regression(self, dataset, log_transform_y=False, output_predict=False):
        """
        Create a stacked generalization that combines multiple regreassion mdoels
        to obtain the optimal result and output the rmse score.
        Output prediction data's target result is optional.

        Args:
        dataset: a list of input and target datasets.
                Please follow this order [x_train, x_test, x_predict, y_train, y_test].
        gradientboosting_parameters: a dictionary of tuning gradient boosting hyperparameters.
                    e.g. {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]}
                    All datasets use the same parameters for tuning.
        xgboosting_parameters: a dictionary of tuning xgboosting hyperparameters.
                    e.g. {"eta": [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]}
                    All datasets use the same parameters for tuning.
        log_transform_y: a list of boolen values.
                        Each boolen value indicates whether the input y_train is log transformed.
        output_predict: by default is False.
                        When it is True,
                        the function will also output the prediction data's target result.
        """
        x_train, x_test, x_predict, y_train, y_test = dataset
        # Shape of target input requires (n_samples, ).
        y_train = y_train.reshape(y_train.shape[0], )
        y_test = y_test.reshape(y_test.shape[0], )

        estimator = [
            (
                "gridsearch_gradientboosting",
                GradientBoostingRegressor(
                    learning_rate=0.1, min_samples_leaf=16, subsample=0.7, random_state=19
                )
            ),
            (
                "random_forest",
                RandomForestRegressor(
                    bootstrap=False, min_samples_leaf=2, random_state=21
                )
            ),
            (
                "knr",
                KNeighborsRegressor(n_neighbors=3)
            ),
            (
                "xgb",
                xgb.XGBRegressor(
                    max_depth=6, learning_rate=0.1, eta=0.1, subsample=0.7, random_state=26
                )
            )
        ]

        final_estimator = Ridge(random_state=21)

        stack_reg = StackingRegressor(estimators=estimator, final_estimator=final_estimator)
        stack_reg.fit(x_train, y_train)
        stack_reg_test_pred = np.log(stack_reg.predict(x_test))
        stack_reg_pred = stack_reg.predict(x_predict)

        y_test_log = np.log(y_test)

        if log_transform_y == True:
            stack_reg_test_pred = stack_reg.predict(x_test)
            stack_reg_pred = np.exp(stack_reg_pred)

        rmse = np.sqrt(mean_squared_error(y_test_log, stack_reg_test_pred))

        if output_predict == True:
            return rmse, stack_reg_pred
        else:
            return rmse
