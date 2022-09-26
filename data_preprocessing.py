"""
This module is for data preprocessing.

Class:

    PreprocessData

Functions:

    find_nullvalue_cols
    cols_unique_values
    replace_with_frequent_values
    replace_with_frequent_values
    scale_numerical_values
    encode_categorical_values
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing

class PreprocessData():
    """This class is for data cleaning."""
    def __init__(self) -> None:
        pass

    def find_nullvalue_cols(self, dataframe):
        """
        This function is to output a list of column names
        that the columns in the dataframe have null values.
        """

        nullvalue_cols_list = [
            col_name for col_name in dataframe.columns if dataframe[col_name].isna().sum()>0
        ]
        full_cols_list = dataframe.columns.tolist()
        for col in full_cols_list:

            # Apart from checking if columns have null/missing values,
            # it is also better to check unique values of each column
            # in case some null values are in str type, such as "NONE", "none" or "None".
            # Above none values cannot be detected by pd.isna or pd.isnull.
            unique_value_list = dataframe[col].unique().tolist()
            if "None" and "NONE" and "none" in unique_value_list:
                nullvalue_cols_list.append(col)
            else:
                continue
        return nullvalue_cols_list

    def cols_unique_values(self, dataframe):
        """
        This function is to create a dictionary that keys are column names and
        values are columns' unique values.
        """

        unique_values = {}
        for col_name in dataframe.columns.tolist():
            col_unique_values = dataframe[col_name].unique().tolist()
            unique_values[col_name] = col_unique_values

        return unique_values

    def replace_with_frequent_values(self, dataframe, null_value_list):
        """
        This function is to fill null/missing values of columns
        with frequent values in the corresponding columns.

        Args:
        dataframe: a dataframe contains all columns in the null_value_list.
        null_value_list: a list of columns's names that have null/missing values.
        """

        for col_name in null_value_list:
            
            col_dtype = dataframe[col_name].dtype
            if col_dtype == "object":
                dataframe[col_name].fillna(
                    value=dataframe[col_name].mode()[0], inplace=True
                )
            elif col_dtype in ("float64", "int64"):
                dataframe[col_name].fillna(
                    value=dataframe[col_name].median(), inplace=True
                )

        return dataframe
    
    def replace_with_ownchoice_values(self, dataframe, null_value_list, replace_values_list):
        """
        This function is to replace null/missing values with your preferred values.

        Args:
        dataframe: a dataframe contains all columns in the null_value_list.
        null_value_list: a list of columns's names that have null/missing values.
        replace_values_list: a list of values that you want to replace the null values.
        """

        for col_name_idx, col_name in enumerate(null_value_list):
            replace_value = replace_values_list[col_name_idx]
            dataframe[col_name].replace(
                to_replace=[np.nan, None, "nan", "NAN", "None", "NONE", "none"],
                value=replace_value,
                inplace=True
            )

        return dataframe

    def scale_numerical_values(self, dataframe, scale_cols_list,
                                scale_method="standarscale", log_trans_repalcevalue=None):
        """
        This function is to scale numerical columns to standard normal distributed data.

        Args:
        scale_cols_list:a list containing columns names, which the columns need to be scaled.
        scale_method: default is "standardscale". Other options: "minmaxscale", "log_transform".
        """
        data = dataframe[scale_cols_list]
        if scale_method == "standarscale":

            scale = preprocessing.StandardScaler()
            scaled_data = scale.fit_transform(data)
            # Drop original columns and add scaled columns back to the dataframe
            dataframe = dataframe.drop(columns=scale_cols_list)
            dataframe_index_list = dataframe.index.to_list()
            scaled_cols_df = pd.DataFrame(data=scaled_data,
                                        index=dataframe_index_list,
                                        columns=scale_cols_list)
            dataframe = pd.concat([dataframe, scaled_cols_df], axis=1)

        elif scale_method == "minmaxscale":

            scale = preprocessing.MinMaxScaler()
            scaled_data = scale.fit_transform(data)
            # drop original columns and add scaled columns back to the dataframe
            dataframe = dataframe.drop(columns=scale_cols_list)
            dataframe_index_list = dataframe.index.to_list()
            scaled_cols_df = pd.DataFrame(data=scaled_data,
                                        index=dataframe_index_list,
                                        columns=scale_cols_list)
            dataframe = pd.concat([dataframe, scaled_cols_df], axis=1)

        elif scale_method == "log_transform":

            transformed_numpy = np.log(
                dataframe[scale_cols_list].replace(0, log_trans_repalcevalue)
            )
            dataframe = dataframe.drop(columns=scale_cols_list)
            dataframe_index_list = dataframe.index.to_list()
            scaled_cols_df = pd.DataFrame(data=transformed_numpy,
                                        index=dataframe_index_list,
                                        columns=scale_cols_list)
            dataframe = pd.concat([dataframe, scaled_cols_df], axis=1)

        return dataframe

    def encode_categorical_values(
        self, dataframe, encoding_cols_list, encoding_method="onehot",
        ordinal_encode=False, ordinal_cate_list=None
    ):
        """
        This function is to encode categorical data columns.

        Args:
        encoding_cols_list: a list of objective categorical data columns' names.
                            If it is ordinal encoding or label encoding,
                            then encoding method for all columns needs to be the same.
                            It cannot be mixing ordinal encoding and labele encoding together.
        encoding_method: default is onehot encoding. Other options: "ordinal".
        ordinal_encode: indicating if the ordinal encoding is acting as label encoding or not.
                        By default is False meaning only label encodeing.
                        When True, it is ordinal encoding.
        ordinal_cate_list: a list of strings or numeric values representing the ordered categories
        """
        data = dataframe[encoding_cols_list]
        if encoding_method == "onehot":

            encoder = preprocessing.OneHotEncoder()
            # this is a compressed matrix. Need to convert it to a numpy array
            encoding_data = encoder.fit_transform(data)
            encoding_data = encoding_data.toarray()
            # get new columns' names after one-hot encoding and convert it to a list
            encoded_cols_names = encoder.get_feature_names_out(encoding_cols_list)
            encoded_cols_names = encoded_cols_names.tolist()
            # drop original columns and add scaled columns back to the dataframe
            dataframe.drop(columns=encoding_cols_list, inplace=True)
            dataframe_index_list = dataframe.index.to_list()
            encoded_cols_df = pd.DataFrame(data=encoding_data,
                                            index=dataframe_index_list,
                                            columns=encoded_cols_names)
            dataframe = pd.concat([dataframe, encoded_cols_df], axis=1)

        elif encoding_method == "ordinal":
            if ordinal_encode == True:
                encoder = preprocessing.OrdinalEncoder(categories=ordinal_cate_list)
            else:
                encoder = preprocessing.OrdinalEncoder()
            encoding_data = encoder.fit_transform(data)
            # drop original columns and add scaled columns back to the dataframe
            dataframe.drop(columns=encoding_cols_list, inplace=True)
            dataframe_index_list = dataframe.index.to_list()
            encoded_cols_df = pd.DataFrame(data=encoding_data,
                                            index=dataframe_index_list,
                                            columns=encoding_cols_list)
            dataframe = pd.concat([dataframe, encoded_cols_df], axis=1)

        return dataframe
