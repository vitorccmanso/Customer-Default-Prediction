import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
import pickle

class DataPreprocess:
    """
    A class for preprocessing data, including feature engineering, transformation, and splitting data into train-test sets

    Methods:
    - __init__: Initializes the DataPreprocess object
    - save_preprocessor: Saves the preprocessor object to a file
    - load_preprocessor: Loads the preprocessor object from a file
    - get_feature_names: Retrieves feature names after applying transformations in the preprocessor pipeline
    - preprocessor: Creates and returns a preprocessor pipeline for data preprocessing
    - preprocess_new_data: Preprocesses new input data, applying transformations and feature engineering
    - preprocess_data: Preprocesses data for training, applying transformations, feature engineering, and splitting into train-test sets
    """
    def __init__(self):
        pass

    def save_preprocessor(self, preprocessor):
        """
        Saves the preprocessor object to a file

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object to be saved
        """
        if not os.path.exists("../artifacts"):
            os.makedirs("../artifacts")
        with open("../artifacts/preprocessor.pkl", "wb") as f_all:
            pickle.dump(preprocessor, f_all)
    
    def load_preprocessor(self):
        """
        Loads the preprocessor object from a file

        Returns:
        - preprocessor: The loaded preprocessor object
        """
        with open("../artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return preprocessor

    def get_feature_names(self, preprocessor, no_transformation_cols, cbrt_cols, one_hot_cols):
        """
        Retrieves the feature names after preprocessing is applied

        Parameters:
        - preprocessor (sklearn.pipeline.Pipeline): The preprocessor object used for transformations
        - no_transformation_cols (list): List of numeric columns without transformations
        - cbrt_cols (list): List of columns where cubic root transformation is applied
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - feature_names (list): List of feature names after preprocessing
        """
        numeric_features = no_transformation_cols + cbrt_cols
        one_hot_features = list(preprocessor.named_transformers_["cat_onehot"]["onehot"].get_feature_names_out(one_hot_cols))
        feature_names = numeric_features + one_hot_features
        return feature_names

    def preprocessor(self, no_transformation_cols, cbrt_cols, one_hot_cols):
        """
        Creates and returns a preprocessor pipeline for data preprocessing

        Parameters:
        - no_transformation_cols (list): List of numeric columns without transformations
        - cbrt_cols (list): List of columns where cubic root transformation is applied
        - one_hot_cols (list): List of categorical columns for which OneHotEncoding is applied

        Returns:
        - preprocessor (sklearn.compose.ColumnTransformer): Preprocessor pipeline for data preprocessing
        """
        # Define transformers for numeric columns
        cubic_transformer = Pipeline(steps=[
            ("cbrt_transformation", FunctionTransformer(np.cbrt, validate=True)),
            ("scaler", RobustScaler())
        ])

        #Define transformer for categorical columns
        categorical_transformer = Pipeline(steps=[
            ("onehot", OneHotEncoder())
        ])

        # Combine transformers for numeric and categorical columns
        transformers=[
            ("num_only_scale", RobustScaler(), no_transformation_cols),
            ("num_cbrt", cubic_transformer, cbrt_cols),
            ("cat_onehot", categorical_transformer, one_hot_cols)
        ]

        preprocessor = ColumnTransformer(transformers=transformers, verbose_feature_names_out=False, remainder="drop")
        return preprocessor

    def preprocess_new_data(self, data):
        """
        Preprocesses new input data, including feature engineering and transformations

        Parameters:
        - data (pandas.DataFrame): The input DataFrame containing raw data

        Returns:
        - X (pandas.DataFrame): Preprocessed data ready for model inference
        """
        data["SAFRA_REF"] = pd.to_datetime(data["SAFRA_REF"])
        data["DATA_VENCIMENTO"] = pd.to_datetime(data["DATA_VENCIMENTO"])
        data["DATA_EMISSAO_DOCUMENTO"] = pd.to_datetime(data["DATA_EMISSAO_DOCUMENTO"])
        
        # Replace "Unknown" in "DATA_CADASTRO" with corresponding "DATA_EMISSAO_DOCUMENTO" values
        data["DATA_CADASTRO"] = data["DATA_CADASTRO"].replace("Unknown", np.nan)
        data["DATA_CADASTRO"] = data["DATA_CADASTRO"].fillna(data["DATA_EMISSAO_DOCUMENTO"])
        data["DATA_CADASTRO"] = pd.to_datetime(data["DATA_CADASTRO"])

        # Feature engineering
        data["month_vencimento"] = data["DATA_VENCIMENTO"].dt.month
        data["days_until_due"] = (data["DATA_VENCIMENTO"] - data["SAFRA_REF"]).dt.days
        data["days_since_registration"] = (data["DATA_EMISSAO_DOCUMENTO"] - data["DATA_CADASTRO"]).dt.days

        # Drop unnecessary columns
        data = data.drop(columns=["ID_CLIENTE", "SAFRA_REF", "DATA_EMISSAO_DOCUMENTO", "DATA_VENCIMENTO", "DATA_CADASTRO", "FLAG_PF"])

        no_transformation_cols = ["TAXA", "month_vencimento", "NO_FUNCIONARIOS", "days_since_registration"]
        cbrt_cols = ["VALOR_A_PAGAR", "RENDA_MES_ANTERIOR", "days_until_due"]
        one_hot_cols = data.select_dtypes(include="object").columns
        
        preprocessor = self.load_preprocessor()
        X = preprocessor.transform(data)
        feature_names = self.get_feature_names(preprocessor, no_transformation_cols, cbrt_cols, one_hot_cols)
        X = pd.DataFrame(X, columns=feature_names)
        return X

    def preprocess_data(self, data, target_name=None, test_size=None, test_data=None):
        """
        Preprocesses the input data, including feature engineering, transformation, and splitting into train-test sets

        Parameters:
        - data (pandas.DataFrame): The input DataFrame containing raw data
        - target_name (str): The name of the target variable to predict
        - test_size (float): Proportion of the dataset to include in the test split
        - test_data (bool): Flag indicating if the input data is test data for inference. If true, the function returns only the test dataframe

        Returns:
        - X (pandas.Dataframe): Preprocessed test data for predictions
        - X_train (pandas.DataFrame): Preprocessed features for training set
        - X_test (pandas.DataFrame): Preprocessed features for testing set
        - y_train (pandas.Series): Target labels for training set
        - y_test (pandas.Series): Target labels for testing set
        """
        # Process the test dataset
        if test_data:
            X = self.preprocess_new_data(data=data)
            return X

        # Drop the target and specify columns needing log transformation, square root transformation
        data_process = data.drop(columns=[target_name])
        no_transformation_cols = ["TAXA", "month_vencimento", "NO_FUNCIONARIOS", "days_since_registration"]
        cbrt_cols = ["VALOR_A_PAGAR", "RENDA_MES_ANTERIOR", "days_until_due"]
        one_hot_cols = data_process.select_dtypes(include="object").columns
        
        # Build preprocessor
        preprocessor = self.preprocessor(no_transformation_cols, cbrt_cols, one_hot_cols)

        # Fit and transform data
        data_preprocessed = preprocessor.fit_transform(data_process)

        feature_names = self.get_feature_names(preprocessor, no_transformation_cols, cbrt_cols, one_hot_cols)
        data_preprocessed = pd.DataFrame(data_preprocessed, columns=feature_names)

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data[target_name], test_size=test_size, stratify=data[target_name], shuffle=True, random_state=42)

        # Save preprocessor if not already saved
        self.save_preprocessor(preprocessor)

        return X_train, X_test, y_train, y_test