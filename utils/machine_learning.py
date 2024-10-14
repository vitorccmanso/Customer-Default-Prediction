import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score, roc_auc_score, recall_score, roc_curve, confusion_matrix, precision_score, accuracy_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
import pickle

mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
uri = os.environ.get("uri")

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

class ModelTraining:
    """
    A class for training machine learning models, evaluating their performance, and saving the best one

    Methods:
    - __init__: Initializes the ModelTraining object
    - save_model: Saves the specified model to a pkl file
    - initiate_model_trainer: Initiates the model training process and evaluates multiple models
    - evaluate_models: Evaluates multiple models using random search cross-validation and logs the results with MLflow
    """
    def __init__(self):
        pass

    def save_model(self, model_name, version, save_folder, save_filename):
        """
        Save the specified model to a pkl file

        Parameters:
        - model_name (str): The name of the model to save
        - version (int): The version of the model to save
        - save_folder (str): The folder path where the model will be saved
        - save_filename (str): The filename for the pkl file
        """
        mlflow.set_tracking_uri(uri)
        client = mlflow.tracking.MlflowClient(tracking_uri=uri)

        # Get the correct version of the registered model
        model_versions = client.search_model_versions(f"name='{model_name}'")
        model_versions_sorted = sorted(model_versions, key=lambda v: int(v.version))
        requested_version = model_versions_sorted[version - 1]
        
        # Construct the logged model path
        run_id = requested_version.run_id
        artifact_path = requested_version.source.split("/")[-1]
        logged_model = f"runs:/{run_id}/{artifact_path}"

        # Load the model from MLflow and saves it to a pkl file
        loaded_model = mlflow.sklearn.load_model(logged_model)
        file_path = os.path.join(save_folder, f"{save_filename}.pkl")
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(loaded_model, f)

    def initiate_model_trainer(self, train_test, experiment_name, scoring, refit, use_smote=False):
        """
        Initiates the model training process

        Parameters:
        - train_test (tuple): A tuple containing the train-test split data in the format (X_train, y_train, X_test, y_test)
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - scoring (list): The scoring metrics used for evaluating the models
        - refit (str): The metric to refit the best model
        - use_smote (bool): A boolean indicating whether to apply SMOTE for balancing the classes. Default is False

        Returns:
        - dict: A dictionary containing the evaluation report for each model
        """
        mlflow.set_tracking_uri(uri)
        X_train, y_train, X_test, y_test = train_test
        
        models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        
        params = {
            "Logistic Regression": {
                "solver": ["liblinear", "lbfgs"],
                "penalty":["l2", "l1", "elasticnet", None], 
                "C":[1.5, 1, 0.5, 0.1]
            },
            "Random Forest":{
                "criterion":["gini", "entropy", "log_loss"],
                "max_features":["sqrt", "log2"],
                "n_estimators": [25, 50, 100],
                "max_depth": [10, 20, 30, 50]
            },
            "Gradient Boosting":{
                "loss":["log_loss", "exponential"],
                "max_features":["sqrt", "log2"],
                "n_estimators": [25, 50, 100],
                "max_depth": [10, 20, 30, 50],
                "learning_rate": [0.001, 0.01, 0.1],
            },
        }
        
        model_report = self.evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, 
                                           models=models, params=params, experiment_name=experiment_name, 
                                           scoring=scoring, refit=refit, use_smote=use_smote)
        
        return model_report

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params, experiment_name, scoring, refit, use_smote):
        """
        Evaluates multiple models using random search cross-validation and logs the results with MLflow

        Parameters:
        - X_train (array-like): Features of the training data
        - y_train (array-like): Target labels of the training data
        - X_test (array-like): Features of the testing data
        - y_test (array-like): Target labels of the testing data
        - models (dict): A dictionary containing the models to be evaluated
        - params (dict): A dictionary containing the hyperparameter grids for each model
        - experiment_name (str): Name of the MLflow experiment where the results will be logged
        - scoring (list): The scoring metrics used for evaluating the models
        - refit (str): The metric to refit the best model
        - use_smote (bool): A boolean indicating whether to apply SMOTE for balancing the classes

        Returns:
        - dict: A dictionary containing the evaluation report for each model.
        """
        mlflow.set_experiment(experiment_name)
        report = {}
        if use_smote:
            # Apply SMOTE only to the training data
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
        for model_name, model in models.items():
            with mlflow.start_run(run_name=model_name):
                param = params[model_name]
                if model_name != "Gradient Boosting":
                    param["class_weight"] = [None] if use_smote else ["balanced"]

                rs = RandomizedSearchCV(model, param, cv=5, scoring=scoring, refit=refit, random_state=42)
                search_result = rs.fit(X_train, y_train)
                model = search_result.best_estimator_
                y_pred = model.predict(X_test)
                mlflow.set_tags({"model_type": f"{model_name}-{experiment_name}", "smote_applied": use_smote})

                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred)
                roc = roc_curve(y_test, model.predict_proba(X_test)[:,1])
                recall = recall_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Log metrics to MLflow
                mlflow.log_params(search_result.best_params_)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("roc_auc_score", roc_auc)
                mlflow.log_metric("recall_score", recall)
                mlflow.log_metric("precision_score", precision)
                mlflow.log_metric("accuracy_score", accuracy)
                mlflow.sklearn.log_model(model, model_name, registered_model_name=f"{model_name} - {experiment_name}")
                
                # Store the model for visualization
                report[model_name] = {"model": model, "y_pred": y_pred, "roc_auc_score": roc_auc, "roc_curve": roc}      
        return report


class MetricsVisualizations:
    """
    A class for visualizing model evaluation metrics and results

    Attributes:
    - models (dict): A dictionary containing the trained models, with metrics and predictions for each model

    Methods:
    - __init__: Initializes the MetricsVisualizations object with a dictionary of models
    - create_subplots: Creates a figure and subplots with common settings
    - visualize_roc_curves: Visualizes ROC curves for each model, showing the trade-off between true positive and false positive rates
    - visualize_confusion_matrix: Visualizes confusion matrices for each model, displaying absolute and relative values
    - plot_precision_recall_threshold: Plots precision and recall against thresholds for each model, providing insight into performance at different classification thresholds
    - plot_feature_importance: Plots feature importance for each model using permutation importance
    """
    def __init__(self, models):
        """
        Initializes the MetricsVisualizations object with a dictionary of models

        Parameters:
        - models (dict): A dictionary containing the trained models, with metrics and predictions for each model
        """
        self.models = models

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple): Figure size. Default is (18, 12)
        
        Returns:
        - fig (matplotlib.figure.Figure): The figure object
        - ax (numpy.ndarray): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def visualize_roc_curves(self):
        """
        Visualizes ROC curves for each model
        """
        plt.figure(figsize=(12, 6))
        plt.plot([0, 1], [0, 1], "r--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")

        for model_name, model_data in self.models.items():
            model_roc_auc = model_data["roc_auc_score"]
            fpr, tpr, thresholds = model_data["roc_curve"]
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {model_roc_auc:.3f})")
        plt.legend()
        plt.show()

    def visualize_confusion_matrix(self, y_test, rows, columns):
        """
        Visualizes confusion matrices for each model

        Parameters:
        - y_test (array-like): True labels of the test data.
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(14, 10))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred = model_data["y_pred"]
            matrix = confusion_matrix(y_test, y_pred)

            # Plot the first heatmap
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[i * 2])
            ax[i * 2].set_title(f"Confusion Matrix: {model_name} - Absolute Values")
            ax[i * 2].set_xlabel("Predicted Values")
            ax[i * 2].set_ylabel("Observed values")

            # Plot the second heatmap
            sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[i * 2 + 1])
            ax[i * 2 + 1].set_title(f"Relative Values")
            ax[i * 2 + 1].set_xlabel("Predicted Values")
            ax[i * 2 + 1].set_ylabel("Observed values")

        fig.tight_layout()
        plt.show()

    def plot_precision_recall_threshold(self, y_test, X_test, rows, columns):
        """
        Plots precision and recall vs thresholds for each model

        Parameters:
        - y_test (array-like): True labels of the test data
        - X_test (pandas.DataFrame): Features of the test data
        - rows (int): Number of rows for subplots
        - columns (int): Number of columns for subplots
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 6))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            y_pred_prob = model_data["model"].predict_proba(X_test)[:,1]
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_prob)

            # Plot Precision-Recall vs Thresholds for each model
            ax[i].set_title(f"Precision X Recall vs Thresholds - {model_name}")
            ax[i].plot(thresholds, precisions[:-1], "b--", label="Precision")
            ax[i].plot(thresholds, recalls[:-1], "g-", label="Recall")
            ax[i].plot([0.5, 0.5], [0, 1], "k--")
            ax[i].set_ylabel("Score")
            ax[i].set_xlabel("Threshold")
            ax[i].legend(loc="center left")

            # Annotate precision and recall at 0.5 threshold
            y_pred = model_data["y_pred"]
            metrics = precision_recall_fscore_support(y_test, y_pred)
            precision = metrics[0][1]
            recall = metrics[1][1]
            ax[i].plot(0.5, precision, "or")
            ax[i].annotate(f"{precision:.2f} precision", (0.51, precision))
            ax[i].plot(0.5, recall, "or")
            ax[i].annotate(f"{recall:.2f} recall", (0.51, recall))
            ax[i].annotate("0.5 threshold", (0.39, -0.04))

        fig.tight_layout()
        plt.show()

    def plot_feature_importance(self, y_test, X_test, metric, rows, columns):
        """
        Plots feature importance for each model using permutation importance

        Parameters:
        - y_test (array-like): True labels of the test data
        - X_test (DataFrame): Features of the test data, where each column represents a feature
        - metric (str): The scoring metric used for evaluating feature importance (e.g., "accuracy", "f1", etc.)
        - rows (int): Number of rows for the subplot grid
        - columns (int): Number of columns for the subplot grid
        """
        fig, ax = self.create_subplots(rows, columns, figsize=(16, 6))
        for i, (model_name, model_data) in enumerate(self.models.items()):
            # Calculate and sort permutation importances
            result = permutation_importance(model_data["model"], X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring=metric)
            sorted_importances_idx = result["importances_mean"].argsort()[::-1]

            # Select top 5 features
            top_features_idx = sorted_importances_idx[:5][::-1]
            top_features = X_test.columns[top_features_idx]
            importances = pd.DataFrame(result.importances[top_features_idx].T, columns=top_features)

            # Plot boxplot of feature importances
            box = importances.plot.box(vert=False, whis=10, ax=ax[i])
            box.set_title(f"Top 5 Feature Importance - {model_name}")
            box.axvline(x=0, color="k", linestyle="--")
            box.set_xlabel(f"Decay in {metric}")
            box.figure.tight_layout()

        fig.tight_layout()
        plt.show()

def compare_confusion_matrices(y_test, X_test, model_dicts_with_metrics, thresholds, rows, columns):
    """
    Compare confusion matrices for multiple models and thresholds

    Parameters:
    - y_test (array-like): True labels of the test data
    - X_test (DataFrame): Features of the test data, where each column represents a feature
    - model_dicts_with_metrics (list of tuples): List containing tuples with the following structure:
        - model_dict (dict): A dictionary where keys are model names and values contain model data
        - type_of_balancing (str): The type of balancing technique used to train the model (e.g., "Smote")
        - performance_metric (str): The performance metric used to evaluate the model (e.g., "accuracy")
    - thresholds (dict): A dictionary mapping model identifiers (constructed from model name, balancing, and metric) to their corresponding thresholds.
    - rows (int): Number of rows for the subplot grid
    - columns (int): Number of columns for the subplot grid
    """
    fig, ax = MetricsVisualizations(model_dicts_with_metrics).create_subplots(rows, columns, figsize=(14, 10))
    plot_index = 0

    # Track which models have already been processed
    processed_models = set()

    # Iterate through the provided model dictionaries with their types and metrics
    for model_dict, type_of_balancing, performance_metric in model_dicts_with_metrics:
        for model_name, model_data in model_dict.items():
            model_identifier = f"{model_name} - {type_of_balancing} - {performance_metric}"
            # Skip if this model has already been processed
            if model_identifier in processed_models:
                continue 
            processed_models.add(model_identifier)
            # Get the model and make prediction
            model = model_data["model"]

            # Get the threshold for the current model
            model_threshold = thresholds.get(model_identifier, None)
            y_pred = model_data["y_pred"]
            matrix = confusion_matrix(y_test, y_pred)

            # Skip if threshold is None or calculate y_pred_adjusted based on the threshold found
            if model_threshold is None:
                continue
            elif model_threshold != 0.5:
                y_pred_prob = model.predict_proba(X_test)[:, 1]
                y_pred_adjusted = (y_pred_prob >= model_threshold).astype(int)
                matrix = confusion_matrix(y_test, y_pred_adjusted)

            # Plot the first heatmap (absolute values)
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax[plot_index])
            ax[plot_index].set_title(f"{model_identifier} - Absolute Values (Threshold = {model_threshold:.2f})")
            ax[plot_index].set_xlabel("Predicted Values")
            ax[plot_index].set_ylabel("Actual Values")
            plot_index += 1

            # Plot the second heatmap (relative values)
            sns.heatmap(matrix / np.sum(matrix), annot=True, fmt=".2%", cmap="Blues", ax=ax[plot_index])
            ax[plot_index].set_title("Relative Values")
            ax[plot_index].set_xlabel("Predicted Values")
            ax[plot_index].set_ylabel("Actual Values")
            plot_index += 1

            # Stop if we have filled all subplots
            if plot_index >= rows * columns:
                break

    fig.tight_layout()
    plt.show()