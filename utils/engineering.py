import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats
from scipy.stats import skew, mode

def anova_test(data, target_col, features):
    """
    Performs ANOVA test on multiple features against a binary target column.
    
    Parameters:
    - data (pd.DataFrame): The input DataFrame.
    - target_col (str): The name of the target binary column.
    - features (list): List of feature names to test.
    
    Returns:
    - pd.DataFrame: A DataFrame with feature names and their corresponding p-values.
    """
    anova_results = {}
    for feature in features:
        # Perform ANOVA test for each feature
        f_value, p_value = stats.f_oneway(*(data[data[target_col] == value][feature] for value in data[target_col].unique()))
        anova_results[feature] = p_value
    
    # Convert the results dictionary to a DataFrame
    anova_df = pd.DataFrame(anova_results.items(), columns=["Feature", "p-value"])
    return anova_df

def saving_dataset(data, save_folder, save_filename):
    """
    Saves the dataset, and creates the specified folder if it doesn"t exist

    Parameters:
    - data: DataFrame containing the original dataset
    - save_folder: Folder path where the datasets will be saved
    - save_filename: Base filename for the saved datasets
    """
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    original_save_path = os.path.join(save_folder, f"{save_filename}.csv")
    data.to_csv(original_save_path, index=False)

class Plots:
    """
    A class for plotting data distributions and correlations

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Plots object with a DataFrame
    - plot_transformed_distributions: Plot distributions of a numeric column and its transformed versions
    - plot_corr: Plot a heatmap of the correlation matrix for numerical columns in the DataFrame
    """
    def __init__(self, data):
        """
        Initialize the Plots object with a DataFrame

        Parameters:
        - data: DataFrame containing the data
        """
        self.data = data

    def plot_transformed_distributions(self, col):
        """
        Plot distributions of a numeric column and its transformed versions

        Parameters:
        - col (str): Name of the column to plot
        """
        data = self.data[col]
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))
        ax = ax.ravel()

        # Plot original distribution
        sns.histplot(data, kde=True, ax=ax[0])
        ax[0].set_title(f"Original Distribution\n Skewness:{round(skew(data), 2)}")
        self.plot_legends(ax, 0, data)

        # Check for non-positive values before log transformation
        log_transformed = None
        has_non_positive = np.any(data < 0)

        # Plot log transformation if valid
        if not has_non_positive:
            log_transformed = np.log1p(data)
            sns.histplot(log_transformed, kde=True, ax=ax[1])
            ax[1].set_title(f"Log Transformation\n Skewness: {round(skew(log_transformed), 2)}")
            self.plot_legends(ax, 1, log_transformed)
        else:
            ax[1].text(0.5, 0.5, "Log Transformation\n Not Applicable", fontsize=15, ha="center", va="center")
            ax[1].set_title("Log Transformation\n Not Applicable")

        # Apply cubic root transformation
        sqrt_transformed = np.cbrt(data)
        sns.histplot(sqrt_transformed, kde=True, ax=ax[2])
        ax[2].set_title(f"Cubic Transformation\n Skewness: {round(skew(sqrt_transformed), 2)}")
        self.plot_legends(ax, 2, sqrt_transformed)

        skewness_values = [skew(data)]
        if log_transformed is not None:
            skewness_values.append(skew(log_transformed))
        skewness_values.append(skew(sqrt_transformed))
        closest_to_zero_idx = np.argmin(np.abs(skewness_values))

        # Plot boxplot of the transformation with the lowest skewness
        if closest_to_zero_idx == 0:
            sns.boxplot(x=data, ax=ax[3])
        elif closest_to_zero_idx == 1 and log_transformed is not None:
            sns.boxplot(x=log_transformed, ax=ax[3])
        else:
            sns.boxplot(x=sqrt_transformed, ax=ax[3])
        ax[3].set_title(f"Boxplot: Best Distribution")
        plt.tight_layout()
        plt.show()

    def plot_legends(self, ax, pos, data):
        """
        Plot legends on a given axis for a specific plot position

        Parameters:
        - ax: Axis object to plot on
        - pos (int): Position in the subplot grid
        - data: Data for which legends are plotted
        """
        ax[pos].axvline(data.mean(), color="r", linestyle="--", label="Mean: {:.2f}".format(data.mean()))
        ax[pos].axvline(mode(data)[0], color="g", linestyle="--", label="Mode: {:.2f}".format(mode(data)[0]))
        ax[pos].axvline(data.median(), color="b", linestyle="--", label="Median: {:.2f}".format(data.median()))
        ax[pos].legend()

    def plot_corr(self, method, figsize):
        """
        Plots a heatmap of the correlation matrix for numerical columns in the DataFrame

        Parameters:
        - method: Correlation method to use ("pearson", "kendall", or "spearman")
        """
        plt.figure(figsize=figsize)
        sns.heatmap(self.data.select_dtypes(include="number").corr(method=method), annot=True, fmt=".2f", cmap="RdYlGn")
        plt.show()