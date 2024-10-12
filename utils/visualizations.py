import matplotlib.pyplot as plt
import seaborn as sns

class Visualization:
    """
    A visualization class for data analysis

    Attributes:
    - data (DataFrame): The dataset to be visualized

    Methods:
    - __init__: Initialize the Visualization object with the dataset
    - create_subplots: Creates a figure and subplots with common settings
    - remove_unused_axes: Remove unused axes from the subplots grid
    - plot_columns: Plot specified graphs using the given plotting function
    - num_univariate_analysis: Perform univariate analysis for numerical features
    - cat_univariate_analysis: Perform univariate analysis for categorical features
    - datetime_univariate_analysis: Perform univariate analysis for datetime features
    - num_features_vs_target: Plot numerical features against the target variable
    - cat_features_vs_target: Plot categorical features against the target variable
    - datetime_features_vs_target: Plot datetime features against the target variable
    - facegrid_hist_target: Generate FacetGrid histograms for numerical columns based on target values
    - scatter_numericals_target: Plot scatter plots for numerical features with the target variable as hue
    """
    def __init__(self, data):
        """
        Initialize the Visualization object

        Parameters:
        - data (DataFrame): The dataset to be visualized
        """
        self.data = data

    def create_subplots(self, rows, columns, figsize=(18,12)):
        """
        Creates a figure and subplots with common settings

        Parameters:
        - rows (int): Number of rows for subplots grid
        - columns (int): Number of columns for subplots grid
        - figsize (tuple, optional): Figure size. Default is (18, 12)
        
        Returns:
        - fig: The figure object
        - ax (array of Axes): Array of axes objects
        """
        fig, ax = plt.subplots(rows, columns, figsize=figsize)
        ax = ax.ravel()
        return fig, ax

    def remove_unused_axes(self, fig, ax, num_plots):
        """
        Remove unused axes from the subplots grid

        Parameters:
        - fig (Figure): The figure object
        - ax (array of Axes): Array of axes objects
        - num_plots (int): Number of plots to be displayed
        """
        total_axes = len(ax)
        for j in range(num_plots, total_axes):
            fig.delaxes(ax[j])

    def plot_columns(self, cols, plot_func, ax, title_prefix="", target=None, x=None):
        """
        Plots specified graphs using the given plotting function

        Parameters:
        - cols (list): List of column names to be plotted
        - plot_func: The Seaborn plotting function to be used
        - ax (array of Axes): Matplotlib axes to plot on
        - title_prefix (str, optional): Prefix to be added to the title of each subplot (default is an empty string)
        - target (str, optional): Name of the target variable for hue, if applicable
        - x (str, optional): Name of the feature variable for the x-axis when plotting scatter plots (default is None)
        """
        for i, col in enumerate(cols):
            if plot_func == sns.boxplot:
                plot_func(y=self.data[col], x=self.data[target], hue=self.data[target], ax=ax[i], legend=False)
            elif plot_func == sns.histplot:
                if target is not None:
                    plot_func(data=self.data, x=col, hue=target, multiple="stack", ax=ax[i])
                else:
                    plot_func(self.data[col], ax=ax[i], kde=True)
            elif plot_func == sns.barplot and target is not None:
                data_grouped = self.data.groupby(self.data[col])[target].mean().reset_index()
                order = data_grouped.sort_values(target, ascending=False)[col].tolist()
                plot_func(data=data_grouped, x=self.data[col], y=self.data[target], order=order, ci=None, ax=ax[i])
                ax[i].set_title(f"Mean {target} over {col}")
                continue
            elif plot_func == sns.countplot:
                order = self.data[col].value_counts().index
                if target is not None:
                    plot_func(data=self.data, x=col, order=order, hue=target, ax=ax[i])
                else:
                    plot_func(data=self.data, x=col, order=order, ax=ax[i])
            elif plot_func == sns.scatterplot and x is not None:
                plot_func(y=self.data[col], x=self.data[x], hue=self.data[target], ax=ax[i])
            else:
                plot_func(x=self.data[col], ax=ax[i])
            ax[i].set_title(f"{title_prefix}{col}")

    def num_univariate_analysis(self, cols, rows, columns):
        """
        Perform univariate analysis on numerical columns and plot distributions

        Parameters:
        - cols (list): List of numerical columns to analyze
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols]
        self.plot_columns(cols, sns.histplot, ax, title_prefix="Distribution of ")
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def cat_univariate_analysis(self, cols, rows, columns):
        """
        Performs univariate analysis on categorical columns and plot count plots

        Parameters:
        - cols (list): List of categorical columns to analyze
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols]
        self.plot_columns(cols, sns.countplot, ax)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def datetime_univariate_analysis(self, date_cols, rows, columns):
        """
        Plot distributions of year and month extracted from specified datetime columns

        Parameters:
        - date_cols (list): List of datetime column names to extract year and month from
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, date_col in enumerate(date_cols):
            year_counts = self.data[date_col].dt.year.value_counts().index
            month_counts = self.data[date_col].dt.month.value_counts().index

            # Plot year distribution
            sns.countplot(x=self.data[date_col].dt.year, data=self.data, order=year_counts, ax=ax[i * 2])
            ax[i * 2].set_title(f'Distribution of {date_col} - Year')
            ax[i * 2].set_xlabel('Year')
            ax[i * 2].set_ylabel('Count')

            # Plot month distribution
            sns.countplot(x=self.data[date_col].dt.month, data=self.data, order=month_counts, ax=ax[i * 2 + 1])
            ax[i * 2 + 1].set_title(f'Distribution of {date_col} - Month')
            ax[i * 2 + 1].set_xlabel('Month')
            ax[i * 2 + 1].set_ylabel('Count')

        total_plots = len(date_cols) * 2
        self.remove_unused_axes(fig, ax, total_plots)
        plt.tight_layout()
        plt.show()

    def num_features_vs_target(self, rows, columns, target, cols, func):
        """
        Plots numerical features against the target variable

        Parameters:
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        - target (str): Name of the target variable
        - cols (list): List of numerical column names to plot against the target
        - func (str): Name of the Seaborn plotting function to use
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols]
        self.plot_columns(cols, getattr(sns, func.lower(), None), ax, f"{target} x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def cat_features_vs_target(self, rows, columns, target, cols, func):
        """
        Plot categorical features against the target variable using specified plots

        Parameters:
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        - target (str): Name of the target variable
        - cols (list): List of categorical column names to plot against the target
        - func (str): Name of the Seaborn plotting function to use
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols]
        self.plot_columns(cols, getattr(sns, func.lower(), None), ax, f"{target} x ", target=target)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()

    def datetime_features_vs_target(self, date_cols, rows, columns, target):
        """
        Plot datetime features against the target variable

        Parameters:
        - date_cols (list): List of datetime column names to plot against the target
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        - target (str): Name of the target variable
        """
        fig, ax = self.create_subplots(rows, columns)
        for i, date_col in enumerate(date_cols):
            yearly_default = self.data.groupby(self.data[date_col].dt.year)[target].mean().reset_index()
            monthly_default = self.data.groupby(self.data[date_col].dt.month)[target].mean().reset_index()
            order_year = yearly_default.sort_values(target, ascending=False)[date_col].tolist()
            order_month = monthly_default.sort_values(target, ascending=False)[date_col].tolist()

            # Plot year distribution
            sns.barplot(data=yearly_default, x=yearly_default[self.data[date_col].dt.year.name], y=target, order=order_year, ci=None, ax=ax[i * 2])
            ax[i * 2].set_title(f'Mean {target} over {date_col} - Year')
            ax[i * 2].set_xlabel('Year')
            ax[i * 2].set_ylabel('Count')

            # Plot month distribution
            sns.barplot(data=monthly_default, x=monthly_default[self.data[date_col].dt.month.name], y=target, order=order_month, ci=None, ax=ax[i * 2 + 1])
            ax[i * 2 + 1].set_title(f'Mean {target} over {date_col} - Month')
            ax[i * 2 + 1].set_xlabel('Month')
            ax[i * 2 + 1].set_ylabel('Count')

        total_plots = len(date_cols) * 2
        self.remove_unused_axes(fig, ax, total_plots)
        plt.tight_layout()
        plt.show()

    def facegrid_hist_target(self, facecol, color, cols, target):
        """
        Generates FacetGrid histograms for numerical columns based on target values

        Parameters:
        - facecol: Column for creating facets in the FacetGrid
        - color: Color for the histograms
        - cols (list): List of column names for which to plot histograms
        - target (str): Name of the target variable to filter the data
        """
        for col in cols:
            g = sns.FacetGrid(self.data[self.data[target]==1], col=facecol)
            g.map(sns.histplot, col, color=color)
            plt.show()

    def scatter_numericals_target(self, rows, columns, cols, target, x):
        """
        Plot scatter plots for numerical features with the target variable as hue

        Parameters:
        - rows (int): Number of rows for the subplots grid
        - columns (int): Number of columns for the subplots grid
        - cols (list): List of numerical column names to plot against the feature variable
        - target (str): Name of the target variable
        - x (str): Name of the feature variable for the x-axis
        """
        fig, ax = self.create_subplots(rows, columns)
        cols = self.data[cols].drop(columns=[x])
        self.plot_columns(cols, sns.scatterplot, ax, f"{x} x ", target=target, x=x)
        self.remove_unused_axes(fig, ax, cols.shape[1])
        plt.tight_layout()
        plt.show()