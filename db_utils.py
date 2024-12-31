import yaml
import os
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer

"""
This file contains various classes for interacting with a database.
    1. RDSDatabaseConnector(): Connecting to an AWS RDS database, extracting data from SQL DB to Pandas, downloading
    to local .csv.
    2. DataTransform: Changing column datatypes
    3. DataFrameInfo: Summarizing and exploring dataframes
    4. DataFrameTransform: Imputing/dropping null/missing values, replacing values,
       normalizing or removing outliers.
    5. Plotter: various visualizations for 

"""


def get_creds():
    """
    You must create a credentials.yaml file in your current directory with the login details to connect to the AWS db.
    """
    creds = os.path.join(os.path.dirname(__name__), "credentials.yaml")
    with open(creds, "r") as file:
        data = yaml.safe_load(file)
    return data

class RDSDatabaseConnector:
    def __init__(self, credentials: dict):
        """
        Initialize the connector with credentials.

        Args:
            credentials (dict): A dictionary containing AWS RDS credentials.
                Expected keys: ['host', 'port', 'user', 'password', 'database'].
        """
        # Store the credentials as variables
        self.host = credentials.get("RDS_HOST")
        self.port = credentials.get("RDS_PORT")
        self.user = credentials.get("RDS_USER")
        self.password = credentials.get("RDS_PASSWORD")
        self.database = credentials.get("RDS_DATABASE")

        if not all([self.host, self.port, self.user, self.password, self.database]):
            raise ValueError("Missing one or more required database credentials.")

    def engine(self):
        return create_engine(f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}")
    
    def extract_data(self, table_name: str) -> pd.DataFrame:
        """
        Extract data from the specified table and return it as a Pandas DataFrame.
        """
        query = f"SELECT * FROM {table_name};"

        # Use a context manager to ensure the connection is closed
        with self.engine().connect() as connection:
            df = pd.read_sql(query, connection)
        return df
    
    def download_csv(self, dataframe: pd.DataFrame, file_name: str, file_path: str = "./"):
        """
        Save the given DataFrame to a CSV file.

        Args:
            dataframe (pd.DataFrame): The DataFrame to save.
            file_name (str): The name of the output CSV file (include '.csv').
            file_path (str): The directory path where the file will be saved. Defaults to current directory.

        Returns:
            str: The full path of the saved file.
        """
        # Ensure the file name ends with .csv
        if not file_name.endswith(".csv"):
            file_name += ".csv"

        # Construct the full file path
        full_path = f"{file_path.rstrip('/')}/{file_name}"
        
        # Save DataFrame to CSV
        dataframe.to_csv(full_path, index=False)
        print(f"Data saved to {full_path}")
        return full_path

""" creds = get_creds()
connector = RDSDatabaseConnector(creds)
df = connector.extract_data('customer_activity')

if __name__ == "__main__":
     connector.download_csv(df, 'cust_act1') """

class DataFrameTransform:
    def __init__(self, df, columns=None):
        """
        Initialize with a pandas DataFrame and optional columns to target.

        Parameters:
        - df (pd.DataFrame): The DataFrame to transform.
        - columns (list or str, optional): Specific column(s) to target for transformations. 
                                            If None, applies to all columns.
        """
        self.df = df
        self.columns = columns if columns else df.columns

    def impute_missing(self, strategy=None):
        """
        Impute missing values in the specified numeric columns based on skewness or an explicitly provided strategy.

        Parameters:
        - strategy (str, optional): The imputation strategy. 
            Options: "mean", "median", "mode". If None, the strategy is automatically selected based on skewness.
            - For skewness > 1 or < -1, median is used.
            - Otherwise, mean is used.
            - Mode is applied only when explicitly set as the strategy.

        Returns:
        - The DataFrame with missing values imputed.
        """
        # Filter numeric columns from the specified columns
        numeric_cols = self.df[self.columns].select_dtypes(include=['int64', 'float64', 'Int32']).columns

        if numeric_cols.empty:
            print("No numeric columns found to impute.")
            return self.df

        # Perform imputation
        for col in numeric_cols:
            if self.df[col].isna().any():
                if strategy == "mode":
                    # Explicitly use mode
                    mode_value = self.df[col].mode()[0]
                    self.df[col].fillna(mode_value, inplace=True)
                    print(f"Imputed missing values in '{col}' with mode ({mode_value}).")
                else:
                    # Automatically choose mean/median based on skewness
                    skewness = self.df[col].skew()
                    if skewness > 1 or skewness < -1:
                        strategy = "median"
                        value = self.df[col].median()
                    else:
                        strategy = "mean"
                        value = self.df[col].mean()

                    print(f"Before imputation: {self.df[col].isna().sum()} nulls in '{col}'")
                    self.df[col].fillna(value, inplace=True)
                    print(f"After imputation: {self.df[col].isna().sum()} nulls in '{col}'")

                    print(f"Imputed missing values in '{col}' with {strategy} ({value:.2f}, skew={skewness:.2f}).")
        
        return self.df

    def drop_missing(self):
        """
        Drop rows with missing values in the specified columns.

        Returns:
        - The DataFrame with rows containing missing values dropped.
        """
        if isinstance(self.columns, str):  # Ensure self.columns is always iterable
            self.columns = [self.columns]

        initial_shape = self.df.shape

        # Drop rows where specified columns have missing values
        self.df.dropna(subset=self.columns, inplace=True)

        final_shape = self.df.shape
        print(f"Dropped {initial_shape[0] - final_shape[0]} rows with missing values in {self.columns}.")
        
        return self.df

    def replace(self, column, value_to_replace=None, replacement=''):
        """
        Replaces specified values in a column with a given replacement value.

        Parameters:
            column (str): The column to perform replacement on.
            value_to_replace (optional): The value to be replaced (default is None, i.e., NaN).
            replacement: The value to replace with (default is an empty string).
        """
        if column not in self.df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

        self.df[column] = self.df[column].replace(to_replace=value_to_replace, value=replacement)

    def reduce_skewness(self, method='log'):
        """
        Apply transformations to reduce skewness in numerical columns of a DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame
            method (str): Transformation method - 'log', 'sqrt', 'reciprocal', or 'power'
        
        Returns:
            pd.DataFrame: Transformed DataFrame
        """
        transformed_df = self.df.copy()
        
        for column in transformed_df.select_dtypes(include=[np.number]).columns:
            # Skip columns with non-positive values for log and reciprocal transformations
            if method in ['log', 'reciprocal'] and (transformed_df[column] <= 0).any():
                continue
            
            if method == 'log':
                transformed_df[column] = np.log1p(transformed_df[column])  # log1p handles 0 safely
            elif method == 'sqrt':
                transformed_df[column] = np.sqrt(transformed_df[column])
            elif method == 'reciprocal':
                transformed_df[column] = 1 / transformed_df[column]
            elif method == 'power':
                pt = PowerTransformer(method='yeo-johnson')
                transformed_df[column] = pt.fit_transform(transformed_df[[column]])
        
        return transformed_df

    def drop_outliers(self, z_threshold=3):
        """
        Drops rows with outliers based on Z-score for columns in self.columns.

        Parameters:
        - z_threshold (float): Z-score threshold for detecting outliers.

        Returns:
        - pd.DataFrame: A DataFrame with outliers removed.
        """
        if not self.columns:
            print("No columns specified for outlier removal.")
            return self.df

        df_cleaned = self.df.copy()  # Create a copy to avoid modifying the original DataFrame

        for col in self.columns:
            if col not in self.df.columns:
                print(f"Column '{col}' not found in the DataFrame. Skipping.")
                continue

            # Calculate Z-scores
            mean = df_cleaned[col].mean()
            std = df_cleaned[col].std()
            z_scores = (df_cleaned[col] - mean) / std

            # Remove rows where Z-score exceeds threshold
            df_cleaned = df_cleaned[(z_scores <= z_threshold) & (z_scores >= -z_threshold)]

        return df_cleaned

class DataTransform:
    """
    A class to transform the datatype of dataframe columns.

    Attributes:
        df (pd.DataFrame): The dataframe to work with.

    Methods:
        changetype(columns_dtypes): Changes the datatypes of columns specified in a dictionary:
            Keys: columns, values: desired datatypes.
    """
    def __init__(self, df):
        self.df = df

    def changetype(self, columns_dtypes: dict):
        """
        Change the datatypes of specified columns in the dataframe.

        Args:
            columns_dtypes (dict): Dictionary with column names as keys and target datatypes as values.
        """
        for column, dtype in columns_dtypes.items():
            try:
                # Attempt to change the column's dtype
                self.df[column] = self.df[column].astype(dtype)
                print(f"Successfully changed '{column}' to {dtype}")
            except Exception as e:
                print(f"Error changing '{column}' to {dtype}: {e}")
        return self.df

class DataFrameInfo:
    def __init__(self, df, df_column=None):
        self.df = df
        self.df_column = df_column if df_column is not None else df.columns

    def unique_count(self):
        """
        Prints the number of unique values in the specified column.
        """
        if self.df_column is None:
            print("Please include a column argument to check for unique values.")
        else:
            if isinstance(self.df_column, str):
                unique_count = len(set(self.df[self.df_column]))
                print(f"No. of unique values in '{self.df_column}': {unique_count} ()")
                return unique_count
            else:
                print("This method works for a single column. Please pass one column only.")

    def unique(self):
        """
        Prints the unique values in given column.
        """
        if self.df_column is None:
            print("Please include a column argument to check for unique values.")
        else:
            if isinstance(self.df_column, str):
                unique = set(self.df[self.df_column])
                print(f"Here are the unique values in '{self.df_column}':")
                print(unique)
                return unique
            else:
                print("This method works for a single column. Please pass one column only.")

    def null(self):
        """
        Calculates a count and percentage of null values in each column.
        Returns a pd.DataFrame.
        """
        df = self.df
        numnull = df.isna().sum()
        percentnull = df.isna().sum() / len(df) * 100
        null_df = pd.DataFrame({
            'ColumnName': numnull.index,
            'CountNull': numnull.values,
            'PercentageNull': np.round(percentnull.values, 2)
        })
        return null_df
    
    def nullpercent(self):
        pass

    def range(self):
        
        dfcol = self.df_column
        max = max(dfcol)
        min = min(dfcol)
        range = max - min
        print(f"Range of column: {range} (max: {max}, min: {min})")

    def count_outliers(self, z_treshold=3):
        """
        Counts the number & percentage of outliers in each column using Z-score and IQR methods.

        Parameters:
        - df (pd.DataFrame): The input DataFrame.
        - columns (list): List of column names to analyze for outliers.

        Returns:
        - pd.DataFrame: A summary DataFrame with counts & percentages of outliers for each column.
        """
        outlier_summary = {
            "Column": [],
            "Z-Score Outliers": [],
            "Z Percentage": [],
            "IQR Outliers": [],
            "IQR Percentage": []
        }

        for col in self.df_column:
            if col not in self.df.columns:
                print(f"Column '{col}' not found in the DataFrame. Skipping.")
                continue

            # Drop NaN values for analysis
            data = self.df[col].dropna()

            # Z-score method
            mean = data.mean()
            std = data.std()
            z_scores = (data - mean) / std
            z_outliers = ((z_scores > z_treshold) | (z_scores < -z_treshold)).sum()
            z_percentage = z_outliers / len(self.df) * 100

            # IQR method
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            iqr_percentage = iqr_outliers / len(self.df) * 100

            # Append results
            outlier_summary["Column"].append(col)
            outlier_summary["Z-Score Outliers"].append(z_outliers)
            outlier_summary["Z Percentage"].append(z_percentage)
            outlier_summary["IQR Outliers"].append(iqr_outliers)
            outlier_summary["IQR Percentage"].append(iqr_percentage)
            

        # Convert to DataFrame
        summary_df = pd.DataFrame(outlier_summary)
        return summary_df

class Plotter:
    def __init__(self, df):
        self.df = df

    def numeric_distributions(self, bins=30, plot_type='hist', highlight_outliers=True, columns=None, z_threshold=3):
        """
        Plots histograms or scatterplots for all numeric columns in the DataFrame,
        with the option to highlight outliers in scatterplots based on Z-score.

        Parameters:
        - bins (int): Number of bins for the histograms (applies only if plot_type='hist').
        - plot_type (str): Type of plot - 'hist' for histograms (default), 'scatter' for scatterplots.
        - columns (list): List of numeric columns to plot. If None, all numeric columns are plotted.
        - z_threshold (float): Z-score threshold for detecting outliers (applies to scatterplots).

        Returns:
        - A grid of plots for the specified numeric columns.
        """
        # Automatically select numeric columns if none are specified
        if columns is None:
            columns = self.df.select_dtypes(include=['int64', 'float64', 'Int32']).columns

        # Ensure columns is a list and check if it's empty
        if isinstance(columns, pd.Index):
            columns = columns.tolist()

        if len(columns) == 0:
            print("No numeric columns found to plot.")
            return

        # Determine grid size (rows and columns)
        n_cols = 3  # Number of plots per row
        n_rows = (len(columns) + n_cols - 1) // n_cols  # Calculate required rows

        plt.figure(figsize=(n_cols * 5, n_rows * 4))  # Adjust figure size

        # Create a plot for each numeric column
        for i, col in enumerate(columns, 1):
            if col not in self.df.columns:
                print(f"Column '{col}' not found in the DataFrame. Skipping.")
                continue

            plt.subplot(n_rows, n_cols, i)

            if plot_type == 'hist':
                sns.histplot(data=self.df, x=col, bins=bins, kde=True, color="skyblue")
                plt.title(f"Histogram of '{col}'", fontsize=12)
                plt.xlabel(col, fontsize=10)
                plt.ylabel("Frequency", fontsize=10)

            elif plot_type == 'scatter':
                # Drop NaN values for analysis
                data = self.df[col].dropna()

                # Calculate Z-scores
                mean = data.mean()
                std = data.std()
                z_scores = (data - mean) / std

                # Identify outliers (Z > z_threshold or Z < -z_threshold)
                is_outlier = (z_scores > z_threshold) | (z_scores < -z_threshold)

                # Plot normal points in blue and outliers in red
                if highlight_outliers:
                    plt.scatter(data.index, data, alpha=0.5, color="skyblue", label="Inliers")
                    plt.scatter(data.index[is_outlier], data[is_outlier], alpha=0.5, color="red", label="Outliers")
                else:
                    plt.scatter(data.index, data, alpha=0.5, color="skyblue")

                plt.title(f"Scatterplot of '{col}'", fontsize=12)
                plt.xlabel("Index", fontsize=10)
                plt.ylabel(col, fontsize=10)
                plt.legend()

        plt.tight_layout()
        plt.show()

    def correlation_map(self):
        """
        Generates a heatmap to visualize the correlations between numerical columns in the dataframe.
        Non-numerical columns are ignored.
        """
        # Filter to numeric columns only
        numeric_df = self.df.select_dtypes(include=['number'])

        if numeric_df.empty:
            print("No numeric columns found in the DataFrame to calculate correlations.")
            return

        # Calculate the correlation matrix
        corr_matrix = numeric_df.corr()

        # Set up the figure and aesthetic settings
        plt.figure(figsize=(10, 8))
        sns.set_theme(style="white")

        # Create the heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,            # Show correlation coefficients
            fmt=".2f",             # Format numbers to 2 decimal places
            cmap="YlOrRd",         # Intuitive colour palette
            cbar=True,             # Show the colour bar
            square=True,           # Make the cells square
            linewidths=0.5,        # Add lines between cells for clarity
        )

        # Add title and adjust layout
        plt.title("Correlation Heatmap", fontsize=16)
        plt.tight_layout()

        # Show the plot
        plt.show()










        