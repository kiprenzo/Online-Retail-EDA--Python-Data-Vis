import yaml
import os
from sqlalchemy import create_engine
import pandas as pd

"""
Running this file will download the 'customer_activity' table as a .csv to your current directory, given you have access to the AWS db credentials.
"""

dtype_change = {
    "visitor_type": "category",
    "traffic_type": "category",
    "region": "category",
    "browser": "category",
    "operating_systems": "category",
    "month": "category",
}

def get_creds():
    """
    You must create a credentials.yaml file in your current directory with the login details to connect to the AWS db.
    """
    creds = os.path.join(os.path.dirname(__name__), "credentials.yaml")
    with open(creds, "r") as file:
        data = yaml.safe_load(file)
    return data


class RDSDatabaseConnector():
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


creds = get_creds()
connector = RDSDatabaseConnector(creds)
df = connector.extract_data('customer_activity')

# if __name__ == "__main__":
#     connector.download_csv(df, 'cust_act1')

class DataTransform():
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

class GetInfo():
    def __init__(self, df, df_column=None):
        self.df = df
        self.df_column = df_column if df_column is not None else df.columns

    def unique(self):
        """
        Prints the number of unique values in the specified column.
        """
        if self.df_column is None:
            print("Please include a column argument to check for unique values.")
        else:
            if isinstance(self.df_column, str):
                unique_count = len(set(self.df[self.df_column]))
                print(f"No. of unique values in '{self.df_column}': {unique_count}")
                return unique_count
            else:
                print("This method works for a single column. Please pass one column only.")









        