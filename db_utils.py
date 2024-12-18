import yaml
import os
from sqlalchemy import create_engine
import pandas as pd

"""
Running this file will download the 'customer_activity' table as a .csv to your current directory, given you have access to the AWS db credentials.
"""

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

connector.download_csv(df, 'cust_act1')



        