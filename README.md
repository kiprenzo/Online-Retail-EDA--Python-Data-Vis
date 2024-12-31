# Exploratory Data Analysis Exercise

The aim of this project is to hone my skills in EDA by working on a sample database end-to-end:

* Establishing a connection with an AWS RDS with secure credentials
* Saving data to a local .csv
* Exploring the data using various plots & aggregate tables to find patterns and identify areas of improvement
* Cleaning the data by:
    * making datatype conversions to save memory
    * imputing and/or dropping null values
    * normalizing/transforming data to reduce skewness
    * removing outliers to reduce biases
    * removing overly correlated columns

All of the above is achieved using neatly constructed python classes and methods in `db_utils.py`.

# Requirements

**Important:** in order to use `RDSDatabaseConnector`, you must have the necessary credentials to connect to an AWS RDS db and store them in a `credentials.yaml` file in a dictionary format:
> RDS_HOST: *host url*  
> RDS_PASSWORD: *pw*  
> RDS_USER: *user*  
> RDS_DATABASE: *db*  
> RDS_PORT: *port*

### Libraries

- [pandas](https://github.com/pandas-dev/pandas) - run `pip install pandas`
- [yaml](https://github.com/yaml/pyyaml) - run `pip install pyYAML`
- [numpy](https://github.com/numpy/numpy) - run `pip install numpy`
- [seaborn](https://github.com/mwaskom/seaborn) - run `pip install seaborn`
- [matplotlib](https://github.com/matplotlib/matplotlib) - run `pip install matplotlib`
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - run `pip install scikit-learn`
- [SQLAlchemy](https://github.com/sqlalchemy/sqlalchemy) - run `pip install sqlalchemy`

# Usage

`db_utils.py` has many reusable classes, useful for EDA.  
--TODO expand

