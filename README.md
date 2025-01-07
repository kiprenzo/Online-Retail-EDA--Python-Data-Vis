# Exploratory Data Analysis Exercise

The aim of this project is to hone my skills in EDA by working on a sample database (retail website) end-to-end:

* Establishing a connection with an AWS RDS with secure credentials
* Saving data to a local .csv
* Exploring the data using various plots & aggregate tables to find patterns and identify areas of improvement
* Cleaning the data by:
    * making datatype conversions to save memory
    * imputing and/or dropping null values
    * normalizing/transforming data to reduce skewness
    * removing outliers to reduce biases
    * removing overly correlated columns

All of the above is achieved using neatly constructed python classes and methods in `db_utils.py` and some public libraries.

I then perform EDA on the cleaned dataset, answering questions that management/exec might ask their data analysts.
## Demo
You can view a sample of my data analysis by opening my data_vis.ipynb in NBViewer by [clicking here](https://nbviewer.org/github/kiprenzo/exploratory-data-analysis---online-shopping-in-retail986/blob/main/data_vis.ipynb).

Alternatively, if you'd like to only see the tidy outputs without the code, you can download and view the `data_vis.html` file in your browser.

You can likewise see the decision making behind my data cleaning by opening `dataplay.ipynb` in NBViewer by [clicking here](https://nbviewer.org/github/kiprenzo/exploratory-data-analysis---online-shopping-in-retail986/blob/main/dataplay.ipynb).
## Requirements

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
- [Plotly](https://plotly.com/python/) - run `pip install plotly`
- [Missingno](https://github.com/ResidentMario/missingno) - run `pip install missingno`

Note: you can chain together space-separated modules `pip install module1 module2` so if you'd like, run:
```bash
pip install pandas pyYAML numpy seaborn matplotlib scikit-learn sqlalchemy plotly missingno
```

## File Structure / Usage

1. `db_utils.py` - this stores the reusable classes and methods, useful for any EDA project.
3. `data_vis` - a Jupyter notebook & tidy .html counterpart. This is a example of a report/investigation answering questions about the data.
4. `dataplay` - a Jupyter notebook and tidy-ish .html counterpart. This explains the ideology and decision-making behind the data cleaning steps taken to produce a clean pandas dataframe.
1. `data_dictionary.md` - a handy description of each column in the sample dataset. This is all the 'domain knowledge' I had access to during this project.

## License
 
The MIT License (MIT)

Copyright (c) 2025 Kipras Varanavicius

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.