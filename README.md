
# 📊 EDA Using Pandas:

The ___Pandas___ library in Python provides powerful tools for conducting __EDA__ efficiently.

It provides the data stuctures like __One dimentionl__ (Series) and __Two diamentional__ (DataFrame) to handle data efficiently.


we have to install pandas library uinsg _pip install pandas_ command as below

![Image](https://github.com/user-attachments/assets/72da5ebe-e31c-43fb-8c0c-1c300dcbbb85)

After installing library we need to _import_ library as follows so as to use all the functions available in the library.

![Image](https://github.com/user-attachments/assets/55898263-eb24-4573-b728-7c1df4e28275)

We can read data from _excel files, json, html_
using respective function in pandas library.

![Image](https://github.com/user-attachments/assets/e2f35194-88aa-48ae-a3bb-b10cf4238a3c)

## Read data into a DataFrame
In pandas, ***read_*** functions are used to read different types of data files into a DataFrame. Here are some commonly used ones:

**Pandas read_  Functions**

| Function               | File Type        | Example Usage                      | Description |
|------------------------|-----------------|------------------------------------|-------------|
| `pd.read_csv()`       | CSV              | `pd.read_csv("data.csv")`         | Reads a CSV file (comma-separated values). |
| `pd.read_excel()`     | Excel            | `pd.read_excel("data.xlsx")`      | Reads an Excel file (`.xlsx`, `.xls`). |
| `pd.read_json()`      | JSON             | `pd.read_json("data.json")`       | Reads a JSON file. |
| `pd.read_sql()`       | SQL Database     | `pd.read_sql("SELECT * FROM table", conn)` | Reads data from an SQL database. |
| `pd.read_parquet()`   | Parquet          | `pd.read_parquet("data.parquet")` | Reads a Parquet file (optimized for big data). |
| `pd.read_html()`      | HTML Table       | `pd.read_html("https://example.com")[0]` | Reads tables from an HTML page. |
| `pd.read_pickle()`    | Pickle           | `pd.read_pickle("data.pkl")`      | Loads a serialized pandas object. |
| `pd.read_feather()`   | Feather          | `pd.read_feather("data.feather")` | Reads a Feather file (fast binary format). |
| `pd.read_clipboard()` | Clipboard        | `pd.read_clipboard()`             | Reads data copied to the clipboard. |
| `pd.read_table()`     | Text Table       | `pd.read_table("data.txt", sep="\t")` | Reads a general text file (similar to `read_csv`). |


📌 *Note: Reads an Excel file (requires openpyxl or xlrd for .xlsx or .xls files)*

 
## 📂 Reading Excel Files
Using pandas we can read data from csv file.
__read_csv()__ function helps us to read data from csv file.

![Image](https://github.com/user-attachments/assets/8b9f11ba-a4b4-4ca8-b5c1-ee99ddeb65d6)


## 🛢️ Reading SQL Databases
```python
import sqlite3
conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT * FROM table_name", conn)
```
## 🌐Reading Parquet Files (Efficient for Big Data)
```python
df = pd.read_parquet("data.parquet")
```
📌 Reads a Parquet file (optimized columnar format).


-------------------------

By using pandas library, we can clean, manupulate data as per the business requirnments.

📂 Most widly used _functions_ are:
 * info()
 * head()
 * tail()
 * transform()
 * merge()
 * join()
 * describe()
 * isnull()
 * columns()


## _info()_
The *info()* finction provides a concise summary of a Pandas DataFrame, including:
 * Number of entries (rows)
 * Number of columns
 * Column names and data types
 * Non-null value counts
 * Memory usage


![Image](https://github.com/user-attachments/assets/3d9981d8-c6bb-49fd-942c-4347d504d29d)


## head()

The *head()* function returns the *first n rows* for the object based on position. It is useful for quickly testing if your object has the right type of data in it.

*Dafault* value is 5 rows, which can be reset to any desired value.

![Image](https://github.com/user-attachments/assets/0abb4c1b-7679-4a9b-ba53-97be086ae13a)

## *tail()*
The *tail()* function returns *last n rows* from the object based on position. It is useful for quickly verifying data, for example, after sorting or appending rows.

*Dafault* value is 5 rows, which can be reset to any desired value.

![Image](https://github.com/user-attachments/assets/48948db2-dfba-4e11-8cf2-387d206ef20d)

## *columns*
*Columns* is attribue to the data frame. it returns all the columns in the dataframe.

![Image](https://github.com/user-attachments/assets/a12390fe-b4e3-4ed7-bdf7-a16be8cdb766)

## shape
The *shape* attribute in Pandas returns the number of rows and columns in a DataFrame as a tuple (rows, columns).

Does not require parentheses () because it is an attribute, not a method.
Returns a tuple → (number_of_rows, number_of_columns)

![Image](https://github.com/user-attachments/assets/d64e9957-d58e-49e9-b1c3-08b0bd8e074d)

## isnull()
The *isnull()* returns a boolean same-sized object indicating if the values are NA. NA values, such as None or numpy.NaN, gets mapped to True values. Everything else gets mapped to False values. Characters such as empty strings '' or numpy.inf are not considered NA values (unless you set pandas.options.mode.use_inf_as_na = True).

![Image](https://github.com/user-attachments/assets/a26e79b2-b1d9-4fa0-8437-0de8491c25e5)

The ***.isnull().sum()*** function is used to identify missing (NaN) values in a DataFrame.

* df.isnull() → Returns a DataFrame with True (for missing values) and False (for non-missing values)

* .sum() → Counts the number of True values (i.e., missing values) in each column.

How to Interpret?
* Column Name → 0 missing values
* Column Age → 1 missing value
* Column Salary → 2 missing values

Why Use df.isnull().sum()?
* Helps identify missing values before analysis
* Useful before handling missing data (e.g., fillna(), dropna())
* Essential in data preprocessing for machine learning

![Image](https://github.com/user-attachments/assets/5ffbd20f-eb29-4e7a-bcab-b70be3d9b2fe)

## describe()
The *describe()* function provides summary statistics for numerical columns in a DataFrame. It helps in Exploratory Data Analysis (EDA) by giving insights into the dataset’s distribution.

By default, it returns statistics only for numerical columns.
For categorical columns, use df.describe(include='object').

![Image](https://github.com/user-attachments/assets/37a43f4e-c36c-44bd-8399-8e891fd2a154)

## Add a new column

We can add a new column in existing datafram in pandas:




## 🔄 Convert DataFrames into different formats

Pandas provides several to_ functions to convert DataFrames into different formats. Here’s a structured table for quick reference:

**Pandas `to_` Functions**

Pandas provides several `to_` functions to convert DataFrames into different formats.

| Function        | Description                                   | Example Usage                 |
|---------------|--------------------------------|--------------------------------|
| `to_csv()`   | Save DataFrame as a CSV file 📄 | `df.to_csv('file.csv')`       |
| `to_excel()` | Export DataFrame to an Excel file 📊 | `df.to_excel('file.xlsx')`    |
| `to_json()`  | Convert DataFrame to JSON format 🌐 | `df.to_json('file.json')`     |
| `to_sql()`   | Write records to an SQL database 🗄️ | `df.to_sql('table_name', conn)` |
| `to_dict()`  | Convert DataFrame to a dictionary 📋 | `df.to_dict()`                |
| `to_records()` | Convert to a NumPy record array 📑 | `df.to_records()`            |
| `to_parquet()` | Save as a Parquet file 📦 | `df.to_parquet('file.parquet')` |
| `to_clipboard()` | Copy DataFrame to clipboard 📎 | `df.to_clipboard()`           |
| `to_markdown()` | Convert DataFrame to Markdown format 📝 | `df.to_markdown()`           |
| `to_html()` | Convert DataFrame to an HTML table 🌍 | `df.to_html('file.html')`     |

📌 *Use these functions to export your data efficiently!*
