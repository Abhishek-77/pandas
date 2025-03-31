
# 📊 EDA Using Pandas:

The ___Pandas___ library in Python provides powerful tools for conducting __EDA__ efficiently.

It provides the data stuctures like __One dimentionl__ (Series) and __Two diamentional__ (DataFrame) to handle data efficiently.
Pandas is powerful for handling structured data, and NumPy brings efficiency for numerical operations.

We have to install pandas library uinsg _pip install pandas_ command as below

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
 * mode()



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

## .mode()
In Pandas, the *.mode()* function is used to find the most frequently occurring value(s) in a column or DataFrame. It returns a Series or DataFrame with the mode(s) for each column.

#### 🔹 Syntax:

df.mode()

#### 📌 Parameters:
   * axis=0 (default) → Finds mode along columns.

   * axis=1 → Finds mode along rows.

   * numeric_only=True → Returns mode for only numeric columns.

   * dropna=True (default) → Ignores NaN values when calculating mode.


#### 🔹 Example 1: Finding Mode of a Column

```python
import pandas as pd

# Sample DataFrame
data = {'Product': ['Espresso', 'Latte', 'Latte', 'Cappuccino', 'Espresso'],
        'Price': [3.5, 4.0, 4.0, 4.5, 3.5]}

df = pd.DataFrame(data)

# Find mode of 'Product' column
print(df['Product'].mode())
```

✅ Output:
```python
0    Espresso
1       Latte
dtype: object
(Since "Espresso" and "Latte" both appear twice, both are returned.)
```
#### 🔹 Example 2: Finding Mode for All Columns

``` python
df.mode()
```
✅ Output:

```pyhton
      Product  Price
0   Espresso    3.5
1      Latte    4.0

The mode for Product is "Espresso" and "Latte" (both appear twice).
The mode for Price is 3.5 and 4.0 (both appear twice).
```
#### 🔹 Example 3: Finding Mode Along Rows (axis=1)
```pyhton
df.mode(axis=1)
```
This finds the most common values per row, but it’s usually more useful for categorical data.

#### 🔹 Handling Missing Values (dropna=False)
By default, .mode() ignores missing values. To include them:
```python
df['Product'].mode(dropna=False)
```

#### 💡 Key Points
* .mode() returns ***all*** most frequent values (not just one).

* Works with ***both numbers and text*** (categorical data).

* Can be used ***column-wise (default) or row-wise (axis=1)***.   


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

 *Use these functions to export your data efficiently!*

===================================================================

Here are some ***Pandas functions*** categorized by functionality:

### 1. Data Exploration & Summary:


| Function       | Description                                        |
|---------------|----------------------------------------------------|
| `df.head(n)`  | Displays the first `n` rows (default 5)           |
| `df.tail(n)`  | Displays the last `n` rows                        |
| `df.info()`   | Summary of DataFrame (data types, non-null values) |
| `df.describe()` | Summary statistics of numeric columns            |
| `df.shape`    | Returns tuple `(rows, columns)`                   |


#### 📑 Ex:

```python
import pandas as pd

# Sample DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 40],
        'Salary': [50000, 60000, 70000, 80000]}
df = pd.DataFrame(data)

# Display first 2 rows
print(df.head(2))

# Summary of the DataFrame
print(df.info())

# Summary statistics
print(df.describe())
```


### 2. Selecting & Filtering Data:


| Function | Description |
|----------|------------|
| `df['col_name']` | Selects a column as Series |
| `df[['col1', 'col2']]` | Selects multiple columns |
| `df.loc[row_label, col_label]` | Selects data by label |
| `df.iloc[row_index, col_index]` | Selects data by position |
| `df[df['col'] > value]` | Filters rows based on condition |




#### 📑 Ex:

```python
# Selecting a single column
print(df['Name'])

# Selecting multiple columns
print(df[['Name', 'Salary']])

# Selecting using loc (by labels)
print(df.loc[1, 'Name'])  # Bob

# Selecting using iloc (by index positions)
print(df.iloc[2, 1])  # 35

# Filtering rows where Age > 30
print(df[df['Age'] > 30])

```

### 3. Handling Missing Data:

| Function | Description |
|----------|------------|
| `df.isnull().sum()` | Counts missing values per column |
| `df.dropna()` | Removes rows with missing values |
| `df.fillna(value)` | Replaces missing values with a given value |


#### 📑 Ex:

```python
data_with_nan = {'Name': ['Alice', 'Bob', 'Charlie'],
                 'Age': [25, None, 35],
                 'Salary': [50000, 60000, None]}
df_nan = pd.DataFrame(data_with_nan)

# Check for missing values
print(df_nan.isnull().sum())

# Fill missing values
df_filled = df_nan.fillna({'Age': df_nan['Age'].mean(), 'Salary': 0})
print(df_filled)

# Drop rows with missing values
df_cleaned = df_nan.dropna()
print(df_cleaned)

```


### 4. Data Transformation:

| Function | Description |
|----------|------------|
| `df.apply(func)` | Applies a function to each row/column |
| `df.map(func)` | Maps a function to elements of a Series |
| `df.groupby('col')` | Groups data by a column |
| `df.pivot_table(values, index, columns)` | Creates pivot tables |



#### 📑 Ex:

```python
# Apply function (increase salary by 10%)
df['Salary'] = df['Salary'].apply(lambda x: x * 1.1)
print(df)

# Map function (categorizing age)
df['Age_Group'] = df['Age'].map(lambda x: 'Young' if x < 30 else 'Old')
print(df)

# Grouping data by Age_Group and calculating mean Salary
print(df.groupby('Age_Group')['Salary'].mean())

```


### 5. Merging & Joining:

| Function | Description |
|----------|------------|
| `pd.concat([df1, df2])` | Concatenates DataFrames along an axis |
| `df.merge(df2, on='col')` | Merges DataFrames on a column |




#### 📑 Ex:

```python
df1 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Alice', 'Bob', 'Charlie']})
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Salary': [50000, 60000, 70000]})

# Merge DataFrames on ID
df_merged = df1.merge(df2, on='ID')
print(df_merged)

# Concatenating two DataFrames
df_concat = pd.concat([df1, df2], axis=1)
print(df_concat)

```

Here are ***examples*** for each category of Pandas functions:

Here are more Pandas function examples across different categories:

### 6. Sorting Data:

#### 📑 Ex:

```python
# Sorting by Age in ascending order
df_sorted = df.sort_values(by='Age', ascending=True)
print(df_sorted)

# Sorting by multiple columns (Age then Salary)
df_sorted = df.sort_values(by=['Age', 'Salary'], ascending=[True, False])
print(df_sorted)

```

### 7. Handling Duplicates:
#### 📑 Ex:

```python
# Sample DataFrame with duplicates
data = {'Name': ['Alice', 'Bob', 'Alice', 'David'],
        'Age': [25, 30, 25, 40]}
df_dup = pd.DataFrame(data)

# Check for duplicates
print(df_dup.duplicated())

# Remove duplicate rows
df_unique = df_dup.drop_duplicates()
print(df_unique)

```

### 8. Renaming Columns:

#### 📑 Ex:

```python
# Rename columns
df_renamed = df.rename(columns={'Name': 'Employee_Name', 'Salary': 'Annual_Income'})
print(df_renamed)

```


### 9. Pivot Tables
#### 📑 Ex:

```python
# Creating a pivot table
df_pivot = df.pivot_table(values='Salary', index='Age_Group', aggfunc='mean')
print(df_pivot)
```

### 10. Reshaping Data (Melt & Pivot):
#### 📑 Ex:

```python
# Sample DataFrame
df_wide = pd.DataFrame({'ID': [1, 2], 'Math': [90, 85], 'Science': [80, 75]})

# Melting DataFrame (Wide to Long format)
df_long = df_wide.melt(id_vars=['ID'], var_name='Subject', value_name='Score')
print(df_long)

# Pivoting back (Long to Wide format)
df_wide_again = df_long.pivot(index='ID', columns='Subject', values='Score')
print(df_wide_again)

```


### 11. Working with Date & Time:

#### 📑 Ex:

```python
# Convert column to datetime
df['Joining_Date'] = pd.to_datetime(['2023-01-01', '2022-05-15', '2021-06-30', '2020-12-10'])

# Extract year, month, day
df['Year'] = df['Joining_Date'].dt.year
df['Month'] = df['Joining_Date'].dt.month
df['Day'] = df['Joining_Date'].dt.day

print(df)

```

### 12. Exporting Data:

#### 📑 Ex:

```python
# Export to CSV
df.to_csv('output.csv', index=False)

# Export to Excel
df.to_excel('output.xlsx', index=False)

# Export to JSON
df.to_json('output.json', orient='records')

```

Here are some examples of ***adding a new column*** and updating data in an existing column in Pandas:

### 1️⃣ Adding a New Column:
#### 📑 Ex. 1: Adding a column with constant values:


```python
import pandas as pd

# Creating a sample DataFrame
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'],
                   'Age': [25, 30, 35]})

# Adding a new column with a constant value
df['Country'] = 'USA'

print(df)

```

#### 📑 Ex. 2: Adding a column based on another column:


```python
# Adding a new column using a calculation
df['Age in 5 years'] = df['Age'] + 5

print(df)

```

#### 📑 Ex. 3: Adding a column using apply():


```python
# Adding a column based on a condition
df['Age Category'] = df['Age'].apply(lambda x: 'Young' if x < 30 else 'Adult')

print(df)
```

### 2️⃣ Updating an Existing Column

#### 📑 Ex 1: Updating column values using assignment:


```python
# Updating all values in the 'Country' column
df['Country'] = 'Canada'

print(df)
```

#### 📑 Ex 2: Updating specific rows using conditions:


```python
# Updating Age for a specific Name
df.loc[df['Name'] == 'Alice', 'Age'] = 26

print(df)
```

#### 📑 Ex 3: Updating values using apply():


```python
# Increasing age by 2 years
df['Age'] = df['Age'].apply(lambda x: x + 2)

print(df)
```
### 📌 map(), replace(), and where()

Here are some advanced ways to update data in an existing Pandas DataFrame using *map(), replace(), and where()*:

#### 🔹Updating Column Values with map():
The *map()* function is useful when "replacing values" based on a dictionary or function.

📑 Ex. 1: Mapping values using a dictionary

```python
import pandas as pd

# Creating a DataFrame
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'],
                   'Department': ['HR', 'IT', 'Finance']})

# Mapping department names to codes
dept_map = {'HR': 101, 'IT': 102, 'Finance': 103}

# Updating the 'Department' column
df['Dept_Code'] = df['Department'].map(dept_map)

print(df)

```

#### 🔹 Updating Column Values with replace():

The *replace()* function is useful when you want to substitute specific values.

📑 Ex. 2: Replacing specific values in a column

```python
# Replacing values in the 'Department' column
df['Department'] = df['Department'].replace({'HR': 'Human Resources', 'IT': 'Information Technology'})

print(df)

```

#### 🔹 Updating Column Values Conditionally with where()
The *where()* method keeps values unchanged unless they match a condition.

📑 Ex 3: Updating values only if they meet a condition

```python
# Increase Dept_Code by 10 only if it's greater than 101
df['Dept_Code'] = df['Dept_Code'].where(df['Dept_Code'] <= 101, df['Dept_Code'] + 10)

print(df)

```


#### 🔹 Updating with apply() using multiple columns

📑 Ex 4:


```python

# Updating salary based on department
df['Salary'] = df['Dept_Code'].apply(lambda x: 50000 if x == 101 else 70000)

print(df)

```

#### ✅ Key Takeaways:

| Method   | Best Used For                                      |
|----------|---------------------------------------------------|
| `map()`  | Replace values based on a dictionary or function |
| `replace()` | Replace values directly with another value   |
| `where()` | Modify values only where a condition is met    |
| `apply()` | Apply a function to each row/column           |


### 📌 Handling NaN (Missing Values) While Updating Data in Pandas
When updating DataFrame values, *missing values* (NaN) can cause issues. Here are advanced ways to handle and update missing values while ensuring data integrity.


#### 🔹 1. Using fillna() to Replace NaN Values
The *fillna()* method replaces all missing values with a *specified value*.

📑 Ex: Fill NaN with a Default Value:

```python
import pandas as pd
import numpy as np

# Creating a DataFrame with missing values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Salary': [50000, np.nan, 60000, np.nan]
})

# Fill NaN values with a default salary of 55000
df['Salary'] = df['Salary'].fillna(55000)

print(df)

```


✅ Best for: Replacing all missing values in a column with a default value.


--------

#### 🔹 2. Using fillna() with Different Values for Each Row:

Instead of a single value, use a *dictionary* to fill missing values with different values.

📑 Ex: Fill NaN with Different Values


```python

# Fill missing values with department-based default salaries
salary_map = {'Alice': 50000, 'Bob': 52000, 'Charlie': 60000, 'David': 58000}
df['Salary'] = df['Salary'].fillna(df['Name'].map(salary_map))

print(df)

```
✅ Best for: Assigning different default values based on another column.

-----------

#### 🔹 3. Using apply() to Handle NaN with a Custom Function:

The *apply()* method lets you conditionally modify NaN values.

📑 Ex: Conditional Salary Updates:

```python
df['Salary'] = df['Salary'].apply(lambda x: 55000 if pd.isna(x) else x)
print(df)
```
✅ Best for: Applying logic dynamically while updating.

---------


#### 🔹 4. Using where() to Update Values with Conditions:

The *where()* function keeps existing values and only updates where the "condition" is met.

📑 Ex: Increase Salary Only for NaN Values

```python
df['Salary'] = df['Salary'].where(df['Salary'].notna(), 60000)
print(df)
```

✅ Best for: Selectively replacing NaN without affecting existing values.

---

#### 🔹 5. Using interpolate() for Numeric Data
For numerical data, *interpolate()* fills missing values by estimating based on surrounding values.

📑 Ex: Interpolating Missing Salary

```python
df['Salary'] = df['Salary'].interpolate()
print(df)
```

✅ Best for: Filling missing numeric data with estimated values.

------

#### 🔹 6. Using replace() for NaN
If you want to explicitly replace NaN with a specific value, *replace()* can be used.

📑 Ex: Replace NaN with a Fixed Value

```python
df['Salary'] = df['Salary'].replace(np.nan, 50000)
print(df)
```
✅ Best for: Replacing NaN with a specific value without modifying other values.

---

#### ✅ Comparison of Methods

| Method                         | Best Used For                                      |
|--------------------------------|---------------------------------------------------|
| `fillna(value)`               | Replacing all missing values with a single value  |
| `fillna(map)`                 | Filling NaN with different values per row        |
| `apply(lambda x: ...)`        | Applying custom logic for filling missing values |
| `where(condition, new_value)` | Only changing NaN while keeping existing values  |
| `interpolate()`               | Estimating missing values for numeric data       |
| `replace(np.nan, value)`      | Directly replacing NaN with a specific value     |



### 📌 Handling Missing Values in Multiple Columns in Pandas
When working with real-world datasets, missing values (NaN) can appear in multiple columns. Here’s how you can efficiently handle them while maintaining data integrity.

#### 🔹 1. Fill Missing Values with a Default Value in Multiple Columns

Use *.fillna()* to replace all NaN values with a fixed value for *multiple columns*.

📑 Ex: Fill All NaN with 0:

```python
import pandas as pd
import numpy as np

# Creating a DataFrame with missing values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Salary': [50000, np.nan, 60000, np.nan],
    'Age': [25, np.nan, 30, np.nan],
    'Department': ['HR', 'Finance', np.nan, 'IT']
})

# Fill all NaN values with a default value (e.g., 0 for numerical columns, 'Unknown' for categorical)
df.fillna({'Salary': 50000, 'Age': 27, 'Department': 'Unknown'}, inplace=True)

print(df)

```

✅ Best for: Filling different values for different columns.

--------

#### 🔹 2. Fill NaN in Numeric Columns with Mean/Median
Use *.fillna(df.mean())* to replace missing values with the *mean* of each column.

📑 Ex: Fill Missing Values with Column Mean:

```python
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())
df['Age'] = df['Age'].fillna(df['Age'].median())

print(df)
```

✅ Best for: Handling missing values in numeric columns.

----

#### 🔹 3. Forward and Backward Filling (ffill and bfill)
You can propagate previous values *forward (ffill) or backward (bfill)* to fill NaN.

📑 Ex: Fill Missing Values with Previous or Next Value:


```python
df.fillna(method='ffill', inplace=True)  # Forward fill
# df.fillna(method='bfill', inplace=True)  # Backward fill (alternative)
print(df)

```
✅ Best for: Filling sequential data like time-series.
--------

#### 🔹 4. Interpolating Missing Numeric Data
If the data is continuous, you can use *interpolation* to estimate missing values.

📑 Ex: Interpolate Missing Values:

```python
df['Salary'] = df['Salary'].interpolate()
df['Age'] = df['Age'].interpolate()

print(df)
```
✅ Best for: Estimating missing values based on patterns.

----

#### 🔹 5. Drop Rows or Columns with NaN Values
Use *.dropna()* to remove rows or columns containing NaN values.

📑 Ex: Remove Rows with Any NaN Values:


```python
df_cleaned = df.dropna()  # Drops any row with NaN
print(df_cleaned)
```
✅ Best for: When you want to remove incomplete records.


#### ✅ Comparison of Methods

| Method                  | Best Used For                                   |
|-------------------------|-----------------------------------------------|
| `fillna(value)`        | Replacing missing values with a default       |
| `fillna(df.mean())`    | Filling missing values with mean/median       |
| `fillna(method='ffill')` | Filling with previous value (forward fill)    |
| `fillna(method='bfill')` | Filling with next value (backward fill)      |
| `interpolate()`        | Estimating missing values for numeric data    |
| `dropna()`            | Removing rows/columns with missing values     |


### 📌 Handling Missing Values in Categorical Columns

#### 🔹 What Are Categorical Columns?
*Categorical columns* contain discrete values that represent different categories or labels. They do not hold continuous numerical values but rather *groups* of related values.

##### Examples of Categorical Columns:

* Gender: ['Male', 'Female', 'Other']
* Department: ['HR', 'Finance', 'IT', 'Marketing']
* City: ['New York', 'San Francisco', 'Chicago']
* Education Level: ['High School', 'Bachelor', 'Master', 'PhD']

When dealing with missing categorical data, you *cannot* use methods like *mean or interpolation*. Instead, here are some practical approaches:

#### 1️⃣ Replace NaN with a Default Category:
The simplest approach is to replace NaN with *"Unknown" or "Other"*.

```python
import pandas as pd
import numpy as np

# Creating a DataFrame with missing categorical values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Department': ['HR', 'Finance', np.nan, 'IT'],
    'City': ['New York', np.nan, 'Chicago', 'San Francisco']
})

# Fill missing values with a default category
df.fillna({'Department': 'Unknown', 'City': 'Not Provided'}, inplace=True)

print(df)
```

✅ Best for: When missing values represent an unknown or irrelevant category.
 
-----

#### 2️⃣ Replace NaN with the Most Frequent Value (Mode):

If a category appears frequently, filling missing values with the most common category is a good option.

```python
# Fill missing values with the most frequent category (mode)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)

print(df)
```
#### How It Works:
* .mode() finds the most frequently occurring value in the column.
* [0] selects the first mode (in case multiple values have the highest frequency).
* fillna() replaces missing (NaN) values with the mode.

```python
import pandas as pd
import numpy as np

# Sample DataFrame with missing categorical values
df = pd.DataFrame({
    'Department': ['HR', 'Finance', 'IT', 'HR', np.nan, 'IT', np.nan],
    'City': ['New York', 'London', np.nan, 'New York', 'London', np.nan, 'London']
})

# Fill missing values with the most frequent category (mode)
df['Department'].fillna(df['Department'].mode()[0], inplace=True)
df['City'].fillna(df['City'].mode()[0], inplace=True)

print(df)
```

![Image](https://github.com/user-attachments/assets/32f988fd-bf0a-42cd-8bea-4dfadfd083eb)


✅ When to Use This?
 
    ✔️ When missing values are not random and depend on a pattern.

    ✔️ When categorical data has a dominant class (e.g., most employees work in 'HR').

    ✔️ When dropping rows is not an option due to small dataset size.

❌ When to Avoid Mode Imputation?
    
    ⚠️ If the dataset is highly imbalanced, mode imputation can bias the model towards the dominant class.
    ⚠️ If missing values are randomly distributed, consider 'Unknown' instead of mode.

##### ✅ Best for: When you want to preserve data trends by using the most common category.

------- 

#### 3️⃣ Use Forward or Backward Fill (ffill or bfill):

For ordered categorical data (e.g., education levels or ranks), forward or backward filling can help.
```python
df.fillna(method='ffill', inplace=True)  # Forward fill
# df.fillna(method='bfill', inplace=True)  # Backward fill (alternative)

print(df)
```

✅ Best for: When previous or next values logically make sense.

--------


#### 4️⃣ Drop Rows with Missing Categorical Values
If missing data is small and insignificant, you can remove those rows.
```python
df_cleaned = df.dropna()
print(df_cleaned)
```

✅ Best for: When only a few records have missing data.

--------------



✅ Comparison of Methods for Categorical Data:

| Method                  | Best Used For                                      |
|-------------------------|---------------------------------------------------|
| `fillna("Unknown")`     | When missing values represent unknown categories  |
| `fillna(df.mode()[0])`  | Replacing with the most common category           |
| `fillna(method='ffill')` | Carrying forward previous values                 |
| `dropna()`             | Removing incomplete rows                           |



### 🔍 Comparison of Categorical Data Imputation Techniques
When handling missing values in categorical columns, different methods can be used based on the dataset and problem context. Here's a comparison:

#### 1️⃣ Mode Imputation (Most Frequent Value)
*Concept:* Replaces missing values with the most frequently occurring category.


```python
df['Department'].fillna(df['Department'].mode()[0], inplace=True)
```


✔️ Pros:
    
    ✅ Simple and quick to implement.
    ✅ Retains existing categories.

❌ Cons:

    ❌ Can introduce bias if one category dominates.
    ❌ Not ideal for balanced datasets with diverse values.

💡 Best for: Columns where a single category occurs frequently (e.g., "HR" department is dominant).

### 2️⃣ Fill with a Placeholder Category ('Unknown')
*Concept:* Assigns a new category like 'Unknown' to missing values.


```python
df['Department'].fillna('Unknown', inplace=True)
```



✔️ Pros:

    ✅ Keeps missing values distinguishable.
    ✅ Prevents bias from dominant categories.
    ✅ Works well with One-Hot Encoding.

❌ Cons:
    
    ❌ Might introduce a category that does not naturally exist.

💡 Best for: When missing values are truly unknown and shouldn't be assigned to an existing category.

### 3️⃣ Predictive Imputation (Using ML Models)
 *Concept:* Uses a model (like Decision Trees) to predict the missing category based on other available features.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
df['Department_encoded'] = encoder.fit_transform(df['Department'].astype(str))

imputer = SimpleImputer(strategy='most_frequent')
df['Department_encoded'] = imputer.fit_transform(df[['Department_encoded']])

df['Department'] = encoder.inverse_transform(df['Department_encoded'])
```

✔️ Pros:

    ✅ More accurate than simple replacements.
    ✅ Uses patterns in the data to determine missing values.

❌ Cons:
    
    ❌ Requires additional computation.
    ❌ Might introduce model bias if trained on limited data.

💡 Best for: When other columns provide clues about missing categories (e.g., Job Title, Salary).

### 4️⃣ Drop Rows with Missing Values
*Concept:* Removes rows that contain missing categorical values.

```python
df.dropna(subset=['Department'], inplace=True)
```


✔️ Pros:

    ✅ Ensures only complete data is used.

❌ Cons:

    ❌ Risky for small datasets, as it reduces data size.

💡 Best for: When missing values are random and the dataset is large enough to handle data loss.

📝 Which One Should You Use?

| Method                | Best For                                       | Drawbacks                  |
|-----------------------|----------------------------------------------|----------------------------|
| Mode (Most Frequent Value) | When a dominant category exists           | Can introduce bias.        |
| 'Unknown' Category   | When missing values should be distinguished  | Adds a new category.       |
| ML-Based Prediction  | When data patterns suggest missing values    | Computationally expensive. |
| Dropping Rows        | When dataset is large & missing data is random | Reduces dataset size.      |


#### 🚀 Conclusion
* If you want a quick fix → Mode Imputation ✅
* If you want to preserve missing values distinctly → Fill with ‘Unknown’ ✅
* If you want higher accuracy → ML-Based Imputation ✅
* If you have a large dataset & missing values are random → Drop Rows ✅





