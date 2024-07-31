
## Unit 3: Introduction to Pandas

### 1. Introduction to Pandas

**What is Pandas?**
- **Definition and Importance:**
  - Pandas is a powerful data manipulation and analysis library built on top of NumPy.
  - It provides data structures like Series and DataFrame for handling and analyzing data efficiently.
- **Key Features and Benefits:**
  - Data cleaning and preparation
  - Data wrangling and manipulation
  - Data analysis and visualization integration

**Installation and Setup**
- **Installing Pandas:**
  - You can install Pandas using pip or through Anaconda.

**Code Sample:**
```bash
pip install pandas
```

- **Importing Pandas in Python:**

**Code Sample:**
```python
import pandas as pd
```

### 2. Working with Pandas Data Structures

**Series**

A `Series` is a one-dimensional array-like object that can hold any data type. It is similar to a column in a spreadsheet.

**Code Sample:**
```python
# Creating a Series
data = [10, 20, 30, 40]
index = ['a', 'b', 'c', 'd']
series = pd.Series(data, index=index)

print("Series:\n", series)

# Accessing data
print("Element with index 'b':", series['b'])

# Basic operations
print("Series plus 5:\n", series + 5)
```

**DataFrame**

A `DataFrame` is a two-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).

**Code Sample:**
```python
# Creating a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df = pd.DataFrame(data)

print("DataFrame:\n", df)

# Accessing data
print("Column 'Age':\n", df['Age'])
print("Row 1:\n", df.loc[1])

# Adding a new column
df['Salary'] = [50000, 60000, 70000]
print("DataFrame with new column:\n", df)
```

### 3. Data Manipulation with Pandas

**Reading Data**

Pandas can read data from various file formats including CSV, Excel, and SQL databases.

**Code Sample:**
```python
# Reading from CSV
df = pd.read_csv('data.csv')
print("DataFrame from CSV:\n", df)

# Reading from Excel
df = pd.read_excel('data.xlsx')
print("DataFrame from Excel:\n", df)
```

**Data Selection and Filtering**

You can select and filter data using various methods.

**Code Sample:**
```python
# Selecting columns
print("Names Column:\n", df['Name'])

# Selecting rows by index
print("First Row:\n", df.iloc[0])

# Filtering rows
print("Rows where Age > 30:\n", df[df['Age'] > 30])
```

**Data Cleaning**

Cleaning data is an essential part of data analysis.

**Code Sample:**
```python
# Handling missing values
df = pd.DataFrame({
    'Name': ['Alice', 'Bob', None],
    'Age': [25, None, 35]
})

# Filling missing values
df_filled = df.fillna({'Name': 'Unknown', 'Age': df['Age'].mean()})
print("DataFrame with missing values filled:\n", df_filled)

# Dropping rows with missing values
df_dropped = df.dropna()
print("DataFrame with missing values dropped:\n", df_dropped)
```

**Data Aggregation and Grouping**

Pandas allows you to perform data aggregation and grouping operations.

**Code Sample:**
```python
# Aggregation
df = pd.DataFrame({
    'Category': ['A', 'B', 'A', 'B', 'A'],
    'Value': [10, 20, 10, 30, 40]
})

# Grouping and aggregating
grouped = df.groupby('Category').sum()
print("Grouped DataFrame:\n", grouped)
```

### 4. Data Visualization with Pandas

Pandas integrates well with data visualization libraries like Matplotlib.

**Code Sample:**
```python
import matplotlib.pyplot as plt

# Plotting a Series
series.plot(kind='bar')
plt.title('Series Data')
plt.show()

# Plotting a DataFrame
df.plot(x='Name', y='Age', kind='bar')
plt.title('Age of Individuals')
plt.show()
```

### Hands-On Exercises and Examples

**Exercise 1: Creating and Accessing Series**
```python
# Create a Series
data = [100, 200, 300]
index = ['x', 'y', 'z']
series = pd.Series(data, index=index)

# Accessing elements
print("Series element 'y':", series['y'])
```

**Exercise 2: Creating and Manipulating DataFrames**
```python
# Create a DataFrame
data = {
    'Product': ['A', 'B', 'C'],
    'Price': [20, 30, 40]
}
df = pd.DataFrame(data)

# Add a new column
df['Quantity'] = [100, 200, 300]

# Display DataFrame
print("DataFrame:\n", df)
```

**Exercise 3: Reading Data from a CSV File**
```python
# Assuming 'data.csv' exists in the working directory
df = pd.read_csv('data.csv')
print("DataFrame from CSV:\n", df.head())
```

**Exercise 4: Data Filtering and Aggregation**
```python
# Create a DataFrame
df = pd.DataFrame({
    'Department': ['HR', 'Finance', 'IT', 'HR', 'IT'],
    'Salary': [50000, 60000, 70000, 55000, 72000]
})

# Filter data
filtered = df[df['Salary'] > 60000]
print("Filtered DataFrame:\n", filtered)

# Group and aggregate data
grouped = df.groupby('Department').mean()
print("Grouped DataFrame:\n", grouped)
```