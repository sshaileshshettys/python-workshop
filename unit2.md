

## Unit 2: Knowing the Jupyter Notebook and Jupyter Lab, Numpy

### 1. Introduction to Jupyter Notebook and Jupyter Lab

#### Introduction to Jupyter Notebook

**What is Jupyter Notebook?**
- **Definition and Purpose**: Jupyter Notebook is an open-source web application for creating and sharing documents containing live code, equations, visualizations, and narrative text.
- **Key Features**: Interactive computing, rich text support, integration with various visualization libraries.

**Installation and Setup**
```bash
# Installing Jupyter Notebook
pip install notebook

# Starting Jupyter Notebook
jupyter notebook
```

**Basic Usage and Navigation**
- **Creating a New Notebook**: Click on `New` and select `Python 3` from the Jupyter Notebook dashboard.
- **Writing and Executing Code**: Type code into a cell and press `Shift + Enter` to run it.
- **Using Markdown**: Switch to Markdown to add explanations and notes.
- **Saving and Exporting**: Save your work via `File > Save and Checkpoint`, and export using `File > Download as`.

**Code Sample:**
```python
# Python code cell
print("Hello, Jupyter Notebook!")

# Markdown cell
# This is a Markdown cell where you can document your work.
```

#### Introduction to Jupyter Lab

**What is Jupyter Lab?**
- **Definition and Comparison**: Jupyter Lab is an advanced interface for Project Jupyter, offering a more flexible and interactive environment compared to Jupyter Notebook.
- **Key Features**: Multiple document interface, integrated development environment, extensibility with plugins.

**Installation and Setup**
```bash
# Installing Jupyter Lab
pip install jupyterlab

# Starting Jupyter Lab
jupyter lab
```

**Basic Usage and Navigation**
- **Overview of the Interface**: Includes a file browser, text editor, terminal, and notebook panels.
- **Creating and Managing Notebooks**: Use `File > New > Notebook` to create notebooks and manage them via the file browser.

**Code Sample:**
```python
# Python code cell in Jupyter Lab
print("Welcome to Jupyter Lab!")

# Markdown cell in Jupyter Lab
# This is a Markdown cell where you can document your work.
```

### 2. Introduction to Numpy

**Basics of Numpy**

**What is Numpy?**
- **Definition and Importance**: Numpy is a fundamental package for scientific computing in Python, providing support for arrays, matrices, and mathematical functions.
- **Key Features**: Efficient array operations, mathematical functions, broadcasting, and advanced indexing.

**Installation and Setup**
```bash
# Installing Numpy
pip install numpy

# Importing Numpy
import numpy as np
```

**Numpy Arrays and Matrices**

**Creating Numpy Arrays**
```python
import numpy as np

# One-dimensional array
arr1 = np.array([1, 2, 3, 4, 5])
print("One-dimensional array:", arr1)

# Multi-dimensional array
arr2 = np.array([[1, 2, 3], [4, 5, 6]])
print("Two-dimensional array:\n", arr2)
```

**Basic Operations with Numpy Arrays**
```python
# Indexing and slicing
print("Element at index 2:", arr1[2])
print("Slice from index 1 to 4:", arr1[1:4])

# Reshaping arrays
arr3 = np.arange(12).reshape(3, 4)
print("Reshaped array:\n", arr3)
```

**Matrix Operations**
```python
# Creating matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Basic matrix operations
print("Matrix addition:\n", matrix1 + matrix2)
print("Matrix subtraction:\n", matrix1 - matrix2)
print("Matrix multiplication:\n", np.dot(matrix1, matrix2))

# Transpose and inverse
print("Transpose:\n", np.transpose(matrix1))
print("Inverse:\n", np.linalg.inv(matrix1))
```

**Mathematical and Statistical Operations with Numpy**

**Mathematical Functions**
```python
# Common functions
print("Sine of array elements:\n", np.sin(arr1))
print("Exponential of array elements:\n", np.exp(arr1))

# Aggregation functions
print("Sum of array elements:", np.sum(arr1))
print("Mean of array elements:", np.mean(arr1))
print("Median of array elements:", np.median(arr1))
print("Standard deviation of array elements:", np.std(arr1))
```

**Statistical Functions**
```python
# Generating random numbers
rand_array = np.random.rand(5)
print("Random array:", rand_array)

# Statistical summaries
print("Min value:", np.min(arr1))
print("Max value:", np.max(arr1))
print("Percentile (50th):", np.percentile(arr1, 50))
```

### 3. Data Analysis and Visualization

**Introduction to Matplotlib**

**What is Matplotlib?**
- **Definition and Purpose**: Matplotlib is a plotting library for Python that provides a wide range of data visualization options.
- **Key Features**: Versatile plotting, customization options, integration with Pandas.

**Installation and Setup**
```bash
# Installing Matplotlib
pip install matplotlib
```

**Basic Plotting with Matplotlib**
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3, 4], [10, 20, 25, 30])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Simple Line Plot')
plt.show()

# Scatter plot
x = np.random.rand(50)
y = np.random.rand(50)
plt.scatter(x, y)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatter Plot')
plt.show()
```

**Advanced Plotting**
```python
# Histogram
data = np.random.randn(1000)
plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')
plt.show()

# Subplots
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].plot([1, 2, 3], [4, 5, 6])
axs[0].set_title('Subplot 1')
axs[1].bar(['A', 'B', 'C'], [10, 20, 15])
axs[1].set_title('Subplot 2')
plt.show()
```

### 4. Introduction to Pandas

**What is Pandas?**
- **Definition and Purpose**: Pandas is a data manipulation and analysis library for Python. It provides data structures and functions needed to work on structured data seamlessly.
- **Key Features**: DataFrames, data manipulation, data cleaning.

**Installation and Setup**
```bash
# Installing Pandas
pip install pandas
```

**Basic Data Manipulation with Pandas**
```python
import pandas as pd

# Creating a DataFrame
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Sales': [250, 150, 300, 200]
}
df = pd.DataFrame(data)
print("DataFrame:\n", df)

# Basic DataFrame operations
print("Summary Statistics:\n", df.describe())
print("Column Names:", df.columns)
print("Rows where Sales > 200:\n", df[df['Sales'] > 200])
```

**Data Cleaning and Aggregation**
```python
# Handling missing values
df = pd.DataFrame({
    'Product': ['A', 'B', 'C', 'D'],
    'Sales': [250, np.nan, 300, 200]
})
df.fillna(df['Sales'].mean(), inplace=True)
print("DataFrame with missing values filled:\n", df)

# Grouping and Aggregating
grouped_df = df.groupby('Product').agg({'Sales': 'sum'})
print("Grouped DataFrame:\n", grouped_df)
```

### 5. Practical Case Studies and Exercises

**Case Study 1: Sales Data Analysis**
- **Objective**: Analyze sales data to determine trends and generate reports.
```python
# Load sales data
sales_df = pd.read_csv('sales_data.csv')  # Assume this file exists

# Analyze data
total_sales = sales_df['Sales'].sum()
average_sales = sales_df['Sales'].mean()
print(f"Total Sales: {total_sales}")
print(f"Average Sales: {average_sales}")

# Plot sales trends
sales_df.plot(x='Date', y='Sales', kind='line')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Sales Trend Over Time')
plt.show()
```

**Case Study 2: Customer Segmentation**
- **Objective**: Segment customers based on purchasing behavior and visualize segments.
```python
# Load customer data
customer_df = pd.read_csv('customer_data.csv')  # Assume this file exists

# Perform clustering (e.g., KMeans)
from sklearn.cluster import KMeans

# Assume we have a 'Spending' column
X

 = customer_df[['Spending']].values
kmeans = KMeans(n_clusters=3)
customer_df['Segment'] = kmeans.fit_predict(X)

# Visualize segments
plt.scatter(customer_df['Spending'], customer_df['Income'], c=customer_df['Segment'])
plt.xlabel('Spending')
plt.ylabel('Income')
plt.title('Customer Segmentation')
plt.show()
```

**Hands-On Exercises**

**Exercise 1: Interactive Data Visualization**
- **Objective**: Create an interactive plot with Matplotlib and ipywidgets.
```python
import ipywidgets as widgets
from IPython.display import display

def interactive_plot(freq):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(freq * x)
    plt.plot(x, y)
    plt.title(f'Sine Wave with Frequency {freq}')
    plt.show()

freq_slider = widgets.FloatSlider(value=1.0, min=0.1, max=10.0, step=0.1, description='Frequency:')
widgets.interactive(interactive_plot, freq=freq_slider)
```

**Exercise 2: Data Cleaning and Analysis with Pandas**
- **Objective**: Load a dataset, clean it, and perform basic analysis.
```python
# Load and clean data
df = pd.read_csv('dataset.csv')  # Assume this file exists
df.dropna(inplace=True)  # Remove rows with missing values

# Perform analysis
summary = df.describe()
print("Data Summary:\n", summary)
```

**Exercise 3: Visualizing Business Metrics**
- **Objective**: Create visualizations to represent business metrics effectively.
```python
# Create sample data
metrics_df = pd.DataFrame({
    'Month': ['Jan', 'Feb', 'Mar', 'Apr'],
    'Revenue': [5000, 6000, 7000, 8000]
})

# Plot revenue trends
metrics_df.plot(x='Month', y='Revenue', kind='bar')
plt.xlabel('Month')
plt.ylabel('Revenue')
plt.title('Monthly Revenue')
plt.show()
```

#### Advanced Jupyter Notebook Features(Optional)

**Magic Commands**
- **Definition**: Special commands in Jupyter Notebook that start with `%` or `%%` for specific tasks.
```python
# Timing a code execution
%timeit np.arange(10000).sum()

# Run a shell command
!ls
```

**Creating and Using Notebooks with Git Integration**
- **Git Integration**: Use Git to version control Jupyter notebooks.
```bash
# Initialize a Git repository
git init

# Add your Jupyter notebooks to version control
git add *.ipynb

# Commit changes
git commit -m "Initial commit of Jupyter notebooks"
```

**Customizing Jupyter Notebooks**
- **Themes and Extensions**: Install and apply themes to change the notebook's appearance.
```bash
# Install Jupyter themes
pip install jupyterthemes

# Apply a theme
jt -t <theme-name>
```

**Code Sample:**
```python
# Magic command to time execution
%timeit np.random.rand(10000).mean()

# Magic command to run shell command
!echo "Hello from the shell"
```

#### Advanced Jupyter Lab Features

**Using the Terminal in Jupyter Lab**
- **Terminal Commands**: Access and use the command line interface within Jupyter Lab.
```bash
# Check Python version from the terminal
python --version
```

**Creating Custom Extensions**
- **Building Extensions**: Develop custom extensions for Jupyter Lab.
```bash
# Install Jupyter Lab extension tools
pip install jupyterlab
pip install jupyterlab_server
```

**Code Sample:**
```bash
# Open a terminal in Jupyter Lab and check disk usage
df -h
```

### 2. Introduction to Numpy

**Advanced Numpy Array Operations**

**Broadcasting Rules**
- **Explanation**: Broadcasting allows Numpy to perform operations on arrays of different shapes.
```python
# Broadcasting example
a = np.array([1, 2, 3])
b = np.array([[10], [20], [30]])
result = a + b
print("Broadcasting result:\n", result)
```

**Advanced Indexing and Slicing**
- **Fancy Indexing and Boolean Indexing**: Use advanced indexing techniques for selecting array elements.
```python
# Fancy indexing
arr = np.arange(10)
indices = [1, 3, 5]
print("Fancy indexing result:", arr[indices])

# Boolean indexing
mask = arr % 2 == 0
print("Boolean indexing result:", arr[mask])
```

**Performance Optimization**
- **Vectorization vs. Loops**: Use Numpy's vectorized operations for performance efficiency.
```python
# Vectorized operation
arr = np.arange(1, 10001)
result = np.sin(arr) * np.log(arr)

# Loop-based operation (slower)
result_loop = [np.sin(x) * np.log(x) for x in arr]
```

### 3. Introduction to Pandas

**Advanced Data Manipulation with Pandas**

**Data Aggregation and Grouping**
- **Aggregation Functions**: Perform aggregation operations such as sum, mean, and count.
```python
# Load data
df = pd.DataFrame({
    'Department': ['HR', 'Sales', 'IT', 'HR', 'IT', 'Sales'],
    'Salary': [50000, 60000, 70000, 55000, 75000, 65000]
})

# Group by department and aggregate
grouped = df.groupby('Department').agg({'Salary': ['mean', 'sum']})
print("Grouped DataFrame:\n", grouped)
```

**Handling Time Series Data**
- **Date and Time Operations**: Work with time series data using Pandas' datetime functionality.
```python
# Create time series data
dates = pd.date_range(start='2023-01-01', periods=6, freq='M')
data = pd.Series([100, 200, 300, 400, 500, 600], index=dates)
print("Time Series Data:\n", data)

# Resampling data
monthly_data = data.resample('M').sum()
print("Resampled Monthly Data:\n", monthly_data)
```

### 4. Data Visualization with Matplotlib and Seaborn

**Introduction to Seaborn**

**What is Seaborn?**
- **Definition and Purpose**: Seaborn is a Python visualization library based on Matplotlib that provides a high-level interface for drawing attractive statistical graphics.

**Installation and Setup**
```bash
# Installing Seaborn
pip install seaborn
```

**Basic Plotting with Seaborn**
```python
import seaborn as sns

# Load example dataset
df = sns.load_dataset('tips')

# Scatter plot
sns.scatterplot(data=df, x='total_bill', y='tip', hue='day')
plt.title('Tip Amount vs Total Bill')
plt.show()

# Box plot
sns.boxplot(data=df, x='day', y='total_bill')
plt.title('Total Bill by Day')
plt.show()
```

**Advanced Plotting with Seaborn**
```python
# Pairplot
sns.pairplot(df, hue='day')
plt.show()

# Heatmap
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

### 5. Practical Case Studies and Exercises

**Case Study 1: Sales and Revenue Forecasting**
- **Objective**: Use historical sales data to forecast future sales using linear regression.
```python
from sklearn.linear_model import LinearRegression

# Load sales data
sales_df = pd.read_csv('sales_data.csv')  # Assume this file exists

# Prepare data
sales_df['Date'] = pd.to_datetime(sales_df['Date'])
sales_df.set_index('Date', inplace=True)
sales_df['Month'] = sales_df.index.month
sales_df['Year'] = sales_df.index.year

# Feature engineering
X = sales_df[['Month', 'Year']]
y = sales_df['Sales']

# Train a linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict future sales
future_dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
future_X = pd.DataFrame({
    'Month': future_dates.month,
    'Year': future_dates.year
})
predictions = model.predict(future_X)

print("Forecasted Sales:\n", predictions)
```

**Case Study 2: Customer Lifetime Value Analysis**
- **Objective**: Analyze and predict customer lifetime value (CLV) using historical data.
```python
# Load customer data
customer_df = pd.read_csv('customer_data.csv')  # Assume this file exists

# Calculate CLV based on historical purchase data
customer_df['CLV'] = customer_df['Purchase Amount'] * customer_df['Purchase Frequency']
print("Customer Lifetime Value:\n", customer_df[['Customer ID', 'CLV']].head())

# Visualize CLV distribution
sns.histplot(customer_df['CLV'], bins=30, kde=True)
plt.title('Distribution of Customer Lifetime Value')
plt.show()
```

**Hands-On Exercises**

**Exercise 1: Interactive Data Visualization with Seaborn**
- **Objective**: Create interactive plots with Seaborn for data exploration.
```python
import seaborn as sns

# Load dataset
df = sns.load_dataset('iris')

# Pairplot with interactive widgets
import ipywidgets as widgets
from IPython.display import display

def plot_species(species):
    subset = df[df['species'] == species]
    sns.pairplot(subset, hue='species')
    plt.show()

species_dropdown = widgets.Dropdown(options=df['species'].unique(), description='Species:')
widgets.interactive(plot_species, species=species_dropdown)
```

**Exercise 2: Data Cleaning and Visualization**
- **Objective**: Clean and visualize data from a provided dataset.
```python
# Load and clean data
df = pd.read_csv('data.csv')  # Assume this file exists
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])

# Plot cleaned data
df.plot(x='Date', y='Value', kind='line')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Cleaned Data Over Time')
plt.show()
```

**Exercise 3: Predictive Modeling**
- **Objective**: Build a predictive model for a business-related dataset.
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data_df = pd.read_csv('business_data.csv')  # Assume this file exists

# Prepare features and target variable
X = data_df[['Feature1', 'Feature2', 'Feature3']]
y = data_df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")
```
