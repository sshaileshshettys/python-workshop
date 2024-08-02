
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

**Example 1: Creating and Accessing Series**
```python
# Create a Series
data = [100, 200, 300]
index = ['x', 'y', 'z']
series = pd.Series(data, index=index)

# Accessing elements
print("Series element 'y':", series['y'])
```

**Example 2: Creating and Manipulating DataFrames**
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

**Example 3: Reading Data from a CSV File**
```python
# Assuming 'data.csv' exists in the working directory
df = pd.read_csv('data.csv')
print("DataFrame from CSV:\n", df.head())
```

**Example 4: Data Filtering and Aggregation**
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
**Exercise 1: Titanic Dataset Analysis with Pandas**
- This project involves analyzing the Titanic dataset to gain insights into passenger demographics and survival rates. The Titanic dataset contains information about passengers aboard the RMS Titanic, including their age, class, and survival status. By exploring and visualizing this data, we can uncover patterns related to survival rates and demographic factors.

#### Dataset

The Titanic dataset used in this analysis can be downloaded from [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data).
#### Loading and Exploring the Dataset
```python

import pandas as pd
df_titanic = pd.read_csv('data/titanic.csv')
```
#### Explore the dataset
```python
print(df_titanic.info())
print(df_titanic.describe())
```
#### Check for missing values and Fill missing age values with median
```python
print(df_titanic.isnull().sum())
df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True)
```
#### Drop columns with too many missing values
```python
df_titanic.drop(columns=['Cabin'], inplace=True)
```
#### Average age by passenger class
```python
average_age_by_class = df_titanic.groupby('Pclass')['Age'].mean()
print(average_age_by_class)
```
#### Survival rate by gender
```python
survival_rate_by_gender = df_titanic.groupby('Sex')['Survived'].mean()
print(survival_rate_by_gender)
```
#### Data Visualization 
```python
import matplotlib.pyplot as plt
df_titanic['Pclass'].value_counts().plot(kind='bar')
plt.title('Number of Passengers by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()
```
**Exercise 2: Wine Quality Dataset Analysis with Pandas**
- This project involves analyzing the Wine Quality dataset to explore the characteristics of red wines and their quality ratings. The dataset includes attributes such as fixed acidity, volatile acidity, citric acid, residual sugar, and more, along with a quality score for each wine sample. The analysis covers data loading, cleaning, manipulation, aggregation, and visualization.
#### Dataset 
The Wine Quality dataset used in this analysis can be downloaded from [Kaggle Wine Quality Dataset](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009).


#### Load the Dataset
```python
import pandas as pd
df_wine = pd.read_csv('data/winequality-red.csv', delimiter=';')
print(df_wine.head())
```
#### Explore and Clean the Data
```python
print(df_wine.info())
print(df_wine.describe())

# Check for missing values
print(df_wine.isnull().sum())

# Drop duplicates if any
df_wine.drop_duplicates(inplace=True)
```
#### Data Manipulation
- Normalization
```python
from sklearn.preprocessing import MinMaxScaler

# Normalizing the features (Min-Max Scaling)
scaler = MinMaxScaler()
df_wine_normalized = df_wine.copy()
df_wine_normalized[df_wine.columns[:-1]] = scaler.fit_transform(df_wine[df_wine.columns[:-1]])

print(df_wine_normalized.head())
```
- Feature Engineering
```python
   # Adding a new feature 'High_Quality' based on a threshold
df_wine['High_Quality'] = df_wine['quality'] >= 7
print(df_wine[['quality', 'High_Quality']].head())
```
#### Data Aggregation
```python
# Grouping by 'High_Quality' and calculating average of other features
avg_quality_by_type = df_wine.groupby('High_Quality').mean()
print(avg_quality_by_type)
```
####  Data Visualization
- Correlation Matrix
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
corr = df_wine.corr()

# Generate a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix of Wine Quality Dataset')
plt.show()
```
- Distribution of Wine Quality Scores
```python
# Plotting the distribution of wine quality scores
plt.figure(figsize=(8, 6))
sns.histplot(df_wine['quality'], bins=10, kde=True)
plt.title('Distribution of Wine Quality Scores')
plt.xlabel('Quality')
plt.ylabel('Frequency')
plt.show()
```
- Feature Selection
```python
# Calculate correlation with the target variable 'quality'
correlations = df_wine.corr()['quality'].sort_values(ascending=False)
print(correlations)
```
**Exercise 3: Iris Dataset Analysis with Pandas**

- This project involves analyzing the Iris dataset to understand the characteristics of iris flowers and their species. The Iris dataset includes measurements of sepal length, sepal width, petal length, and petal width, along with the species of each iris flower. The analysis includes data cleaning, manipulation, aggregation, and visualization.

#### Dataset

The Iris dataset used in this analysis can be downloaded from [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris).

#### Loading and Exploring the Dataset

```python
import pandas as pd

# Load Iris dataset
df_iris = pd.read_csv('data/iris.csv')
print(df_iris.head())
```
#### Handling Missing Values
```python
# Check for missing values
print(df_iris.isnull().sum())
```
#### Data Manipulation
##### Standardization
```python
from sklearn.preprocessing import StandardScaler

# Standardizing the features
scaler = StandardScaler()
df_iris_standardized = df_iris.copy()
df_iris_standardized[df_iris.columns[:-1]] = scaler.fit_transform(df_iris[df_iris.columns[:-1]])

print(df_iris_standardized.head())
```

##### Creating New Features
```python
# Create a new feature 'Petal_Size' as the product of petal length and width
df_iris['Petal_Size'] = df_iris['petal_length'] * df_iris['petal_width']
print(df_iris[['petal_length', 'petal_width', 'Petal_Size']].head())
```
#### Data Aggregation
##### Average Measurements by Species
```python
# Group by 'species' and calculate the mean of the features
average_measurements_by_species = df_iris.groupby('species').mean()
print(average_measurements_by_species)
```
#### Data Visualization
##### Pair Plot
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Pair plot of the dataset
sns.pairplot(df_iris, hue='species')
plt.title('Pair Plot of Iris Dataset')
plt.show()

```
##### Histogram of Petal Length
```python
# Plotting the histogram of petal length
plt.figure(figsize=(8, 6))
sns.histplot(df_iris['petal_length'], bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length')
plt.ylabel('Frequency')
plt.show()
```


**Exercise 4: Boston Housing Dataset Analysis with Pandas**
- This project involves analyzing the Boston Housing dataset to explore housing values and their related features. The dataset includes attributes such as crime rate, average number of rooms, and distance to employment centers. The analysis includes data loading, cleaning, manipulation, aggregation, and visualization.

#### Dataset
The Boston Housing dataset used in this analysis can be downloaded from [Kaggle Boston Housing Dataset](https://www.kaggle.com/datasets/camnugent/boston-housing-dataset).
#### Loading and Exploring the Dataset

```python
import pandas as pd

# Load Boston Housing dataset
df_boston = pd.read_csv('data/boston_housing.csv')
print(df_boston.head())
```
#### Handling Missing Values
```python
# Check for missing values
print(df_boston.isnull().sum())
```
#### Data Manipulation
##### Feature Engineering
```python
# Create a new feature 'Crime_Per_Room' as the ratio of crime rate to average number of rooms
df_boston['Crime_Per_Room'] = df_boston['CRIM'] / df_boston['RM']
print(df_boston[['CRIM', 'RM', 'Crime_Per_Room']].head())
```
##### Normalization

from sklearn.preprocessing import MinMaxScaler
```python
# Normalize the 'PRICE' feature
scaler = MinMaxScaler()
df_boston['PRICE'] = scaler.fit_transform(df_boston[['PRICE']])
print(df_boston[['PRICE']].head())
```
#### Data Aggregation
##### Average Housing Prices by Number of Rooms
```python
# Group by 'RM' and calculate the average 'PRICE'
average_price_by_rooms = df_boston.groupby('RM')['PRICE'].mean()
print(average_price_by_rooms)
```
#### Data Visualization
##### Scatter Plot of Crime Rate vs. Housing Prices
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Scatter plot of crime rate vs. housing prices
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_boston, x='CRIM', y='PRICE')
plt.title('Crime Rate vs. Housing Prices')
plt.xlabel('Crime Rate')
plt.ylabel('Normalized Housing Price')
plt.show()
```
##### Distribution of Average Number of Rooms
```python
# Plotting the histogram of average number of rooms
plt.figure(figsize=(8, 6))
sns.histplot(df_boston['RM'], bins=20, kde=True)
plt.title('Distribution of Average Number of Rooms')
plt.xlabel('Average Number of Rooms')
plt.ylabel('Frequency')
plt.show()
```
### Analysis Questions

#### Dataset

Download the dataset from the link below 👇:

- [Mall Customer Segmentation Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Global Happiness Dataset](https://www.kaggle.com/unsdsn/world-happiness)
- [COVID-19 Data](https://www.kaggle.com/datasets/robertopl/coronavirus-covid19-cases)
- [Supermarket Sales Dataset](https://www.kaggle.com/aungpyaeap/supermarket-sales)
- [Movie Ratings Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset) 
#### Instructions

##### 1. Load and Explore the Dataset

- Load the dataset into a Pandas DataFrame.
- Display the first few rows of the dataset.
- Check the data types of each column and provide summary statistics.

##### 2. Handle Missing Values

- Identify if there are any missing values in the dataset.
- Propose methods to handle any missing values found.

##### 3. Feature Engineering

- Create a new feature that combines or transforms existing features. For example, you could create a feature representing the ratio of age to annual income.
- Normalize a numeric feature of your choice and compare the normalized values to the original values.

##### 4. Data Aggregation

- Group the dataset by a categorical feature and compute the average of a numeric feature. For example, compute the average spending score for different age groups or income brackets.
- Analyze the results and describe any patterns observed.

##### 5. Data Visualization

- Generate a scatter plot showing the relationship between two numeric features, such as age and spending score.
- Create a histogram to visualize the distribution of a numeric feature, such as annual income.



