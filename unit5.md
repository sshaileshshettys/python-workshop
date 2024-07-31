## Unit 5: Analytics using Python

### 1. Introduction to Analytics in Python

**What is Analytics?**
- **Definition:**
  - Analytics involves the discovery, interpretation, and communication of meaningful patterns in data.
- **Importance:**
  - Helps in making data-driven decisions
  - Provides insights into trends and patterns
  - Supports strategic planning and operational efficiency

### 2. Installing and Using Python Packages for Analytics

**Installing Packages via pip**

**Code Sample:**
```bash
pip install pandas numpy scipy scikit-learn statsmodels matplotlib seaborn
```

### 3. Data Analysis with Pandas and Numpy

**1. Pandas for Data Manipulation**

Pandas is a powerful library for data manipulation and analysis.

**Code Sample:**
```python
import pandas as pd

# Sample data
data = {
    'Product': ['A', 'B', 'C', 'D'],
    'Sales': [100, 200, 300, 400],
    'Revenue': [1000, 1500, 2000, 2500]
}
df = pd.DataFrame(data)

# Basic DataFrame operations
print("DataFrame:\n", df)
print("Summary Statistics:\n", df.describe())
```

**2. Numpy for Numerical Operations**

Numpy is essential for numerical computations and working with arrays.

**Code Sample:**
```python
import numpy as np

# Sample data
array = np.array([1, 2, 3, 4, 5])

# Basic operations
mean = np.mean(array)
std_dev = np.std(array)
print("Mean:", mean)
print("Standard Deviation:", std_dev)

# Mathematical functions
squared = np.square(array)
print("Squared Values:", squared)
```

### 4. Statistical Analysis with Scipy and Statsmodels

**1. Scipy for Statistical Functions**

Scipy provides a wide range of statistical functions.

**Code Sample:**
```python
from scipy import stats

# Sample data
data = [2.3, 3.4, 3.2, 4.5, 5.1]

# Calculating mean and standard deviation
mean = np.mean(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Standard Deviation:", std_dev)

# Performing a t-test
t_statistic, p_value = stats.ttest_1samp(data, 3.0)
print("T-statistic:", t_statistic)
print("P-value:", p_value)
```

**2. Statsmodels for Regression Analysis**

Statsmodels is used for statistical modeling and regression analysis.

**Code Sample:**
```python
import statsmodels.api as sm
import pandas as pd

# Sample data
data = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y': [2.2, 2.8, 3.6, 4.5, 5.1]
})

# Regression model
X = sm.add_constant(data['X'])  # Adding a constant term for the intercept
model = sm.OLS(data['Y'], X).fit()

# Summary of the model
print(model.summary())
```

### 5. Machine Learning with Scikit-Learn

**1. Data Preprocessing**

Scikit-Learn is used for machine learning and includes tools for data preprocessing.

**Code Sample:**
```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Sample data
data = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])

# Standardization
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print("Scaled Data:\n", scaled_data)
```

**2. Implementing a Machine Learning Model**

**Code Sample:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1.1, 2.0, 3.1, 4.2, 5.0])

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

### 6. Time Series Analysis

**1. Time Series Decomposition**

**Code Sample:**
```python
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Sample time series data
dates = pd.date_range(start='2021-01-01', periods=12, freq='M')
data = pd.Series([10, 12, 13, 12, 14, 16, 18, 17, 16, 14, 15, 16], index=dates)

# Decomposing the time series
result = seasonal_decompose(data, model='additive')

# Plotting the decomposition
result.plot()
plt.show()
```

### Hands-On Exercises and Examples

**Exercise 1: Data Cleaning and Aggregation with Pandas**
```python
import pandas as pd

# Sample data
data = {
    'City': ['A', 'B', 'A', 'B'],
    'Temperature': [25, 30, 22, 28],
    'Humidity': [60, 65, 55, 70]
}
df = pd.DataFrame(data)

# Aggregating data
aggregated = df.groupby('City').mean()
print("Aggregated Data:\n", aggregated)
```

**Exercise 2: Statistical Analysis with Scipy**
```python
from scipy import stats

# Sample data
data = [1.2, 2.3, 2.9, 3.1, 4.0]

# Descriptive statistics
mean = np.mean(data)
median = np.median(data)
print("Mean:", mean)
print("Median:", median)

# Hypothesis testing
t_stat, p_value = stats.ttest_1samp(data, 2.5)
print("T-statistic:", t_stat)
print("P-value:", p_value)
```

**Exercise 3: Linear Regression with Scikit-Learn**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 3, 5, 7, 11])

# Training the model
model = LinearRegression()
model.fit(X, y)

# Making predictions
y_pred = model.predict(X)

# Model evaluation
r2 = r2_score(y, y_pred)
print("R-squared:", r2)
```

**Exercise 4: Time Series Analysis**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample time series data
data = pd.Series([1, 3, 2, 5, 6, 8, 7, 9, 10], 
                 index=pd.date_range(start='2024-01-01', periods=9, freq='D'))

# Plotting the time series
data.plot()
plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
```
