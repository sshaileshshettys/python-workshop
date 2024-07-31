## Unit 4: Data Visualization in Python

### 1. Introduction to Data Visualization

**What is Data Visualization?**
- **Definition:**
  - The graphical representation of information and data.
- **Importance:**
  - Helps in understanding complex data sets
  - Makes data insights easier to comprehend
  - Aids in identifying trends, patterns, and outliers

### 2. Visualization Libraries in Python

**1. Matplotlib**

Matplotlib is one of the most widely used libraries for creating static, animated, and interactive visualizations in Python.

**Installation:**
```bash
pip install matplotlib
```

**Basic Usage:**

**Code Sample:**
```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

# Create a simple line plot
plt.plot(x, y, marker='o', linestyle='-', color='b')
plt.title('Line Plot Example')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

**2. Seaborn**

Seaborn is built on top of Matplotlib and provides a high-level interface for drawing attractive and informative statistical graphics.

**Installation:**
```bash
pip install seaborn
```

**Basic Usage:**

**Code Sample:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = sns.load_dataset('tips')

# Create a scatter plot
sns.scatterplot(x='total_bill', y='tip', data=data, hue='day')
plt.title('Scatter Plot Example')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.legend(title='Day')
plt.show()
```

**3. Plotly**

Plotly is a graphing library that makes interactive, publication-quality graphs online.

**Installation:**
```bash
pip install plotly
```

**Basic Usage:**

**Code Sample:**
```python
import plotly.express as px

# Sample data
data = px.data.iris()

# Create an interactive scatter plot
fig = px.scatter(data, x='sepal_width', y='sepal_length', color='species', size='petal_length')
fig.update_layout(title='Interactive Scatter Plot Example')
fig.show()
```

**4. Plotnine**

Plotnine is a grammar of graphics library for Python that allows for a powerful way to create complex plots.

**Installation:**
```bash
pip install plotnine
```

**Basic Usage:**

**Code Sample:**
```python
from plotnine import ggplot, aes, geom_point, labs

# Sample data
from plotnine.data import mtcars

# Create a scatter plot
p = ggplot(mtcars, aes(x='wt', y='mpg', color='cyl')) + \
    geom_point() + \
    labs(title='Scatter Plot Example', x='Weight', y='Miles per Gallon')

print(p)
```

### 3. Creating Various Types of Visualizations

**1. Line Plot**

**Code Sample:**
```python
import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

plt.plot(x, y, marker='o')
plt.title('Line Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
```

**2. Bar Chart**

**Code Sample:**
```python
import matplotlib.pyplot as plt

# Sample data
categories = ['A', 'B', 'C', 'D']
values = [10, 15, 7, 12]

plt.bar(categories, values, color='skyblue')
plt.title('Bar Chart')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

**3. Histogram**

**Code Sample:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Sample data
data = np.random.randn(1000)

plt.hist(data, bins=30, color='purple', edgecolor='black')
plt.title('Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

**4. Box Plot**

**Code Sample:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = sns.load_dataset('iris')

# Create a box plot
sns.boxplot(x='species', y='sepal_length', data=data)
plt.title('Box Plot')
plt.xlabel('Species')
plt.ylabel('Sepal Length')
plt.show()
```

**5. Heatmap**

**Code Sample:**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = sns.load_dataset('flights').pivot("month", "year", "passengers")

# Create a heatmap
sns.heatmap(data, cmap='YlGnBu', annot=True)
plt.title('Heatmap')
plt.show()
```

**6. Pie Chart**

**Code Sample:**
```python
import matplotlib.pyplot as plt

# Sample data
sizes = [20, 30, 25, 25]
labels = ['A', 'B', 'C', 'D']

plt.pie(sizes, labels=labels, autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral', 'lightskyblue'])
plt.title('Pie Chart')
plt.show()
```

### Hands-On Exercises and Examples

**Exercise 1: Line Plot with Seaborn**
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = sns.load_dataset('flights').pivot("month", "year", "passengers")

# Create a line plot
sns.lineplot(data=data)
plt.title('Line Plot of Flights Data')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.show()
```

**Exercise 2: Bar Chart with Matplotlib**
```python
import matplotlib.pyplot as plt

# Sample data
categories = ['X', 'Y', 'Z']
values = [5, 10, 15]

plt.bar(categories, values, color=['red', 'blue', 'green'])
plt.title('Bar Chart Example')
plt.xlabel('Categories')
plt.ylabel('Values')
plt.show()
```

**Exercise 3: Interactive Plot with Plotly**
```python
import plotly.express as px

# Sample data
data = px.data.gapminder()

# Create an interactive scatter plot
fig = px.scatter(data, x='gdpPercap', y='lifeExp', color='continent', size='pop', animation_frame='year')
fig.update_layout(title='Interactive Scatter Plot of GDP vs Life Expectancy')
fig.show()
```

**Exercise 4: Box Plot with Plotnine**
```python
from plotnine import ggplot, aes, geom_boxplot, labs
from plotnine.data import mtcars

# Create a box plot
p = ggplot(mtcars, aes(x='cyl', y='mpg')) + \
    geom_boxplot() + \
    labs(title='Box Plot of Miles per Gallon by Cylinder', x='Number of Cylinders', y='Miles per Gallon')

print(p)
```
