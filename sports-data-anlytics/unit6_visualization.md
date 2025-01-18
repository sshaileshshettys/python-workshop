# Unit 6: Sport Data Visualization

## Overview

Data visualization is a crucial aspect of sports analytics. With the increasing amount of data available in modern sports, the ability to clearly communicate insights through visual representations has become indispensable. Effective data visualization not only aids in better understanding of complex datasets but also helps in decision-making, performance analysis, and fan engagement.

This unit will cover the various techniques used to visualize sports data, including basic plots, advanced visualizations, and interactive dashboards. We will use Python libraries such as **Matplotlib**, **Seaborn**, and **Plotly** to create a range of visualizations, along with practical applications for analyzing player performance, team statistics, and match outcomes.

---

## Topics Covered:
1. **Introduction to Data Visualization**
2. **Basic Visualization Techniques**
3. **Advanced Visualization Techniques**
4. **Creating Interactive Dashboards**
5. **Visualizing Player and Team Performance**
6. **Heatmaps and Correlation Matrices**
7. **Time-Series Data Visualization**
8. **Case Study: Visualizing Match Outcomes and Player Stats**

---

## 1. Introduction to Data Visualization

Data visualization is the graphical representation of data and information. In sports analytics, the goal is to provide clear and insightful visualizations that reveal patterns, trends, and correlations within the data. Effective visualizations can help teams, coaches, analysts, and fans make better decisions and gain a deeper understanding of the game.

**Key Benefits of Data Visualization in Sports:**

- **Simplifies complex data**: Transforms raw data into easily digestible insights.
- **Enhances decision-making**: Helps stakeholders make data-driven decisions.
- **Improves performance analysis**: Visualizing player and team stats provides insights into strengths and weaknesses.
- **Boosts fan engagement**: Eye-catching visuals can be shared on social media to engage fans.

---

## 2. Basic Visualization Techniques

### a. Line Plot

A **line plot** is used to visualize trends over time, such as player performance or team statistics across multiple seasons.

```python
import matplotlib.pyplot as plt

# Sample data: Player goals over several matches
matches = ['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5']
goals = [1, 2, 0, 3, 1]

plt.figure(figsize=(8, 6))
plt.plot(matches, goals, marker='o', color='b', linestyle='-', linewidth=2)
plt.title('Player Goals Over Matches')
plt.xlabel('Matches')
plt.ylabel('Goals')
plt.show()
```

### b. Bar Plot

A **bar plot** is useful for comparing quantities across different categories, such as goals scored by players.

```python
import seaborn as sns
import pandas as pd

# Sample data: Goals scored by different players
data = {'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
        'Goals': [10, 15, 7, 20]}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.barplot(x='Player', y='Goals', data=df, palette='viridis')
plt.title('Goals Scored by Players')
plt.show()
```

### c. Histogram

A **histogram** is used to display the distribution of a variable, such as the distribution of player goals in a season.

```python
# Sample data: Goals scored by players in a season
goals = [10, 15, 7, 20, 25, 14, 18, 12, 10, 11]

plt.figure(figsize=(8, 6))
sns.histplot(goals, bins=5, kde=True, color='green')
plt.title('Distribution of Player Goals')
plt.xlabel('Goals')
plt.ylabel('Frequency')
plt.show()
```

---

## 3. Advanced Visualization Techniques

### a. Scatter Plot

A **scatter plot** helps visualize the relationship between two continuous variables, such as goals scored vs. assists for a player.

```python
# Sample data: Goals and Assists for players
data = {'Player': ['Player A', 'Player B', 'Player C', 'Player D'],
        'Goals': [10, 15, 7, 20],
        'Assists': [5, 8, 3, 10]}
df = pd.DataFrame(data)

plt.figure(figsize=(8, 6))
sns.scatterplot(x='Goals', y='Assists', data=df, hue='Player', palette='coolwarm')
plt.title('Goals vs Assists')
plt.xlabel('Goals')
plt.ylabel('Assists')
plt.show()
```

### b. Box Plot

A **box plot** is useful for visualizing the distribution and identifying outliers in datasets, such as the number of goals scored by players in a season.

```python
# Sample data: Goals scored by players across different teams
goals = [10, 15, 7, 20, 25, 14, 18, 12, 10, 11]
teams = ['Team A', 'Team B', 'Team A', 'Team B', 'Team A', 'Team B', 'Team A', 'Team B', 'Team A', 'Team B']

df = pd.DataFrame({'Goals': goals, 'Team': teams})

plt.figure(figsize=(8, 6))
sns.boxplot(x='Team', y='Goals', data=df, palette='Blues')
plt.title('Goals Scored by Team')
plt.show()
```

---

## 4. Creating Interactive Dashboards

**Plotly** and **Dash** are Python libraries used for creating interactive and dynamic visualizations that can be embedded in web applications.

### a. Interactive Line Plot with Plotly

```python
import plotly.express as px

# Sample data: Player goals over matches
data = {'Match': ['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5'],
        'Goals': [1, 2, 0, 3, 1]}

df = pd.DataFrame(data)

fig = px.line(df, x='Match', y='Goals', title='Player Goals Over Matches')
fig.show()
```

### b. Interactive Bar Plot with Plotly

```python
fig = px.bar(df, x='Player', y='Goals', title='Goals Scored by Players')
fig.show()
```

---

## 5. Visualizing Player and Team Performance

### a. Radar Chart for Player Comparison

A **radar chart** allows for comparing multiple attributes of different players in a single view. For example, comparing player statistics like goals, assists, and passing accuracy.

```python
import numpy as np

# Sample data: Player stats (Goals, Assists, Passing Accuracy)
labels = ['Goals', 'Assists', 'Passing Accuracy']
values = [15, 8, 85]  # Player A stats

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()

# Complete the loop by repeating the first value at the end
values += values[:1]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.25)
ax.plot(angles, values, color='red', linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)

plt.title('Player A Performance')
plt.show()
```

---

## 6. Heatmaps and Correlation Matrices

Heatmaps are great for visualizing correlations between various performance metrics, such as the relationship between goals, assists, and minutes played.

```python
# Sample data: Player statistics
df = pd.DataFrame({
    'Goals': [10, 15, 7, 20],
    'Assists': [5, 8, 3, 10],
    'Minutes': [800, 1000, 700, 1200]
})

corr_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Player Statistics')
plt.show()
```

---

## 7. Time-Series Data Visualization

Sports analytics often involves time-series data, such as tracking the performance of a player or team over time.

### a. Time-Series Plot

```python
# Sample data: Player performance over several seasons
seasons = ['2019', '2020', '2021', '2022']
goals = [18, 22, 19, 26]

plt.figure(figsize=(8, 6))
plt.plot(seasons, goals, marker='o', color='blue')
plt.title('Player Goals Over Seasons')
plt.xlabel('Season')
plt.ylabel('Goals')
plt.show()
```

---

## 8. Case Study: Visualizing Match Outcomes and Player Stats

In this case study, we will visualize the performance of a player across different matches, including goals scored, assists, and minutes played.

```python
# Sample data: Player performance in different matches
matches = ['Match 1', 'Match 2', 'Match 3', 'Match 4']
goals = [1, 0, 2, 1]
assists = [0, 1, 1, 0]
minutes = [90, 80, 85, 90]

df = pd.DataFrame({'Match': matches, 'Goals': goals, 'Assists': assists, 'Minutes': minutes})

plt.figure(figsize=(10, 6))
sns.lineplot(data=df

, x='Match', y='Goals', label='Goals', color='blue', marker='o')
sns.lineplot(data=df, x='Match', y='Assists', label='Assists', color='green', marker='o')
plt.title('Player Performance Across Matches')
plt.ylabel('Count')
plt.xlabel('Match')
plt.legend()
plt.show()
```

---

## Conclusion

Data visualization is a key tool in sports analytics, helping teams, coaches, and fans interpret complex data and make informed decisions. Whether through simple plots or advanced interactive dashboards, effective visualizations can reveal insights that improve performance and engagement.
