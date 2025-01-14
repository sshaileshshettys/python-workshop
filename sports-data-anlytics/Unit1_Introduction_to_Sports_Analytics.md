# Unit 1: Introduction to Sports Analytics

## Overview
This unit provides an introduction to sports analytics, covering its meaning, definition, and importance. It explores the data revolution in sports and recent trends across various sports such as athletics, soccer, and cricket.

---

## Learning Objectives
By the end of this unit, students will be able to:

- Define sports analytics.
- Explain the significance of sports analytics.
- Analyze the recent trends in sports analytics.
- Implement programming techniques for sports data analysis.
- Understand and visualize key performance indicators (KPIs) in sports.

---

## Content

### 1. What is Sports Analytics?
Sports analytics refers to the use of data and statistical models to:

- Evaluate player performance.
- Predict game outcomes.
- Develop strategies for improving team performance.
- Optimize training regimens and reduce injury risks.

### 2. Importance of Sports Analytics
- **Player Evaluation**: Identifying strengths and weaknesses.
- **Team Strategy**: Making informed decisions during matches.
- **Fan Engagement**: Enhancing viewer experience with insights.
- **Injury Prevention**: Using data to monitor player health and minimize injuries.

### 3. Recent Trends in Sports Analytics
- The rise of tracking technologies (e.g., GPS, RFID).
- Real-time data analysis during games.
- Advanced metrics for player performance (e.g., Expected Goals (xG) in soccer).
- Use of AI and machine learning to improve predictions.
- Wearable tech to collect player-specific data.

---

## Advanced Programming Examples with Dataset

### Dataset: Player Performance Data
```python
import pandas as pd

# Creating a sample dataset
player_data = pd.DataFrame({
    'Player': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Games Played': [10, 12, 11, 13, 10],
    'Total Points': [250, 300, 280, 310, 260],
    'Assists': [50, 60, 55, 65, 52],
    'Rebounds': [80, 90, 85, 95, 88]
})

print(player_data)
```

### Example 1: Calculating Advanced Metrics
```python
# Calculate Points Per Game
player_data['Points Per Game'] = player_data['Total Points'] / player_data['Games Played']

# Calculate Efficiency Rating
player_data['Efficiency'] = (player_data['Total Points'] + player_data['Assists'] + player_data['Rebounds']) / player_data['Games Played']

print(player_data[['Player', 'Points Per Game', 'Efficiency']])
```

### Example 2: Visualizing Player Performance
```python
import matplotlib.pyplot as plt

# Bar chart for Points Per Game
plt.figure(figsize=(10, 6))
plt.bar(player_data['Player'], player_data['Points Per Game'], color='skyblue')
plt.title('Points Per Game by Player')
plt.xlabel('Player')
plt.ylabel('Points Per Game')
plt.show()
```

### Example 3: Correlation Analysis
```python
# Correlation between metrics
correlation_matrix = player_data[['Total Points', 'Assists', 'Rebounds']].corr()
print("Correlation Matrix:\n", correlation_matrix)
```

### Example 4: Predicting Performance with Scikit-Learn
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Features and target variable
X = player_data[['Games Played', 'Assists', 'Rebounds']]
y = player_data['Total Points']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
predictions = model.predict(X_test)
print("Predictions:", predictions)
```

---

## Competencies
- Define and discuss the core principles of sports analytics.
- Apply basic and advanced programming techniques to sports contexts.
- Analyze trends and patterns using data visualization and statistical tools.
- Perform predictive modeling for sports outcomes.
- Extract actionable insights from real-world datasets.
