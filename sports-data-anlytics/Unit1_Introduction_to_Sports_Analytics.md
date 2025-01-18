# Unit 1: Introduction to Sports Analytics

## Overview

In this unit, we will introduce you to the field of sports analytics, its importance, and how data is transforming the way sports teams, athletes, and fans interact with sports. Sports analytics involves gathering data from various sources, analyzing it to derive insights, and applying these insights to improve performance, enhance fan experience, and make more informed decisions.

We will explore different types of data, the ways it can be collected, and the methods for analyzing and interpreting it. You will also learn the basic tools required to perform sports analytics, including Python and various libraries for data manipulation and visualization.

### Topics Covered
1. **What is Sports Analytics?**
2. **Types of Data in Sports**
3. **Importance of Sports Analytics**
4. **Basic Python for Data Analysis**
5. **Getting Started with Sports Datasets**

---

## 1. What is Sports Analytics?

Sports analytics is the process of collecting, analyzing, and interpreting data from sports events. This data can be related to player performance, team strategies, game outcomes, or fan behavior. The goal is to gain insights that can help improve decisions related to training, game preparation, fan engagement, and more.

### Key Areas in Sports Analytics:
- **Player Evaluation**: Analyzing player statistics to assess performance.
- **Team Strength Measurement**: Understanding team dynamics and performance through data.
- **Outcome Prediction**: Using historical data to predict future game outcomes.
- **Fan Engagement**: Enhancing the experience of fans using data-driven strategies.

---

## 2. Types of Data in Sports

Sports data can be classified into several categories:
- **Performance Data**: Information about player stats (e.g., goals, assists, minutes played).
- **Game Data**: Results, scores, and other details about matches or games.
- **Biometric Data**: Physical and health data, such as heart rate, steps, etc.
- **Fan Data**: Demographic and engagement data about fans.

### Example of Performance Data for a Soccer Player

| Player Name | Goals | Assists | Minutes Played |
|-------------|-------|---------|----------------|
| Lionel Messi| 30    | 10      | 2700           |
| Cristiano Ronaldo | 25 | 5    | 2500           |

---

## 3. Importance of Sports Analytics

The role of data in sports is growing, and analytics can provide teams with a competitive edge. It helps in:
- **Improving player performance**: Identifying strengths and weaknesses.
- **Strategic decision-making**: Enhancing team performance through data-driven tactics.
- **Optimizing fan engagement**: Understanding fan preferences and behavior to enhance their experience.
- **Predicting game outcomes**: Analyzing historical data to make predictions on future games.

---



Sure! Here's an expanded and more comprehensive section with additional Python code examples for data analysis, making it more practical and varied for learning purposes.

---

## 4. Basic Python for Data Analysis

Python has become a popular tool in sports analytics due to its vast ecosystem of libraries and ease of use. In this section, we'll cover some basic concepts and show how Python can be used for analyzing sports data. We will focus on libraries like **Pandas**, **NumPy**, **Matplotlib**, and **Seaborn**.
[Notebook click here](./notebooks/unit1.ipynb).
---

## Assignment

1. **Data Cleaning**:
   - Load a new dataset and check for missing data.
   - Handle missing data by filling or dropping the rows.

2. **Exploratory Data Analysis**:
   - Calculate the mean, median, and standard deviation of goals and assists.
   - Create a correlation matrix between all numeric columns in the dataset.
   - Plot a pairplot to visualize relationships between numeric variables like goals, assists, and minutes played.

3. **Data Visualization**:
   - Create a line plot showing the total goals scored over time (if you have data for multiple seasons).
   - Generate a heatmap to visualize correlations between player stats like goals, assists, and minutes played.

4. **Grouping and Aggregation**:
   - Group the data by `Position` and calculate the total goals and average minutes played for each position.
   - Identify the player with the highest goals per match and plot a bar chart for the top 5 players.

5. **Advanced (Optional)**:
   - Build a simple linear regression model to predict goals based on minutes played using **scikit-learn**.
   - Evaluate the model's performance with mean squared error (MSE) and R² score.



## 5. Getting Started with Sports Datasets

### Sample Dataset: Soccer Player Stats

In this section, we will use a sample dataset containing player statistics to perform basic analysis. The dataset might include columns like "Player Name," "Goals," "Assists," and "Minutes Played." You can download the dataset from [Kaggle - Soccer Player Stats](https://www.kaggle.com).

### Dataset Example:

| Player Name     | Goals | Assists | Minutes Played |
|-----------------|-------|---------|----------------|
| Lionel Messi    | 30    | 10      | 2700           |
| Cristiano Ronaldo| 25   | 5       | 2500           |
| Neymar          | 20    | 7       | 2300           |

---


## Conclusion

In this unit, we covered the fundamentals of sports analytics, including the types of data used in the field, the importance of data-driven decisions, and the tools you will use to perform sports analysis. You should now have a basic understanding of how data is used in sports and how Python helps in analyzing and visualizing this data.

Happy coding and see you in the next unit!
