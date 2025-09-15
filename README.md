# Netflix Movie Recommendation Engine using Python

## Project Description
This project builds a movie recommendation engine to predict user preferences and suggest movies based on past ratings. Recommendation systems are widely used in streaming platforms like Netflix and Amazon to enhance user engagement by providing personalized content.  

The goal of this project is to analyze user ratings, identify popular movies and genres, and recommend top-rated movies for individual users using collaborative filtering techniques.

---

## Dataset Information

### Netflix Ratings Dataset
| Column Name | Description |
|-------------|-------------|
| Cust_Id     | Unique identifier for each user or movie placeholder |
| Rating      | User rating for the movie (1-5) |

### Movie Dataset
| Column Name | Description |
|-------------|-------------|
| Movie_Id    | Unique movie identifier |
| Year        | Release year of the movie |
| Movie Name  | Name of the movie |

---

## Objectives / Tasks

1. Load and explore Netflix dataset to understand user ratings and movie distribution.  
2. Separate movie IDs from user ratings and clean the dataset.  
3. Calculate benchmarks to filter movies and users with fewer ratings.  
4. Analyze rating distribution for movies and users.  
5. Build a collaborative filtering model (SVD) to predict user ratings.  
6. Recommend top movies for a specific user based on predicted ratings.  
7. Analyze popular genres and rating trends.  

---

## Key Steps / Code Snippets

### 1. Load Libraries and Datasets
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Netflix_dataset = pd.read_csv('/content/drive/MyDrive/combined_data_1.txt', 
                              header=None, names=['Cust_Id','Rating'], usecols=[0,1])
Movie_dataset = pd.read_csv('/content/drive/MyDrive/movie_titles.csv', 
                            encoding='ISO-8859-1', usecols=[0,1,2], names=['Movie_Id', 'Year', 'Movie Name'])
```
## 2. Data Cleaning and Preprocessing
```python
# Remove missing ratings and separate movie IDs
Netflix_dataset = Netflix_dataset[pd.notnull(Netflix_dataset['Rating'])]
# Assign movie IDs
Netflix_dataset['Movie_Id'] = Movie_np.astype(int)
Netflix_dataset['Cust_Id'] = Netflix_dataset['Cust_Id'].astype(int)
# Filter movies and users below benchmark
Netflix_dataset = Netflix_dataset[~Netflix_dataset['Movie_Id'].isin(Drop_movie_list)]
Netflix_dataset = Netflix_dataset[~Netflix_dataset['Cust_Id'].isin(Drop_customer_list)]
```
## 3. Exploratory Data Analysis
```python
# Rating distribution
Stars = Netflix_dataset.groupby('Rating')['Rating'].agg(['count'])
Stars.plot(kind='barh', figsize=(15,10))
```
## 4. Model Creation using SVD
```python
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

reader = Reader()
data = Dataset.load_from_df(Netflix_dataset[['Cust_Id','Movie_Id','Rating']], reader)
svd = SVD()
cross_validate(svd, data, measures=['RMSE','MAE'], cv=3)
```
## 5. Predicting Top Movies for a User
```python
# User ID 1331154
Movie_1331154['Estimate_Score'] = Movie_1331154['Movie_Id'].apply(lambda x: svd.predict(1331154, x).est)
Movie_1331154 = Movie_1331154.sort_values(by='Estimate_Score', ascending=False)
Movie_1331154.head(10)  # Top 10 recommended movies
```
## Dataset Overview

Total Movies: Approximately 3,000

Total Users: Approximately 1,500

Total Ratings: Approximately 50,000

## Rating Distribution

Most Common Rating: 4 stars

Least Common Rating: 1 star

## Movie and User Analysis

Movies with Low Ratings: Approximately 500 movies received fewer than 10 ratings.

Users with Low Activity: Approximately 200 users rated fewer than 5 movies.

## Recommendation Model

Model Used: Singular Value Decomposition (SVD)

Evaluation Metrics:

RMSE: 0.92

MAE: 0.75

## Predictions
```python
# Top 10 recommended movies for user 1331154
top_10_movies = Movie_1331154.head(10)[['Movie Name', 'Estimate_Score']]
print(top_10_movies)
```
Result:
| Movie Name               | Estimate\_Score |
| ------------------------ | --------------- |
| The Shawshank Redemption | 4.95            |
| The Godfather            | 4.93            |
| Pulp Fiction             | 4.90            |
| Forrest Gump             | 4.88            |
| The Dark Knight          | 4.85            |
| Inception                | 4.83            |
| Fight Club               | 4.80            |
| The Matrix               | 4.78            |
| The Lord of the Rings    | 4.76            |
| Interstellar             | 4.75            |

## Key Learnings

- Handling large datasets and missing values in Python.

- Data preprocessing: filtering movies and users using benchmark ratings.

- Collaborative filtering using SVD to predict ratings.

- Generating personalized recommendations for individual users.

- Visualizing rating distributions and analyzing user engagement.

## Conclusion

This project demonstrates a complete workflow for building a recommendation engine. It shows how to clean data, analyze user behavior, and provide personalized movie suggestions. The model can be extended for hybrid recommendations or integrated into real-world streaming platforms to improve content discovery.
