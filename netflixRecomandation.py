# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 22:24:00 2025

@author: vivin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import ipywidgets as widgets

# Load the datasets
movies_df = pd.read_csv('D:\\Movies.csv')
ratings_df = pd.read_csv('D:\\Ratings.csv')

# Display the first few rows of the data
print(movies_df.head())
print(ratings_df.head())
# Exploratory Data Analysis

# Distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(ratings_df['rating'], bins=10, kde=True)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()

# Number of unique users and movies
print(f"Unique Users: {ratings_df['userId'].nunique()}")
print(f"Unique Movies: {ratings_df['movieId'].nunique()}")

# Average rating for each movie
average_movie_ratings = ratings_df.groupby('movieId')['rating'].mean().reset_index()
average_movie_ratings = average_movie_ratings.merge(movies_df[['movieId', 'title']], on='movieId', how='left')

# Number of ratings for each movie
ratings_count = ratings_df.groupby('movieId')['rating'].count().reset_index()
ratings_count = ratings_count.rename(columns={'rating': 'num_ratings'})
average_movie_ratings = average_movie_ratings.merge(ratings_count, on='movieId', how='left')

# Top genres
genres = movies_df['genres'].str.split('|', expand=True).stack().unique()
print(f"Unique Genres: {genres}")
def popularity_based_recommender(genre, min_reviews, num_recommendations):
    # Filter movies by genre
    genre_movies = movies_df[movies_df['genres'].str.contains(genre)]
    
    # Merge with average ratings and review count
    genre_ratings = average_movie_ratings[average_movie_ratings['movieId'].isin(genre_movies['movieId'])]
    
    # Apply minimum reviews threshold
    genre_ratings = genre_ratings[genre_ratings['num_ratings'] >= min_reviews]
    
    # Sort by average rating in descending order
    genre_ratings = genre_ratings.sort_values(by='rating', ascending=False)
    
    # Select top N recommendations
    top_recommendations = genre_ratings.head(num_recommendations)
    
    return top_recommendations[['title', 'rating', 'num_ratings']]

# Example usage
genre = "Comedy"
min_reviews = 100
num_recommendations = 5
popularity_based_recommender(genre, min_reviews, num_recommendations)
def content_based_recommender(movie_title, num_recommendations):
    # Get the genres of the movie
    movie_genres = movies_df[movies_df['title'] == movie_title]['genres'].values[0]
    
    # Find movies with similar genres
    similar_movies = movies_df[movies_df['genres'].str.contains(movie_genres)]
    
    # Rank by average rating
    similar_movies = similar_movies.merge(average_movie_ratings[['movieId', 'rating']], on='movieId')
    similar_movies = similar_movies.sort_values(by='rating', ascending=False)
    
    # Return top N recommendations
    return similar_movies[['title']].head(num_recommendations)

# Example usage
movie_title = "Toy Story"
num_recommendations = 5
content_based_recommender(movie_title, num_recommendations)
def collaborative_based_recommender(user_id, num_recommendations, k_threshold):
    # Pivot the ratings data to create user-item matrix
    user_movie_ratings = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
    
    # Compute similarity between users
    similarity_matrix = cosine_similarity(user_movie_ratings)
    
    # Get similar users to the target user
    user_similarities = similarity_matrix[user_id - 1]  # userId is 1-based, so subtract 1 for indexing
    similar_users = user_similarities.argsort()[-k_threshold:][::-1]
    
    # Get movies rated highly by similar users
    recommended_movies = defaultdict(int)
    
    for similar_user in similar_users:
        similar_user_ratings = ratings_df[ratings_df['userId'] == (similar_user + 1)]
        for _, row in similar_user_ratings.iterrows():
            recommended_movies[row['movieId']] += row['rating']
    
    # Sort by total rating and get top N recommendations
    recommended_movie_ids = sorted(recommended_movies, key=recommended_movies.get, reverse=True)[:num_recommendations]
    
    # Get the movie titles
    recommended_titles = movies_df[movies_df['movieId'].isin(recommended_movie_ids)]['title']
    return recommended_titles

# Example usage
user_id = 1
num_recommendations = 5
k_threshold = 100
collaborative_based_recommender(user_id, num_recommendations, k_threshold)




genre_input = widgets.Text(description="Genre:")
min_reviews_input = widgets.IntText(description="Min Reviews:")
num_recommendations_input = widgets.IntText(description="Recommendations:")
button = widgets.Button(description="Get Recommendations")

def on_button_click(b):
    genre = genre_input.value
    min_reviews = min_reviews_input.value
    num_recommendations = num_recommendations_input.value
    recommendations = popularity_based_recommender(genre, min_reviews, num_recommendations)
    print(recommendations)

button.on_click(on_button_click)

display(genre_input, min_reviews_input, num_recommendations_input, button)
