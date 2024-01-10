import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Set the backend to 'TkAgg'

import matplotlib.pyplot as plt
import seaborn as sns

# Ensuring plots are displayed inline in Jupyter Notebooks
#%matplotlib inline
# Load datasets (make sure to replace the file paths with the correct ones)
movies_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/movies.csv")
ratings_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/ratings.csv")
tags_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/tags.csv")


# Calculate average rating and number of ratings for each movie
ratings_summary = ratings_df.groupby('movieId').agg(average_rating=('rating', 'mean'), number_of_ratings=('rating', 'count'))

# Merge the ratings summary with the movies dataframe
movies_with_ratings = movies_df.merge(ratings_summary, on='movieId')

# Sort the movies by number of ratings and get the top 10
top_movies = movies_with_ratings.sort_values(by='number_of_ratings', ascending=False).head(10)

# Generate a pie chart for the top 10 movies
plt.figure(figsize=(10, 8))
plt.pie(top_movies['number_of_ratings'], labels=top_movies['title'], autopct='%1.1f%%', startangle=140)
plt.title('Top 10 Movies by Number of Ratings')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

# Save the plot to a file
plt.savefig('top_movies_pie_chart.png')

# Display the pie chart
plt.show()


# Average rating per movie
avg_ratings = ratings_df.groupby('movieId')['rating'].mean()

# Histogram of average ratings
plt.figure(figsize=(10, 6))
sns.histplot(avg_ratings, bins=30, kde=False)
plt.title('Distribution of Average Movie Ratings')
plt.xlabel('Average Rating')
plt.ylabel('Number of Movies')
plt.show()

# Number of ratings per movie
num_ratings = ratings_df.groupby('movieId')['rating'].count()

# Histogram of number of ratings
plt.figure(figsize=(10, 6))
sns.histplot(num_ratings, bins=30, kde=False)
plt.title('Distribution of Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Movies')
plt.show()

# Counting genres
genre_counts = movies_df['genres'].str.split('|').explode().value_counts()

# Bar chart of genre counts
plt.figure(figsize=(12, 8))
genre_counts.head(10).plot(kind='bar')
plt.title('Top 10 Movie Genres')
plt.xlabel('Genre')
plt.ylabel('Number of Movies')
plt.show()


# User behavior analysis
user_behavior_df = ratings_df.groupby('userId').agg(num_ratings=('rating', 'count'), average_rating=('rating', 'mean'))

# Scatter plot of user ratings count vs average rating
plt.figure(figsize=(10, 6))
sns.scatterplot(data=user_behavior_df, x='num_ratings', y='average_rating')
plt.title('User Rating Behavior')
plt.xlabel('Number of Ratings')
plt.ylabel('Average Rating')
plt.show()

# Histogram of number of ratings by user
plt.figure(figsize=(10, 6))
sns.histplot(user_behavior_df['num_ratings'], bins=30)
plt.title('Distribution of Number of Ratings by Users')
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
plt.show()


# Most common tags
top_tags = tags_df['tag'].value_counts()

# Bar chart of top tags
plt.figure(figsize=(12, 8))
top_tags.head(10).plot(kind='bar')
plt.title('Top 10 Tags')
plt.xlabel('Tag')
plt.ylabel('Frequency')
plt.show()
