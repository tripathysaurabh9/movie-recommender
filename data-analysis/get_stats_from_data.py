import pandas as pd

# Load datasets (make sure to replace the file paths with the correct ones)
movies_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/movies.csv")
ratings_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/ratings.csv")
tags_df = pd.read_csv("/Users/stripathy/Everything/take-home/take-home/data/tags.csv")

# 1. Movie Popularity and Ratings
# Calculate average rating and number of ratings for each movie
avg_ratings = ratings_df.groupby('movieId')['rating'].mean()
num_ratings = ratings_df.groupby('movieId')['rating'].count()

# Merging this data with the movies dataframe
movie_ratings_df = movies_df.set_index('movieId').join(avg_ratings.rename('average_rating'))
movie_ratings_df = movie_ratings_df.join(num_ratings.rename('num_ratings'))

# 2. Genre Preferences
# Extracting and counting genres
genre_counts = movies_df['genres'].str.split('|').explode().value_counts()

# 3. User Behavior
# Analyzing user rating patterns
user_ratings_count = ratings_df.groupby('userId')['rating'].count()
user_avg_ratings = ratings_df.groupby('userId')['rating'].mean()

# Combining the count and average of ratings for each user
user_behavior_df = pd.DataFrame({'num_ratings': user_ratings_count, 'average_rating': user_avg_ratings})

# 4. Tag Analysis
# Analyzing the most common tags
top_tags = tags_df['tag'].value_counts()

# Outputting the results (you can adjust this part to save the results to files or display them as needed)
print("Top Movies by Ratings:")
print(movie_ratings_df.sort_values(by='num_ratings', ascending=False).head(10)[['title', 'average_rating', 'num_ratings']])
print("\nTop Genres:")
print(genre_counts.head(10))
print("\nTop Users by Number of Ratings:")
print(user_behavior_df.sort_values(by='num_ratings', ascending=False).head(10))
print("\nTop Tags:")
print(top_tags.head(10))
