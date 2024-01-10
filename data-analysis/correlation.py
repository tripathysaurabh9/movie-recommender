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


# Convert timestamp to datetime and extract the month
ratings_df['datetime'] = pd.to_datetime(ratings_df['timestamp'], unit='s')
ratings_df['month'] = ratings_df['datetime'].dt.month

# Merge with the movies dataset to get genres
merged_df = pd.merge(ratings_df, movies_df, on='movieId')

# Filter for a specific genre, e.g., Horror
horror_movies = merged_df[merged_df['genres'].str.contains('Comedy')]

# Group by month and count ratings
monthly_horror_ratings = horror_movies.groupby('month')['rating'].count()

# Plotting
plt.figure(figsize=(10, 6))
monthly_horror_ratings.plot(kind='bar')
plt.title('Monthly Ratings Count for Drama Movies')
plt.xlabel('Month')
plt.ylabel('Number of Ratings')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()