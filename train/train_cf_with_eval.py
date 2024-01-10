import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix

class MovieRecommender:
    def __init__(self, ratings_file, movies_file):
        self.ratings_data = pd.read_csv(ratings_file)
        self.movies_data = pd.read_csv(movies_file, index_col='movieId')
        self.train_data, self.test_data = train_test_split(self.ratings_data, test_size=0.20, random_state=42)
        self.train_matrix = self.create_user_movie_matrix(self.train_data)
        self.train_matrix_sparse = csr_matrix(self.train_matrix.values)
        self.model = None

    def save_model(self, file_path):
        """ Save the trained model to a file. """
        joblib.dump(self.model, file_path)
        print(f"Model saved to {file_path}")

    def create_user_movie_matrix(self, data):
        return data.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)

    def train_best_model(self, param_grid):
        best_rmse = float('inf')
        best_n_components = None

        for n_components in param_grid['n_components']:
            model = TruncatedSVD(n_components=n_components, random_state=42)
            model.fit(self.train_matrix_sparse)
            rmse = self.rmse_score(model)
            if rmse < best_rmse:
                best_rmse = rmse
                best_n_components = n_components

        self.model = TruncatedSVD(n_components=best_n_components, random_state=42)
        self.model.fit(self.train_matrix_sparse)
        return best_rmse, best_n_components

    def rmse_score(self, model):
        predictions = model.transform(self.train_matrix_sparse)
        predictions = np.dot(predictions, model.components_)
        mask = self.train_matrix.values != 0
        mse = mean_squared_error(self.train_matrix.values[mask], predictions[mask])
        return np.sqrt(mse)

    def predict_ratings_for_user(self, user_id):
        if user_id not in self.train_matrix.index:
            return None
        user_data = self.train_matrix.loc[user_id, :].values.reshape(1, -1)
        user_latent_matrix = self.model.transform(user_data)
        estimated_ratings = np.dot(user_latent_matrix, self.model.components_).flatten()
        return estimated_ratings

    def recommend_movies_for_user(self, user_id, n_recommendations=5):
        estimated_ratings = self.predict_ratings_for_user(user_id)
        if estimated_ratings is None:
            return None
        all_movies = pd.DataFrame(estimated_ratings, index=self.train_matrix.columns, columns=['estimated_rating'])
        already_rated = self.train_matrix.loc[user_id, :]
        recommendations = all_movies[~all_movies.index.isin(already_rated[already_rated > 0].index)]
        top_recommendations = recommendations.sort_values(by='estimated_rating', ascending=False).head(n_recommendations)
        top_recommendations = top_recommendations.merge(self.movies_data, left_index=True, right_index=True)
        return top_recommendations['title']

    def evaluate_model(self):
        rmse_accum = 0
        num_users = 0
        for user_id in self.test_data['userId'].unique():
            if user_id not in self.train_matrix.index:
                continue
            test_movies = self.test_data[self.test_data['userId'] == user_id]
            test_movies = test_movies[test_movies['movieId'].isin(self.train_matrix.columns)]
            if test_movies.empty:
                continue
            estimated_ratings = self.predict_ratings_for_user(user_id)
            if estimated_ratings is None:
                continue
            actual_ratings = test_movies.set_index('movieId')['rating']
            predicted_ratings = pd.Series(estimated_ratings, index=self.train_matrix.columns).loc[actual_ratings.index]
            rmse_accum += mean_squared_error(actual_ratings, predicted_ratings, squared=False)
            num_users += 1
        return rmse_accum / num_users if num_users > 0 else float('nan')

    def generate_evaluation_report(self):
        num_test_items = len(self.test_data)
        num_test_users = self.test_data['userId'].nunique()
        rmse_accum = 0
        num_users_evaluated = 0
        no_rating_count = 0

        for user_id in self.test_data['userId'].unique():
            if user_id not in self.train_matrix.index:
                continue  # Skip users not in the training set

            # Get the test movies for this user that are also in the training set
            test_movies = self.test_data[self.test_data['userId'] == user_id]
            test_movies = test_movies[test_movies['movieId'].isin(self.train_matrix.columns)]

            if test_movies.empty:
                continue  # Skip if no common movies

            # Predict ratings for these movies
            estimated_ratings = self.predict_ratings_for_user(user_id)
            if estimated_ratings is None:
                no_rating_count += 1
                continue  # Skip if user not in model

            # Calculate RMSE for this user
            actual_ratings = test_movies.set_index('movieId')['rating']
            predicted_ratings = pd.Series(estimated_ratings, index=self.train_matrix.columns).loc[actual_ratings.index]
            rmse_accum += mean_squared_error(actual_ratings, predicted_ratings, squared=False)
            num_users_evaluated += 1

        average_rmse = rmse_accum / num_users_evaluated if num_users_evaluated > 0 else float('nan')

        print("Evaluation Report:")
        print(f"Number of Items in Test Set: {num_test_items}")
        print(f"Number of Users in Test Set: {num_test_users}")
        print(f"Number of Users Evaluated: {num_users_evaluated}")
        print(f"Number of Users with No Ratings Predicted: {no_rating_count}")
        print(f"Average RMSE on Test Set: {average_rmse}")

# Example usage
ratings_file = '/Users/stripathy/Everything/take-home/take-home/data/ratings.csv'
movies_file = '/Users/stripathy/Everything/take-home/take-home/data/movies.csv'
recommender = MovieRecommender(ratings_file, movies_file)

# Hyperparameter tuning
param_grid = {'n_components': [20, 50, 100, 200]}  # Adjust based on your dataset
best_rmse, best_n_components = recommender.train_best_model(param_grid)
print(f"Best RMSE: {best_rmse} for n_components: {best_n_components}")

# Save the trained model
model_file_path = '/Users/stripathy/Everything/take-home/take-home/data/model.pkl'  # Replace with your desired path
recommender.save_model(model_file_path)

# Get recommendations
top_movies = recommender.recommend_movies_for_user(1)  # Replace with the user ID you want recommendations for
print(top_movies)

recommender.generate_evaluation_report()

