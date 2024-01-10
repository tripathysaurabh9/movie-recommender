from flask import Flask, request, jsonify
import pandas as pd
import joblib
from train.train_cf_with_eval import MovieRecommender

app = Flask(__name__)

# Load the trained recommender model
ratings_file = 'data/ratings.csv'
movies_file = 'data/movies.csv'
model_file = 'data/model.pkl'  # Replace with your path to the saved model

recommender = MovieRecommender(ratings_file, movies_file)
recommender.model = joblib.load(model_file)  # Load the pre-trained model

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = request.args.get('user_id', type=int)
    if user_id is None:
        return jsonify({'error': 'Missing user_id parameter'}), 400

    try:
        recommendations = recommender.recommend_movies_for_user(user_id)
        if recommendations is None:
            return jsonify({'error': 'No recommendations available for this user'}), 404
        return jsonify({'user_id': user_id, 'recommendations': recommendations.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
