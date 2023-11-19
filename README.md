# Movie Recommendation System

This Python script implements a collaborative filtering-based movie recommendation system. It predicts movie ratings for a given user based on the ratings of similar users and provides personalized movie recommendations.

## Requirements

- `numpy`
- `pandas`
- `scikit-learn`
- `sortedcontainers`

## Usage

1. **Data Preparation:**
   - Prepare your movie rating data in CSV format with columns `User`, `Movie`, and `Rating`.
   - Ensure that the CSV file is accessible and update the `file_path` variable with the correct file path.

2. **Run the Script:**
   - Execute the script in a Python environment.

3. **Input User Name:**
   - When prompted, enter the user name for whom you want movie recommendations.

4. **View Recommendations:**
   - The script will provide the top 5 movie recommendations for the specified user.

## Code Overview

- **`load_data(file_path)`**: Function to load movie rating data from a CSV file and perform data preprocessing.
- **`create_mappings(df)`**: Function to create mappings for unique user names and movie titles to integer IDs.
- **`update_user_movie_rating_dicts(df, user2movie, movie2user, usermovie2rating)`**: Function to update dictionaries storing user-movie ratings.
- **`train_model(df, user2movie, movie2user, usermovie2rating, X, K, limit)`**: Function to train the collaborative filtering model, calculating user similarities and deviations.
- **`predict(i, m, neighbors, deviations, averages)`**: Function to predict user-movie ratings based on the trained model.
- **`get_user_input()`**: Function to get user input for the user name.
- **`recommend_movies(user_name, df, user2movie, movie2user, usermovie2rating, neighbors, deviations, averages)`**: Function to recommend movies for a given user.
- **Main Block**: Loads data, trains the model, takes user input, and provides movie recommendations.

## Notes

- Ensure that the required libraries (`numpy`, `pandas`, `scikit-learn`, `sortedcontainers`) are installed in your Python environment.
- The script uses collaborative filtering, which relies on finding similar users based on their movie preferences.
- Some error handling is implemented to skip certain cases during the model training process.

