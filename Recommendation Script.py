# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sortedcontainers import SortedList

# Function to load data from a CSV file and perform data preprocessing
def load_data(file_path):
    # Read CSV file into a Pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Remove rows with non-numeric 'Rating'
    df = df[pd.to_numeric(df['Rating'], errors='coerce').notna()]
    
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Convert 'Rating' column to numeric
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    
    # Drop rows with missing 'User' or 'Movie'
    df = df.dropna(subset=['User', 'Movie'])
    
    return df

# Function to create mappings for users and movies
def create_mappings(df):
    # Map unique usernames to integer IDs
    users = {name: i for i, name in enumerate(df['User'].unique())}
    
    # Map unique movie titles to integer IDs
    movies = {title: i for i, title in enumerate(df['Movie'].unique())}
    
    # Add 'user_id' and 'movie_id' columns based on mappings
    df['user_id'] = df['User'].map(users)
    df['movie_id'] = df['Movie'].map(movies)
    
    return df, users, movies

# Function to update dictionaries storing user-movie ratings
def update_user_movie_rating_dicts(df, user2movie, movie2user, usermovie2rating):
    # Function to update user2movie, movie2user, and usermovie2rating
    def update_user2movie_and_movie2user(row):
        i = int(row.user_id) # Get user id
        j = int(row.movie_id) # Get movie id

        # Add movie to user's movie list
        user2movie.setdefault(i, []).append(j)

        # Add user to movie's user list
        movie2user.setdefault(j, []).append(i)

        # Store user-movie rating
        usermovie2rating[(i, j)] = row.Rating

    # Apply the update function to each row of the DataFrame
    df.apply(update_user2movie_and_movie2user, axis=1)
# Function to train the collaborative filtering model
def train_model(df, user2movie, movie2user, usermovie2rating, X, K, limit):
    neighbors = [] # Empty list to store SortedLists of neighbors
    averages = []   # Empty list to store average ratings
    deviations = [] # Empty list to store deviations from average ratings

    # Loop through all users
    for i in range(X):
        if i in user2movie: # If user i is found in the dictionary
            movies_i = set(user2movie[i]) # Get movies watched by user i

            # Get ratings for movies_i and calculate average and deviations
            ratings_i = {m: usermovie2rating[i, m] for m in movies_i}
            avg_i = np.mean(list(map(float, ratings_i.values())))
            dev_i = {m: float(r) - avg_i for m, r in ratings_i.items()}
            dev_i_values = np.array(list(dev_i.values()))
            sigma_i = np.sqrt(dev_i_values.dot(dev_i_values))

            averages.append(avg_i) # Add average rating to the list
            deviations.append(dev_i) # Add deviations to the list

            # Find neighbors for user i
            s1 = SortedList()
            for j in range(X):
                if j != i and j in movie2user: # If user j is found in the dictionary
                    movies_j = set(user2movie[j]) # Get movies watched by user j
                    common_movies = movies_i & movies_j # Find common movies between users i and j

                    if len(common_movies) >= limit: # If the number of common movies is greater than or equal to the limit
                        ratings_j = {m: usermovie2rating[j, m] for m in common_movies} # Get ratings for common movies
                        avg_j = np.mean(list(map(float, ratings_j.values()))) # Calculate average rating for user j
                        dev_j = {m: float(r) - avg_j for m, r in ratings_j.items()} # Calculate deviations from average for user j
                        dev_j_values = np.array(list(dev_j.values()))
                        sigma_j = np.sqrt(dev_j_values.dot(dev_j_values))

                        w_ij = sum(dev_i[m] * dev_j[m] for m in common_movies) / (sigma_i * sigma_j) # Calculate similarity between users i and j

                        s1.add((-w_ij, j)) # Add tuple (similarity, user id) to SortedList

                        if len(s1) > K: # If SortedList has more than K elements
                            del s1[-1] # Remove the last element
                    else:
                        print(f'Common movies less than limit for user {i} and user {j}. Skipping.')
                else:
                    print(f'User {j} not found in user2movie dictionary. Skipping.')
            neighbors.append(s1) # Add SortedList of neighbors to the list
        else:
            print(f'User {i} not found in user2movie dictionary. Skipping.')

    return neighbors, averages, deviations

# Function to predict user-movie ratings using the trained model
def predict(i, m, neighbors, deviations, averages):
    # Initialize numerator and denominator
    numerator = 0
    denominator = 0

    # Check if user has neighbors
    if i < len(neighbors):
        # Iterate through neighbors and calculate prediction
        for neg_w, j in neighbors[i]:
            try:
                numerator += -neg_w * deviations[j][m]
                denominator += abs(neg_w)
            except KeyError:
                pass

        # Handle case where denominator is zero
        if denominator == 0:
            prediction = np.mean(averages)
        else:
            prediction = numerator / denominator + averages[i]
        prediction = min(5, prediction)
        prediction = max(0.5, prediction)

        return prediction
    else:
        print(f'User {i} has no neighbors. Fallback to recommending popular movies.')
        return np.mean(averages)
# Function to get user input (user name)
def get_user_input():
    """Prompt for user name and return."""
    user_name = input("Enter your name: ")
    return user_name

# Function to recommend movies for a given user
def recommend_movies(user_name, df, user2movie, movie2user, usermovie2rating, neighbors, deviations, averages):
    # Check if user exists
    if user_name in users:
        user_id = users[user_name] # Get user id
        user_movies = set(user2movie.get(user_id, [])) # Get movies watched by user
        all_movies = set(movie2user.keys()) # Get all movies

        new_movies = all_movies - user_movies # Calculate new movies
        
        movie_id_to_name = {v: k for k, v in movies.items()}  # Map movie IDs to movie names

        predicted_ratings = [(movie_id_to_name[movie], predict(user_id, movie, neighbors, deviations, averages)) for movie in new_movies] # Predict ratings
        recommended_movies = sorted(predicted_ratings, key=lambda x: x[1], reverse=True) # Sort movies by ratings

        print(f"\nTop 5 movie recommendations for {user_name}:")
        for i, (movie, rating) in enumerate(recommended_movies[:5], start=1):
            print(f"{i}. {movie} - Predicted Rating: {rating:.2f}")
    else:
        print(f"User {user_name} not found. Please check the user name.")

# Main block to execute the recommendation system
if __name__ == "__main__":
    file_path = 'data.csv'
    df = load_data(file_path)
    df, users, movies = create_mappings(df)

    X = df.user_id.max() + 1
    Y = df.movie_id.max() + 1

    df = shuffle(df)
    cutoff = int(0.8 * len(df))
    df_train = df.iloc[:cutoff]
    df_test = df.iloc[cutoff:]

    user2movie = {}
    movie2user = {}
    usermovie2rating = {}
    usermovie2rating_test = {}

    update_user_movie_rating_dicts(df_train, user2movie, movie2user, usermovie2rating)

    X = max(list(user2movie.keys())) + 1
    Y = max(max(list(movie2user.keys()), default=0), max((m for u, m in usermovie2rating_test.keys()), default=0)) + 1

    K = 25
    limit = 5

    neighbors, averages, deviations = train_model(df_train, user2movie, movie2user, usermovie2rating, X, K, limit)

    user_name = get_user_input()
    recommend_movies(user_name, df, user2movie, movie2user, usermovie2rating, neighbors, deviations, averages)
