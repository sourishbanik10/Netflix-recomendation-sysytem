import pandas as pd
import os


def load_data(data_path):
    """
    Load movies and ratings datasets
    """
    ratings = pd.read_csv(os.path.join(data_path, "ratings.csv"))
    movies = pd.read_csv(os.path.join(data_path, "movies.csv"))
    
    return ratings, movies


def clean_data(ratings, movies):
    """
    Basic cleaning
    """
    # Remove duplicates
    ratings = ratings.drop_duplicates()
    movies = movies.drop_duplicates()

    # Remove null values
    ratings = ratings.dropna()
    movies = movies.dropna()

    return ratings, movies


def merge_data(ratings, movies):
    """
    Merge ratings with movies on movieId
    """
    merged_df = pd.merge(ratings, movies, on="movieId")
    return merged_df


def create_user_item_matrix(merged_df):
    """
    Create user-item interaction matrix
    """
    user_item_matrix = merged_df.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    )
    
    return user_item_matrix