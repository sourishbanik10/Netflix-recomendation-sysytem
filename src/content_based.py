import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_content_model(movies):
    """
    Build TF-IDF matrix and similarity matrix
    """

    # Replace | with space
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

    # Convert genres into TF-IDF features
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(movies["genres"])

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

    return similarity_matrix


def recommend_content(movie_title, movies, similarity_matrix, top_n=5):
    """
    Recommend similar movies based on content
    """

    # Create index mapping
    indices = pd.Series(movies.index, index=movies["title"]).drop_duplicates()

    if movie_title not in indices:
        return ["Movie not found"]

    idx = indices[movie_title]

    similarity_scores = list(enumerate(similarity_matrix[idx]))

    # Sort by similarity
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top N similar movies (excluding itself)
    similarity_scores = similarity_scores[1:top_n+1]

    movie_indices = [i[0] for i in similarity_scores]

    return movies["title"].iloc[movie_indices].tolist()