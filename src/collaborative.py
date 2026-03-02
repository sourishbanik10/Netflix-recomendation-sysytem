import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# =====================================================
# TRAIN SVD USING SKLEARN (VERY STABLE)
# =====================================================
def train_svd(user_item_matrix, k=50):
    """
    Train collaborative filtering model using TruncatedSVD.
    Extremely stable compared to numpy/scipy SVD.
    """

    matrix = user_item_matrix.fillna(0).values.astype(float)

    # Safety: limit k
    k = min(k, min(matrix.shape) - 1)

    # Mean center
    user_means = np.mean(matrix, axis=1)
    matrix_centered = matrix - user_means.reshape(-1, 1)

    # Truncated SVD
    svd = TruncatedSVD(n_components=k, random_state=42)
    latent_matrix = svd.fit_transform(matrix_centered)

    # Reconstruct
    reconstructed = np.dot(latent_matrix, svd.components_)

    # Add user means back
    reconstructed = reconstructed + user_means.reshape(-1, 1)

    # Clip ratings
    reconstructed = np.clip(reconstructed, 0, 5)

    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    return reconstructed_df


# =====================================================
# PROPER EVALUATION
# =====================================================
def evaluate_svd_proper(merged_df):

    train_df, test_df = train_test_split(
        merged_df,
        test_size=0.2,
        random_state=42
    )

    user_item_matrix = train_df.pivot_table(
        index="userId",
        columns="title",
        values="rating"
    ).fillna(0)

    reconstructed_df = train_svd(user_item_matrix)

    y_true = []
    y_pred = []

    for _, row in test_df.iterrows():
        user = row["userId"]
        movie = row["title"]

        if user in reconstructed_df.index and movie in reconstructed_df.columns:
            y_true.append(row["rating"])
            y_pred.append(reconstructed_df.loc[user, movie])

    if len(y_true) == 0:
        return None, None

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print(f"Proper RMSE: {rmse:.4f}")
    print(f"Proper MAE: {mae:.4f}")

    return rmse, mae


# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================
def recommend_collaborative(user_id, reconstructed_df, user_item_matrix, top_n=5):

    if user_id not in reconstructed_df.index:
        return []

    rated_movies = user_item_matrix.loc[user_id]
    rated_movies = rated_movies[rated_movies > 0].index.tolist()

    user_predictions = reconstructed_df.loc[user_id]
    user_predictions = user_predictions.drop(rated_movies, errors="ignore")

    recommendations = (
        user_predictions
        .sort_values(ascending=False)
        .head(top_n)
    )

    return recommendations.index.tolist()