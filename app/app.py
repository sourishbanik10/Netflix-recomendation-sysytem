import streamlit as st
import sys
import os
import requests
import re

# =====================================================
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# =====================================================
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# =====================================================
# TMDB API KEY
# =====================================================
TMDB_API_KEY = st.secrets["TMDB_API_KEY"]

# =====================================================
# IMPORT PROJECT MODULES
# =====================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, clean_data, merge_data, create_user_item_matrix
from src.content_based import build_content_model, recommend_content
from src.collaborative import train_svd, recommend_collaborative, evaluate_svd_proper
from src.hybrid import hybrid_recommend


# =====================================================
# NETFLIX STYLE
# =====================================================
st.markdown("""
<style>
body { background-color: #0e1117; }
.big-title {
    font-size:42px !important;
    font-weight:800;
    color: white;
}
.subtitle {
    font-size:18px;
    color: #aaaaaa;
}
.stButton>button {
    background-color: #e50914;
    color: white;
    font-weight: bold;
    border-radius: 8px;
    height: 3em;
}
.stButton>button:hover {
    background-color: #ff1f1f;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🎬 Hybrid Movie Recommendation System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Powered by Content-Based + Collaborative Filtering</p>', unsafe_allow_html=True)
st.divider()

# =====================================================
# DATA PATH
# =====================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_path = os.path.join(BASE_DIR, "data", "raw")


# =====================================================
# CACHED FUNCTIONS
# =====================================================
@st.cache_data
def load_and_prepare_data():
    ratings, movies = load_data(data_path)
    ratings, movies = clean_data(ratings, movies)
    merged_df = merge_data(ratings, movies)
    user_item_matrix = create_user_item_matrix(merged_df)
    return ratings, movies, user_item_matrix


@st.cache_resource
def build_models(movies, user_item_matrix):
    similarity_matrix = build_content_model(movies)
    reconstructed_df = train_svd(user_item_matrix)
    return similarity_matrix, reconstructed_df


@st.cache_data
def evaluate_model(ratings, movies):
    merged = ratings.merge(movies, on="movieId")
    return evaluate_svd_proper(merged)


# =====================================================
# LOAD EVERYTHING
# =====================================================
ratings, movies, user_item_matrix = load_and_prepare_data()
similarity_matrix, reconstructed_df = build_models(movies, user_item_matrix)
rmse, mae = evaluate_model(ratings, movies)


# =====================================================
# SIDEBAR METRICS
# =====================================================
st.sidebar.title("📊 Model Performance")

if rmse is not None:
    st.sidebar.metric("RMSE", f"{rmse:.4f}")
    st.sidebar.metric("MAE", f"{mae:.4f}")


# =====================================================
# HELPER FUNCTIONS
# =====================================================
def fetch_poster(movie_title):
    cleaned_title = re.sub(r"\(\d{4}\)", "", movie_title).strip()

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": TMDB_API_KEY, "query": cleaned_title}

    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("results"):
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except:
        return None

    return None


def get_predicted_rating(user_id, movie_title):
    if user_id in reconstructed_df.index and movie_title in reconstructed_df.columns:
        return round(reconstructed_df.loc[user_id, movie_title], 2)
    return None


def get_movie_genres(movie_title):
    row = movies[movies["title"] == movie_title]
    if not row.empty:
        return row.iloc[0]["genres"].replace("|", " • ")
    return "Unknown"


# =====================================================
# INPUT SECTION
# =====================================================
col1, col2 = st.columns(2)

with col1:
    user_id = st.selectbox(
        "👤 Select User ID",
        sorted(ratings["userId"].unique())
    )

with col2:
    movie_title = st.selectbox(
        "🎥 Select a Movie",
        movies["title"].values
    )

st.divider()


# =====================================================
# RECOMMENDATION BUTTON
# =====================================================
if st.button("✨ Generate Recommendations", use_container_width=True):

    with st.spinner("Generating intelligent recommendations..."):

        recommendations = hybrid_recommend(
            user_id=user_id,
            movie_title=movie_title,
            movies_df=movies,
            similarity_matrix=similarity_matrix,
            reconstructed_df=reconstructed_df,
            user_item_matrix=user_item_matrix,
            recommend_content_func=recommend_content,
            recommend_collaborative_func=recommend_collaborative
        )

    if not recommendations:
        st.warning("No recommendations found.")
    else:
        st.subheader("🍿 Recommended Movies For You")
        cols = st.columns(4)

        for index, movie in enumerate(recommendations):

            poster_url = fetch_poster(movie)
            predicted_rating = get_predicted_rating(user_id, movie)
            genres = get_movie_genres(movie)

            with cols[index % 4]:

                if poster_url:
                    st.image(poster_url, use_container_width=True)

                st.markdown(f"**{movie}**")
                st.caption(genres)

                if predicted_rating is not None:
                    st.write(f"⭐ Predicted Rating: {predicted_rating}/5")