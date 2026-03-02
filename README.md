🎬 Hybrid Movie Recommendation System

🚀 Live Demo

https://netflix-recomendation-sysytem-hitnniveesc4unawsrqljn.streamlit.app/


A full-stack Machine Learning project that builds a Hybrid Recommendation Engine combining:

Content-Based Filtering

Collaborative Filtering (SVD)

TMDB API Integration (Movie Posters)

Streamlit Web Application

Cloud Deployment


📌 Project Overview

This project replicates how streaming platforms like Netflix recommend movies by combining:

🎯 User preference patterns (Collaborative Filtering)

🎥 Movie similarity based on metadata (Content-Based Filtering)

🤝 Hybrid model for improved accuracy

The system provides personalized recommendations with:

Movie Posters

Predicted Ratings

Genre Display

Interactive UI

🧠 Recommendation Approaches Used
1️⃣ Content-Based Filtering

TF-IDF Vectorization

Cosine Similarity

Movie metadata-based similarity

2️⃣ Collaborative Filtering

Matrix Factorization using SVD

User-item interaction matrix

Predicted ratings reconstruction

3️⃣ Hybrid Model

Combines:

Content similarity results

SVD predicted recommendations
to produce better personalized suggestions.

🏗 Project Structure
movie-recommender-system/
│
├── app/
│   └── app.py
│
├── src/
│   ├── data_preprocessing.py
│   ├── content_based.py
│   ├── collaborative.py
│   └── hybrid.py
│
├── data/
│   └── raw/
│
├── requirements.txt
└── README.md
⚙️ Tech Stack

Python

Pandas

NumPy

Scikit-learn

Scikit-Surprise

Streamlit

TMDB API

GitHub

Streamlit Cloud

📊 Features

✅ Hybrid Recommendation Engine
✅ SVD Matrix Factorization
✅ Predicted Rating Display
✅ Genre Tags
✅ Netflix-style UI
✅ Poster Integration via API
✅ Cloud Deployment

🔐 API Usage

Movie posters are fetched using the TMDB API.

API keys are securely managed using Streamlit secrets.

📈 Future Improvements

Add search instead of dropdown

Add trending section

Add user login simulation

Improve ranking strategy


Add evaluation metrics (RMSE, Precision@K)
