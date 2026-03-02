def hybrid_recommend(
    user_id,
    movie_title,
    movies_df,
    similarity_matrix,
    reconstructed_df,
    user_item_matrix,
    recommend_content_func,
    recommend_collaborative_func,
    top_n=8
):
    """
    Hybrid recommendation combining content-based and collaborative filtering
    """

    # Get content-based recommendations
    content_recs = recommend_content_func(
        movie_title,
        movies_df,
        similarity_matrix,
        top_n=top_n
    )

    # Get collaborative recommendations
    collab_recs = recommend_collaborative_func(
        user_id,
        reconstructed_df,
        user_item_matrix,
        top_n=top_n
    )

    # Combine and remove duplicates
    hybrid_recs = list(dict.fromkeys(content_recs + collab_recs))

    # Remove the original movie if present
    if movie_title in hybrid_recs:
        hybrid_recs.remove(movie_title)

    return hybrid_recs[:top_n]