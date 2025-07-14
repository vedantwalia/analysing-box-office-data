import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Loading the dataset and using caching to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")  # change to your actual file
    df['combined_features'] = (
        df['genres'].fillna('') + ' ' +
        df['overview'].fillna('') + ' ' +
        df['tagline'].fillna('')
    )
    return df

movies_df = load_data()

def recommend_by_genre(title, genre, top_n=5):
    df = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].copy()
    df = df.reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame(columns=['title', 'score'])

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    sim_scores = sim_scores[sim_indices]

    return df.iloc[sim_indices][['title']].assign(score=sim_scores)

#-------------------------Streamlit UI-------------------------

st.set_page_config(page_title="🎬 Genre Movie Recommender", layout="centered")

st.title("🎥 Genre-Based Movie Recommender")
st.markdown("Get content-based movie recommendations from a selected genre.")

# Genre selection
available_genres = sorted(set(g for sub in movies_df['genres'].dropna().str.split(',') for g in sub))
selected_genre = st.selectbox("Select a genre", available_genres)

# Movie selection
genre_movies = movies_df[movies_df['genres'].str.contains(selected_genre, case=False, na=False)]
movie_list = genre_movies['title'].dropna().unique()
selected_movie = st.selectbox("Choose a movie", sorted(movie_list))

# Number of recommendations
top_n = st.slider("Number of recommendations", 3, 15, 5)

# Recommend button
if st.button("Recommend"):
    recommendations = recommend_by_genre(selected_movie, selected_genre, top_n)
    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("Recommended Movies:")
        st.dataframe(recommendations.reset_index(drop=True))