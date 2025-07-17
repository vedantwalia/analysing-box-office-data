import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import rapidfuzz 
import requests
from config import API_KEY, BASE_URL

# Loading the dataset and using caching to improve performance
@st.cache_data
def load_data():
    df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")
    df['combined_features'] = (
        df['genres'].fillna('') + ' ' +
        df['overview'].fillna('') + ' ' +
        df['tagline'].fillna('')
    )
    return df

movies_df = load_data()

@st.cache_resource
def compute_tfidf_matrix(df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    return tfidf, tfidf_matrix

def recommend_by_genre(title, genre, top_n=5):
    df = movies_df[movies_df['genres'].str.contains(genre, case=False, na=False)].copy()
    df = df.reset_index(drop=True)

    tfidf, tfidf_matrix = compute_tfidf_matrix(df)

    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    idx = indices.get(title)
    if idx is None:
        return pd.DataFrame(columns=['title', 'score'])

    sim_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-top_n-1:-1][::-1]
    sim_scores = sim_scores[sim_indices]

    return df.iloc[sim_indices][['title']].assign(score=sim_scores)

@st.cache_data
def cached_recommendations(title, genre, top_n):
    return recommend_by_genre(title, genre, top_n)

@st.cache_data
def get_movie_poster(title):
    #Fetches the movie poster URL using TMDB API
    search_url = "https://api.themoviedb.org/3/search/movie"
    params = {
        "api_key": API_KEY,
        "query": title
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if data.get("results"):
        result = data["results"][0]
        poster_path = result.get("poster_path")
        vote = result.get("vote_average", "N/A")
        overview = result.get("overview", "No overview available")

        poster_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
        return poster_url, vote, overview

    return None, "N/A", "No overview available"

@st.cache_data
def get_available_genres():
    return sorted(set(g for sub in movies_df['genres'].dropna().str.split(',') for g in sub))

#-------------------------Streamlit UI-------------------------

st.set_page_config(page_title="🎬 Genre Movie Recommender", layout="centered")

st.title("🎥 Genre-Based Movie Recommender")
st.markdown("Get content-based movie recommendations from a selected genre.")

# Genre selection
available_genres = get_available_genres()
selected_genre = st.selectbox("Select a genre", available_genres)

# Movie selection
genre_movies = movies_df[movies_df['genres'].str.contains(selected_genre, case=False, na=False)]
#movie_list = genre_movies['title'].dropna().unique()
#selected_movie = st.selectbox("Choose a movie", sorted(movie_list))

movie_list = genre_movies['title'].dropna().unique().tolist()
user_input = st.text_input("🎬 Type a movie name", "")

# Fuzzy matching to find the closest movie title

if user_input:
    matches =rapidfuzz.process.extract(user_input, movie_list, limit=10)

    if matches:
        selected_movie = matches[0][0]
        st.markdown(f"🧠 Best match: **{selected_movie}**  _(score: {matches[0][1]:.1f})_")

        st.markdown("💡 Other suggestions:")
        for title, score, _ in matches[1:6]:
            st.markdown(f"- {title} _(score: {score:.1f})_")
    else:
        st.warning("❌ No close matches found.")

# Number of recommendations
top_n = st.slider("Number of recommendations", 3, 15, 5)

# Recommend button
if st.button("Recommend") and selected_movie:
    recommendations = cached_recommendations(selected_movie, selected_genre, top_n)

    if recommendations.empty:
        st.warning("No recommendations found.")
    else:
        st.subheader("🎬 Recommended Movies")

        for _, row in recommendations.iterrows():
            title = row['title']
            poster_url, vote, overview = get_movie_poster(title)

            with st.container():
                cols = st.columns([1, 4])
                with cols[0]:
                    if poster_url:
                        st.image(poster_url, width=120)
                    else:
                        st.markdown("❌ No image")

                with cols[1]:
                    st.markdown(f"### {title}")
                    st.markdown(f"⭐ **Rating**: {vote}")
                    st.markdown(f"📖 *{overview[:250]}...*")  # Limit to 250 chars