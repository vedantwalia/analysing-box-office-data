import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

st.set_page_config(page_title="TMDb Movie Recommender", layout="wide")
st.title("🎬 TMDb Movie Recommender & Explorer")

st.write("📥 Loading data...")
@st.cache_data
def load_data():
    df = pd.read_csv("data/TMDB_movie_dataset_v11.csv")
    return df

df = load_data()
st.success("✅ Data loaded successfully!")

st.write("🧹 Preprocessing...")
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['combined_features'] = df['overview'].astype(str) + " " + df['genres'].astype(str)
df['combined_features'] = df['combined_features'].fillna('').astype(str)
st.success("✅ Preprocessing complete.")

st.write("🔎 Vectorizing...")
tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
st.success("✅ Vectorization complete.")

# Create reverse index
indices = pd.Series(df.index, index=df['title']).drop_duplicates()

def recommend(title, num_recommendations=5):
    idx = indices.get(title)
    if idx is None:
        return []
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    sim_indices = sim_scores.argsort()[-(num_recommendations+1):-1][::-1]
    return df[['title', 'vote_average', 'release_date']].iloc[sim_indices]

st.subheader("🔍 Movie-Based Recommendations")

selected_movie = st.selectbox("🎥 Choose a movie:", sorted(df['title'].dropna().unique()))

if selected_movie:
    st.write(f"🎯 Recommendations based on **{selected_movie}**:")
    recs = recommend(selected_movie)
    if len(recs) > 0:
        st.table(recs)
    else:
        st.warning("No recommendations found.")