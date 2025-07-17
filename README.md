# 🎬 Analysing Box Office Data

This project explores box office movie data to identify trends, patterns, and insights using Python and machine learning techniques. It aims to recommend movies based on content similarity and key metrics like popularity, genres, and ratings.

You can try out the project here:  
🔗 [Live Streamlit App](https://vedantwalia-analysing-box-office-data-app-qm6txi.streamlit.app)

---


## 📊 Features

- Search-based movie recommendations
- TMDB API integration for live metadata (posters, overviews, ratings)
- Cosine similarity-based content recommendations using TF-IDF
- Visualizations and insights from the movie dataset
- Interactive web interface built using Streamlit

---

## 🛠 Tech Stack

- **Python** 🐍
- **Pandas**, **NumPy** for data manipulation
- **Scikit-learn** for ML techniques
- **Streamlit** for UI
- **TMDB API** for movie metadata

---

## 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/vedantwalia/analysing-box-office-data
cd analysing-box-office-data
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🔐 TMDB API Key

Make sure you have a valid [TMDB API key](https://www.themoviedb.org/documentation/api) and store it securely (e.g., using a `.env` file).

---

## 🧠 How it Works

- The app uses TF-IDF vectorization to analyze movie overviews
- Computes cosine similarity between movie pairs
- When a user inputs a movie title, it returns similar titles with posters and trailers

---

## 🎥 Screenshots

_Add screenshots of the UI or visualizations here._

---

## 📁 Dataset

The movie data is sourced from:
- TMDB API
- Pre-cleaned CSV datasets (included in the repo or linked)

---

## ✍️ Author

**Vedant Walia**  
[GitHub](https://github.com/vedantwalia)

---

## 📄 License

This project is licensed under the MIT License.