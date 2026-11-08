# 🎬 Netflix-Style AI Movie & TV Discovery Engine

An immersive movie and TV show recommendation web application built with **Python**, **Streamlit**, and **Machine Learning**. It mimics the cinematic aesthetic of Netflix, complete with dynamic 3D SVG N-ribbon branding, intro splash animations, multithreaded poster fetching, and AI-powered match explanations.

## ✨ Key Features

- **AI-Powered Recommendation Engine:** Utilizes TF-IDF vectorization and cosine similarity on descriptions, genres, directors, and cast to find the closest thematic matches.
- **Cinematic Design System:** Custom dark-mode UI built in Streamlit with CSS gradients, hover-reveal cards, dynamic backdrop blurs, and a custom intro splash animation.
- **Multi-Source Poster Fetching:** Concurrent, thread-pooled poster resolution across IMDb and TVMaze APIs with smart fallbacks.
- **Transparent Match Insights:** Hover or open details to see why a title matched (shared genres, directors, cast, and plot themes).

## 🛠️ Tech Stack

- **Frontend & UI:** Streamlit, Custom CSS3, HTML5, 3D SVG Branding
- **Machine Learning & Data:** Scikit-Learn (TF-IDF, Cosine Similarity), Pandas
- **Concurrency & Networking:** Python `concurrent.futures` (ThreadPoolExecutor), `requests`

## 🚀 Getting Started Locally

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`