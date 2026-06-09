import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬"
)

st.title("🎬 Movie Recommendation System")

# Movie dataset
movies = [
    "Avengers is a superhero action movie",
    "Iron Man is a superhero movie",
    "Batman fights crime in Gotham",
    "Titanic is a romantic drama",
    "The Notebook is a love story",
    "Interstellar is a science fiction space movie",
    "The Martian is about survival in space"
]

# Load model once
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Generate movie embeddings once
@st.cache_data
def get_embeddings():
    return model.encode(movies)

embeddings = get_embeddings()

# User input
query = st.text_input(
    "Describe the movie you want",
    placeholder="e.g. superhero, romance, space adventure"
)

if st.button("Recommend"):

    if query.strip():

        # Query embedding
        query_embedding = model.encode([query])

        # Similarity scores
        scores = cosine_similarity(
            embeddings,
            query_embedding
        ).flatten()

        # DataFrame
        df = pd.DataFrame({
            "Movie": movies,
            "Similarity": scores
        })

        # Sort by similarity
        df = df.sort_values(
            by="Similarity",
            ascending=False
        ).reset_index(drop=True)

        # Top recommendations
        st.subheader("🏆 Top 3 Recommendations")

        for i in range(min(3, len(df))):
            st.write(
                f"**{i+1}. {df.loc[i, 'Movie']}** "
                f"(Score: {df.loc[i, 'Similarity']:.4f})"
            )

        # Full similarity table
        st.subheader("📊 Similarity Scores")
        st.dataframe(df, use_container_width=True)

    else:
        st.warning("Please enter a movie preference.")
