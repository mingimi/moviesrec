import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie data
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Generate embeddings (cached)
@st.cache_data
def get_embeddings():
    return model.encode(movies)

embeddings = get_embeddings()

# User input
query = st.text_input(
    "Describe the type of movie you want:",
    placeholder="e.g. superhero, space adventure, romance"
)

if st.button("Recommend Movies"):

    if query.strip():

        query_embedding = model.encode([query])

        scores = cosine_similarity(
            embeddings,
            query_embedding
        ).flatten()

        top_indices = np.argsort(scores)[::-1][:3]

        st.subheader("Top Recommendations")

        for i, idx in enumerate(top_indices, start=1):
            st.write(
                f"**{i}.** {movies[idx]} "
                f"(Similarity: {scores[idx]:.3f})"
            )

    else:
        st.warning("Please enter a movie preference.")