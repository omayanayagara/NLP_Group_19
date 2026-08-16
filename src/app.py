import streamlit as st
import pickle
import re
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not TFIDF_PATH.exists():
    st.error(
        f"TF-IDF vectorizer was not found.\n\n"
        f"Expected location:\n{TFIDF_PATH}"
    )
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Logistic Regression model was not found.\n\n"
        f"Expected location:\n{MODEL_PATH}"
    )
    st.stop()


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

with open(TFIDF_PATH, "rb") as file:
    vectorizer = pickle.load(file)


# ============================================================
# LOAD LOGISTIC REGRESSION MODEL
# ============================================================

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z]", " ", text)

    text = " ".join(text.split())

    return text


# ============================================================
# APPLICATION INTERFACE
# ============================================================

st.title("🎬 Movie Review Sentiment Analyzer")

st.write(
    "Enter a movie review below and the machine learning model "
    "will predict whether the sentiment is positive or negative."
)


review = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Example: This movie was absolutely fantastic!"
)


# ============================================================
# PREDICTION
# ============================================================

if st.button("Analyze Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        # Clean the review
        cleaned_review = clean_text(review)

        # Convert text into TF-IDF features
        review_vector = vectorizer.transform([cleaned_review])

        # Make prediction
        prediction = model.predict(review_vector)[0]

        # Get prediction probabilities
        probability = model.predict_proba(review_vector)[0]

        negative_probability = probability[0]
        positive_probability = probability[1]


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        if prediction == 1:

            st.success("😊 Positive Sentiment")

            st.write(
                f"Positive probability: "
                f"{positive_probability * 100:.2f}%"
            )

        else:

            st.error("😞 Negative Sentiment")

            st.write(
                f"Negative probability: "
                f"{negative_probability * 100:.2f}%"
            )