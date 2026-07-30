from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle


def create_tfidf_features(
    dataframe,
    text_column="clean_review",
    max_features=5000
):
    """
    Convert cleaned text into TF-IDF features
    """

    vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    X = vectorizer.fit_transform(
        dataframe[text_column]
    )

    return X, vectorizer



def save_vectorizer(vectorizer, path):
    """
    Save TF-IDF vectorizer
    """

    with open(path, "wb") as file:
        pickle.dump(vectorizer, file)



def load_vectorizer(path):
    """
    Load saved TF-IDF vectorizer
    """

    with open(path, "rb") as file:
        vectorizer = pickle.load(file)

    return vectorizer