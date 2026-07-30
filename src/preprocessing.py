import pandas as pd
import re
import string
from bs4 import BeautifulSoup

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK resources (only the first time)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

print("=" * 50)
print("IMDB DATASET PREPROCESSING")
print("=" * 50)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("data/imdb.csv")

print("\nDataset Loaded Successfully!")
print(df.head())

# -----------------------------
# Create Stopwords & Lemmatizer
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Preprocessing Function
# -----------------------------
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords & lemmatize
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            cleaned_words.append(lemma)

    return " ".join(cleaned_words)

# -----------------------------
# Apply preprocessing
# -----------------------------
print("\nCleaning Reviews...")

df["clean_review"] = df["review"].apply(preprocess_text)

print("\nCleaning Completed!")

# -----------------------------
# Save Cleaned Dataset
# -----------------------------
df.to_csv("data/cleaned_imdb.csv", index=False)

print("\nCleaned dataset saved as:")
print("data/cleaned_imdb.csv")

print("\nSample Cleaned Reviews:\n")

print(df[["review", "clean_review"]].head())
