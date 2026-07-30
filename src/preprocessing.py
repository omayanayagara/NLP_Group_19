import re
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text):
    words = text.split()

    filtered_words = [
        word for word in words 
        if word not in stop_words
    ]

    return " ".join(filtered_words)


def preprocess(text):

    text = clean_text(text)

    text = remove_stopwords(text)

    return text