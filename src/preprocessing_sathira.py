import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)          # remove HTML tags like <br />
    text = text.lower()                          # lowercase
    text = re.sub(r'[^a-z\s]', ' ', text)         # remove punctuation/numbers
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

def load_and_clean(path='data/imdb.csv'):
    df = pd.read_csv(path)
    df['clean_review'] = df['review'].apply(clean_text)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    return df

if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())
    df.to_csv('data/imdb_cleaned.csv', index=False)