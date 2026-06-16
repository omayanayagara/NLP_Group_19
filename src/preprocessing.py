import re

# Attempt to import nltk stopwords; if unavailable, provide a small fallback
try:
    from nltk.corpus import stopwords
except Exception:
    class _FallbackStopwords:
        @staticmethod
        def words(lang='english'):
            return {
                'a','an','the','and','or','but','if','while','with','is','was',
                'for','on','in','to','of','by','as','at','from','that','this','it','be'
            }
    stopwords = _FallbackStopwords()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def preprocess(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    return text