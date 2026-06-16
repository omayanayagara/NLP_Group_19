from gensim.models import Word2Vec

def train_word2vec(texts):
    sentences = [text.split() for text in texts]
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1)
    return model
