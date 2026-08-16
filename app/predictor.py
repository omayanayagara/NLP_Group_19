import pickle
import numpy as np
import tensorflow as tf

import sys
sys.path.append("../")

from src.preprocessing import preprocess


# Load Logistic Regression model

with open(
    "../models/logistic_regression.pkl",
    "rb"
) as file:
    logistic_model = pickle.load(file)



# Load TF-IDF vectorizer

with open(
    "../models/tfidf_vectorizer.pkl",
    "rb"
) as file:
    tfidf_vectorizer = pickle.load(file)



# Load LSTM model

lstm_model = tf.keras.models.load_model(
    "../models/lstm_model.keras"
)



def predict_sentiment(review, model_type="ml"):

    cleaned_review = preprocess(review)


    if model_type == "ml":

        vector = tfidf_vectorizer.transform(
            [cleaned_review]
        )

        prediction = logistic_model.predict(vector)


        if prediction[0] == 1:
            return "Positive"
        else:
            return "Negative"



    elif model_type == "dl":

        prediction = lstm_model.predict(
            np.array([cleaned_review])
        )


        if prediction[0][0] > 0.5:
            return "Positive"
        else:
            return "Negative"