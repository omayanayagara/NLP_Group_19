from fastapi import FastAPI
from pydantic import BaseModel

from predictor import predict_sentiment


app = FastAPI(
    title="Movie Review Sentiment Analysis API"
)



class Review(BaseModel):

    text: str

    model: str = "ml"



@app.get("/")
def home():

    return {
        "message":
        "Sentiment Analysis API Running"
    }



@app.post("/predict")
def prediction(review: Review):

    result = predict_sentiment(
        review.text,
        review.model
    )

    return {

        "review": review.text,

        "prediction": result

    }