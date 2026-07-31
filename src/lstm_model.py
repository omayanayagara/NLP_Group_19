from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)


def create_lstm_model(
        vocab_size,
        max_length
):

    model = Sequential()


    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=128,
            input_length=max_length
        )
    )


    model.add(
        LSTM(
            128,
            dropout=0.2,
            recurrent_dropout=0.2
        )
    )


    model.add(
        Dense(
            64,
            activation="relu"
        )
    )


    model.add(
        Dropout(0.5)
    )


    model.add(
        Dense(
            1,
            activation="sigmoid"
        )
    )


    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )


    return model