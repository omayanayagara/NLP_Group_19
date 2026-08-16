import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


def split_data(X, y, test_size=0.2):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test



def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)


    accuracy = accuracy_score(
        y_test,
        predictions
    )


    precision = precision_score(
        y_test,
        predictions
    )


    recall = recall_score(
        y_test,
        predictions
    )


    f1 = f1_score(
        y_test,
        predictions
    )


    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)


    print("\nClassification Report")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


    print("\nConfusion Matrix")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )



def save_model(model, path):

    with open(path, "wb") as file:
        pickle.dump(
            model,
            file
        )



def load_model(path):

    with open(path, "rb") as file:
        model = pickle.load(file)

    return model