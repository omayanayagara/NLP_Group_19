from sklearn.linear_model import LogisticRegression


def create_logistic_regression_model():
    """
    Create and return a Logistic Regression model.
    """

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return model