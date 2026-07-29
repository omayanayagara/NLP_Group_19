import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import time

df = pd.read_csv('data/imdb_cleaned.csv')

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_review'], df['label'], test_size=0.2, random_state=42)

start = time.time()
bow = CountVectorizer(max_features=10000)
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)
print("Feature extraction time:", time.time() - start, "seconds")
print("Feature vector size:", X_train_bow.shape[1])

model = LinearSVC()
model.fit(X_train_bow, y_train)
y_pred = model.predict(X_test_bow)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/svm_bow_model.pkl')
joblib.dump(bow, 'models/bow_vectorizer.pkl')