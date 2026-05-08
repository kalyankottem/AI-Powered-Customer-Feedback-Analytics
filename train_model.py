# train_model.py

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# LOAD DATASET
df = pd.read_csv("final_feedback_dataset.csv")

X = df["feedback"]

y = df["label"]

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)

X_test_vec = vectorizer.transform(X_test)

# MODEL
model = LogisticRegression(max_iter=1000)

model.fit(X_train_vec, y_train)

# PREDICTIONS
y_pred = model.predict(X_test_vec)

# RESULTS
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))

# SAVE MODEL
with open("classifier.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel Saved Successfully!")