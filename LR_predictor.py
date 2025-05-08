import os
import py7zr
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, ConfusionMatrixDisplay, classification_report, accuracy_score

data_dir = "data/without comments/"
data_files = [f for f in os.listdir(data_dir) if f.endswith(".jsonl")] 

CHUNKSIZE = 10000
MAX_FEATURES = 2000
CLASSES = np.array([0, 1])

sample_texts = []
for file in data_files:
    path = os.path.join(data_dir, file)

    for chunk in pd.read_json(path, lines=True, chunksize=CHUNKSIZE):
        chunk["text"] = chunk["title"].fillna("") + " " + chunk["body"].fillna("")
        sample_texts.extend(chunk["text"].tolist())

        if len(sample_texts) >= 20000:
            break

    if len(sample_texts) >= 20000:
        break

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, stop_words="english")
vectorizer.fit(sample_texts)

accuracies = []
precisions = []
recalls = []
f1s = []
all_coefficients = []

for run in range(5):
    print(f"\n--- Run {run + 1} ---")
    scaler = StandardScaler()
    classifier = SGDClassifier(loss="log_loss", max_iter=5)
    X_test_all, y_test_all = [], []

    for file in data_files:
        path = os.path.join(data_dir, file)
        print(f"Processing file: {path}")

        for chunk in pd.read_json(path, lines=True, chunksize=CHUNKSIZE):
            chunk.dropna(subset=["score", "num_comments", "title", "body"], inplace=True)

            threshold = chunk["score"].quantile(0.75)
            chunk["label"] = (chunk["score"] >= threshold).astype(int)
            chunk["text"] = chunk["title"].fillna("") + " " + chunk["body"].fillna("")

            X_text = vectorizer.transform(chunk["text"])
            X_numeric = scaler.fit_transform(chunk[["num_comments"]])
            X = np.hstack([X_text.toarray(), X_numeric])
            y = chunk["label"].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            classifier.partial_fit(X_train, y_train, classes=CLASSES)

            X_test_all.append(X_test)
            y_test_all.append(y_test)

X_test_all = np.vstack(X_test_all)
y_test_all = np.concatenate(y_test_all)

y_pred = classifier.predict(X_test_all)
acc = accuracy_score(y_test_all, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test_all, y_pred)

accuracies.append(acc)
precisions.append(precision)
recalls.append(recall)
f1s.append(f1)

all_coefficients.append(classifier.coef_[0][:-1])

print("\n=== Average Metrics Over 5 Runs ===")
print(f"Accuracy: {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall: {np.mean(recalls):.4f}")
print(f"F1-score: {np.mean(f1s):.4f}")

avg_coefficients = np.mean(all_coefficients, axis=0)
feature_names = vectorizer.get_feature_names_out()

top_popular = np.argsort(avg_coefficients)[-5:][::-1]
top_not_popular = []
sorted_indices = np.argsort(avg_coefficients)

for idx in sorted_indices:
    if feature_names[idx] != "removed":
        top_not_popular.append(idx)
    if len(top_not_popular) == 5:
        break

print("\nTop 5 features for 'Popular' posts:")
for idx in top_popular:
    print(f"{feature_names[idx]:<20} {avg_coefficients[idx]:.4f}")

print("\nTop 5 features for 'Not Popular' posts:")
for idx in top_not_popular:
    print(f"{feature_names[idx]:<20} {avg_coefficients[idx]:.4f}")

disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test_all, y_pred), display_labels=["Not Popular", "Popular"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
print(classification_report(y_test_all, y_pred, target_names=["Not Popular", "Popular"]))

joblib.dump((vectorizer, scaler, classifier), "reddit_incremental_model.pkl")