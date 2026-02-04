import os
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

from feature_extraction import extract_features

DATASET_PATH = "G:/Baby Crying Sounds"
EMOTIONAL_CLASSES = ["tired", "lonely", "scared"]

X, y = [], []

for label in EMOTIONAL_CLASSES:
    path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(path):
        if file.endswith(".wav"):
            X.append(extract_features(os.path.join(path, file)))
            y.append(label)

X = np.array(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y_encoded)

joblib.dump(model, "emotional_model.pkl")
joblib.dump(le, "emotional_label_encoder.pkl")

print("✅ Emotional model trained")
