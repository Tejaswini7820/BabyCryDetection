import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

from feature_extraction import extract_features

DATASET_PATH = "G:/Baby Crying Sounds"
PHYSICAL_CLASSES = ["hungry", "belly pain", "burping", "discomfort"]

X, y = [], []

for label in PHYSICAL_CLASSES:
    path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(path):
        if file.endswith(".wav"):
            X.append(extract_features(os.path.join(path, file)))
            y.append(label)

X = np.array(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

joblib.dump(model, "physical_model.pkl")
joblib.dump(le, "physical_label_encoder.pkl")
joblib.dump(scaler, "physical_scaler.pkl")

print("✅ Physical model trained")
