import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

from feature_extraction import extract_features

# ----------------------------
# Config
# ----------------------------
DATASET_PATH = "G:/Baby Crying Sounds"
EMOTIONAL_CLASSES = ["tired", "lonely", "scared"]
RANDOM_STATE = 42

# ----------------------------
# Load data
# ----------------------------
X, y = [], []

for label in EMOTIONAL_CLASSES:
    class_path = os.path.join(DATASET_PATH, label)
    for file in os.listdir(class_path):
        if file.endswith(".wav"):
            features = extract_features(os.path.join(class_path, file))
            X.append(features)
            y.append(label)

X = np.array(X)
y = np.array(y)

# ----------------------------
# Encode labels
# ----------------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ----------------------------
# Train-test split (BEFORE SMOTE)
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

# ----------------------------
# Feature scaling
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# SMOTE (ONLY on training data)
# ----------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

# ----------------------------
# Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train_resampled, y_train_resampled)

# ----------------------------
# Evaluation
# ----------------------------
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ----------------------------
# Save artifacts
# ----------------------------
joblib.dump(model, "emotional_model.pkl")
joblib.dump(le, "emotional_label_encoder.pkl")
joblib.dump(scaler, "emotional_scaler.pkl")

print("✅ Emotional model trained correctly")
