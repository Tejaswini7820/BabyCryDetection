import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

from feature_extraction import extract_features

DATASET_PATH = "Baby Crying Sounds"

X = []
y = []

# ============================
# Load Dataset
# ============================
for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        if file.endswith(".wav"):
            file_path = os.path.join(label_path, file)

            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(label)
            except Exception as e:
                print("Error:", file_path)

X = np.array(X)
y = np.array(y)

# # ============================
# # Encode Labels
# # ============================
# le = LabelEncoder()
# y_encoded = le.fit_transform(y)



# ============================
# GROUP CONFUSING CLASSES
# ============================
GROUP_MAP = {
    "hungry": "physical_need",
    "belly pain": "physical_need",
    "burping": "physical_need",
    "discomfort": "physical_need",
    "cold_hot": "physical_need",

    "tired": "emotional_need",
    "lonely": "emotional_need",
    "scared": "emotional_need",

    "laugh": "normal",
    "silence": "normal"
}

y_grouped = [GROUP_MAP[label] for label in y]

# Encode grouped labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_grouped)




# ============================
# Scale Features
# ============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Handle Imbalance (CRITICAL)
# ============================
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)

# ============================
# Train-Test Split
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled,
    y_resampled,
    test_size=0.2,
    random_state=42,
    stratify=y_resampled
)

# ============================
# RANDOM FOREST (IMPROVED)
# ============================
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# ============================
# XGBOOST CLASSIFIER
# ============================

from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(le.classes_),
    eval_metric="mlogloss",
    random_state=42
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

joblib.dump(xgb, "baby_cry_xgb_model.pkl")


# ============================
# Evaluation
# ============================
y_pred = rf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ============================
# Save Models
# ============================
joblib.dump(rf, "baby_cry_rf_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")


