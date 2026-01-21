import os
import joblib
from feature_extraction import extract_features

model = joblib.load("baby_cry_xgb_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

folder = r"Baby Crying Sounds/hungry"

results = {}

for file in os.listdir(folder):
    if file.endswith(".wav"):
        features = extract_features(os.path.join(folder, file))
        features = scaler.transform(features.reshape(1, -1))
        pred = model.predict(features)
        label = encoder.inverse_transform(pred)[0]

        results[label] = results.get(label, 0) + 1

print(results)
