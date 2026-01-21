import joblib
import numpy as np
from feature_extraction import extract_features

model = joblib.load("baby_cry_rf_model.pkl")  # use RF first
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

file_path = "Baby Crying Sounds/hungry/0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.wav"  # use a training file

features = extract_features(file_path)
features = scaler.transform(features.reshape(1, -1))

pred = model.predict(features)
print("Predicted:", encoder.inverse_transform(pred)[0])
print("Expected: hungry")
