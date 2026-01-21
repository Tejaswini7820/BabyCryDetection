import joblib
from feature_extraction import extract_features

model = joblib.load("baby_cry_xgb_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

features = extract_features(r"Baby Crying Sounds/hungry/0D1AD73E-4C5E-45F3-85C4-9A3CB71E8856-1430742197-1.0-m-04-hu.wav")
features = scaler.transform(features.reshape(1, -1))

# pred = model.predict(features)
# print("Prediction:", encoder.inverse_transform(pred)[0])

probs = model.predict_proba(features)[0]

# Top 3 predictions
top_indices = probs.argsort()[-3:][::-1]

print("Top predictions:")
for i in top_indices:
    print(f"{encoder.classes_[i]} : {probs[i]*100:.2f}%")

# Final decision with confidence
best_idx = top_indices[0]
best_label = encoder.classes_[best_idx]
confidence = probs[best_idx]

if confidence < 0.45:
    print("Final Decision: general discomfort")
else:
    print("Final Decision:", best_label)
