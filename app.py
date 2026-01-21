from flask import Flask, request, jsonify, render_template
import joblib
import os

from feature_extraction import extract_features

app = Flask(__name__)

# Load trained objects
model = joblib.load("baby_cry_xgb_model.pkl")
encoder = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    # Browser GET
    if request.method == "GET":
        return render_template("index.html")

    # POST (from form / Postman)
    if "audio" not in request.files:
        return render_template("index.html", error="Audio file missing")

    file = request.files["audio"]

    if file.filename == "":
        return render_template("index.html", error="No file selected")

    temp_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(temp_path)

    features = extract_features(temp_path)
    features = scaler.transform(features.reshape(1, -1))

    probs = model.predict_proba(features)[0]
    best_idx = probs.argmax()

    label = encoder.classes_[best_idx]
    confidence = float(probs[best_idx])

    os.remove(temp_path)

    if confidence < 0.45:
        final_prediction = "General Discomfort"
    elif label == "physical_need":
        final_prediction = "Baby needs physical care (feeding / comfort / pain check)"
    elif label == "emotional_need":
        final_prediction = "Baby needs emotional care (sleep / attention)"
    else:
        final_prediction = "Baby is calm or playful"

    return render_template(
        "index.html",
        prediction=final_prediction,
        confidence=round(confidence * 100, 2)
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
