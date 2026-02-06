import streamlit as st
from pydub import AudioSegment
import joblib
import os
import uuid

from feature_extraction import extract_features
from audio_preprocessing import preprocess_audio


# =========================================================
# Load models, encoders, scalers
# =========================================================

group_model = joblib.load("baby_cry_rf_model.pkl")
physical_model = joblib.load("physical_model.pkl")
emotional_model = joblib.load("emotional_model.pkl")

group_le = joblib.load("label_encoder.pkl")
physical_le = joblib.load("physical_label_encoder.pkl")
emotional_le = joblib.load("emotional_label_encoder.pkl")

group_scaler = joblib.load("scaler.pkl")
physical_scaler = joblib.load("physical_scaler.pkl")
emotional_scaler = joblib.load("emotional_scaler.pkl")


# =========================================================
# Page config
# =========================================================

st.set_page_config(
    page_title="Baby Cry Detection",
    page_icon="👶",
    layout="centered"
)


# =========================================================
# UI
# =========================================================

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #6ee7d8, #b3c7f7);
}

.card {
    background: white;
    padding: 40px;
    border-radius: 18px;
    width: 420px;
    margin: auto;
    margin-top: 20px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.15);
    text-align: center;
}

.title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #111827;
}

div.stButton > button {
    background-color: #4f6ef7;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px 28px;
    border: none;
}

div.stButton > button:hover {
    background-color: #3b5bdb;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style="text-align:center;color:">👶 Baby Cry Detection System</h1>
<p style="text-align:center;color:gray;">
Machine Learning based baby cry classification
</p>
<hr>
""", unsafe_allow_html=True)

st.markdown("### 🎵 Baby Cry Audio Input")

input_mode = st.radio(
    "Choose input method",
    ["Upload Audio", "Record Audio"],
    horizontal=True
)

uploaded_file = None
recorded_audio = None

if input_mode == "Upload Audio":
    uploaded_file = st.file_uploader(
        "Upload baby cry audio",
        type=["wav", "mp3", "m4a"]
    )
else:
    recorded_audio = st.audio_input("🎙️ Record baby cry")

predict = st.button("🔍 Predict", use_container_width=True)


# =========================================================
# Prediction logic
# =========================================================
import tempfile

if predict and (uploaded_file is not None or recorded_audio is not None):

    temp_raw = f"raw_{uuid.uuid4().hex}.wav"
    temp_clean = f"clean_{uuid.uuid4().hex}.wav"

    # -----------------------------
    # Handle uploaded audio
    # -----------------------------
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1].lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_input = tmp.name

        audio = AudioSegment.from_file(temp_input)
        audio = audio.set_channels(1).set_frame_rate(22050)
        audio.export(temp_raw, format="wav")
        os.remove(temp_input)

    # -----------------------------
    # Handle recorded audio
    # -----------------------------
    else:
        with open(temp_raw, "wb") as f:
            f.write(recorded_audio.getbuffer())

    # -----------------------------
    # Unified preprocessing
    # -----------------------------
    preprocess_audio(temp_raw, temp_clean)
    os.remove(temp_raw)

    # -----------------------------
    # Feature extraction
    # -----------------------------
    features = extract_features(temp_clean).reshape(1, -1)

    # -----------------------------
    # Stage 1: Group prediction
    # -----------------------------
    group_features = group_scaler.transform(features)
    group_pred = group_model.predict(group_features)
    group_result = group_le.inverse_transform(group_pred)[0]

    proba = group_model.predict_proba(group_features)
    confidence = max(proba[0])

    # -----------------------------
    # Decision with confidence gate
    # -----------------------------
    if confidence < 0.:
        final_output = "UNCERTAIN (Low confidence – recording conditions)"

    else:
        if group_result == "physical_need":
            phys_features = physical_scaler.transform(features)
            exact_pred = physical_model.predict(phys_features)
            exact_need = physical_le.inverse_transform(exact_pred)[0]

        elif group_result == "emotional_need":
            emo_features = emotional_scaler.transform(features)
            exact_pred = emotional_model.predict(emo_features)
            exact_need = emotional_le.inverse_transform(exact_pred)[0]

        else:
            exact_need = "Normal"

        final_output = f"{group_result.replace('_',' ').title()} ({exact_need.title()})"

    # -----------------------------
    # Show result
    # -----------------------------
    st.markdown(f"""
    <div style="
        background:white;
        padding:30px;
        border-radius:18px;
        margin-top:20px;
        text-align:center;
        box-shadow:0px 20px 40px rgba(0,0,0,0.15);
    ">
        <h2>Prediction Result</h2>
        <p style="font-size:22px;color:#2563eb;font-weight:700;">
            {final_output.upper()}
        </p>
        <p style="color:gray;">
            Confidence: {confidence:.2f}
        </p>
    </div>
    """, unsafe_allow_html=True)

    os.remove(temp_clean)

elif predict:
    st.warning("Please upload or record audio before predicting.")


# =========================================================
# Info
# =========================================================

with st.expander("ℹ️ About this Project"):
    st.write("""
    - Machine Learning based baby cry classification
    - Unified preprocessing for uploaded and recorded audio
    - MFCC + spectral audio features
    - Two-stage prediction (Group → Exact Need)
    - Confidence-aware output
    """)

st.markdown("""
<hr>
<p style="text-align:center;color:gray;font-size:14px;">
B.Tech Project • Baby Cry Detection
</p>
""", unsafe_allow_html=True)
