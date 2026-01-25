import streamlit as st
import joblib
import os
from feature_extraction import extract_features

# ----------------------------
# Load model components
# ----------------------------
model = joblib.load("baby_cry_rf_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Baby Cry Detection",
    page_icon="👶",
    layout="centered"
)

# ----------------------------
# Custom CSS (MATCHING YOUR IMAGE)
# ----------------------------
st.markdown("""
<style>
/* Background */
.stApp {
    background: linear-gradient(135deg, #6ee7d8, #b3c7f7);
}

/* Center card */
.card {
    background: white;
    padding: 40px;
    border-radius: 18px;
    width: 420px;
    margin: auto;
    margin-top: 120px;
    box-shadow: 0px 20px 40px rgba(0,0,0,0.15);
    text-align: center;
}

/* Title */
.title {
    font-size: 26px;
    font-weight: 700;
    margin-bottom: 8px;
    color: #111827;
}

/* Subtitle */
.subtitle {
    font-size: 15px;
    color: #6b7280;
    margin-bottom: 25px;
}

/* Predict button */
div.stButton > button {
    background-color: #4f6ef7;
    color: white;
    font-size: 18px;
    font-weight: 600;
    border-radius: 12px;
    padding: 12px 28px;
    width: auto;              /* 🔥 CRITICAL */
    border: none;
}

div.stButton > button:hover {
    background-color: #3b5bdb;
}
            
.button-wrapper {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
}

/* Hide Streamlit default stuff */
#MainMenu, footer, header {
    visibility: hidden;
}
            
            
</style>
""", unsafe_allow_html=True)

# ----------------------------
# UI Card
# ----------------------------
st.markdown("""
    <h1 style='text-align: center;'>👶 Baby Cry Detection System</h1>
    <p style='text-align: center; color: gray;'>
    Machine Learning based baby cry classification
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# Use container to align Streamlit widgets inside card
with st.container():
    # st.markdown("<div style='max-width:420px;margin:auto;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        st.markdown("#### 🎵 Upload Baby Cry Audio (WAV)")

    uploaded_file = st.file_uploader("", type=["wav"],help="Upload a baby cry audio file in WAV format")

    st.markdown("</div>", unsafe_allow_html=True)



predict = False  # important


col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    predict = st.button("🔍 Predict", use_container_width=True)
    

# ----------------------------
# Prediction logic
# ----------------------------
if predict and uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features("temp.wav")
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    prediction = model.predict(features)
    result = label_encoder.inverse_transform(prediction)[0]

    st.markdown(f"""
    <div class="card" style="margin-top:20px;">
        <div class="title">Prediction Result</div>
        <div style="font-size:22px;color:#2563eb;font-weight:700;margin-top:10px;">
            {result.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    os.remove("temp.wav")

# ----------------------------
# Info Section
# ----------------------------

with st.expander("ℹ️ About this Project"):
    st.write(
        """
        - This system uses **Machine Learning** to classify baby cries.
        - Audio features (MFCCs) are extracted using **Librosa**.
        - A trained ML model predicts the cry type.
        - Useful for parents and healthcare support.
        """
    )


# ----------------------------
# Footer
# ----------------------------
st.markdown(
    """
    <hr>
    <p style='text-align: center; color: gray; font-size: 14px;'>
    B.Tech Mini / Major Project • Baby Cry Detection
    </p>
    """,
    unsafe_allow_html=True
)