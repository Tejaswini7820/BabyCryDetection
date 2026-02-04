import streamlit as st
from pydub import AudioSegment
import joblib
import os
from feature_extraction import extract_features



# ----------------------------
# Load model components
# ----------------------------
group_model = joblib.load("baby_cry_rf_model.pkl")
group_model_1 = joblib.load("baby_cry_xgb_model.pkl")
physical_model = joblib.load("physical_model.pkl")
emotional_model = joblib.load("emotional_model.pkl")

group_le = joblib.load("label_encoder.pkl")
physical_le = joblib.load("physical_label_encoder.pkl")
emotional_le = joblib.load("emotional_label_encoder.pkl")

scaler = joblib.load("scaler.pkl")
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

    uploaded_file = st.file_uploader("", type=["wav", "mp3"],help="Upload baby cry audio (wav, mp3)")

    st.markdown("</div>", unsafe_allow_html=True)



predict = False  # important


col1, col2, col3 = st.columns([4, 2, 4])
with col2:
    predict = st.button("🔍 Predict", use_container_width=True)


GROUP_TO_NEEDS = {
    "physical_need": ["Hungry", "Belly Pain", "Burping", "Discomfort", "Cold/Hot"],
    "emotional_need": ["Tired", "Lonely", "Scared"],
    "normal": ["Laughing", "Silence"]
}
    

# ----------------------------
# Prediction logic
# ----------------------------
if predict and uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    temp_file = f"temp.{file_ext}"

    with open(temp_file, "wb") as f:
        f.write(uploaded_file.getbuffer())

    features = extract_features(temp_file)
    features = features.reshape(1, -1)
    features = scaler.transform(features)

    # ---------- Stage 1: Group prediction ----------
    group_pred = group_model.predict(features)
    group_result = group_le.inverse_transform(group_pred)[0]

    # ---------- Stage 2: Exact need prediction ----------
    if group_result == "physical_need":
        exact_pred = physical_model.predict(features)
        exact_need = physical_le.inverse_transform(exact_pred)[0]

    elif group_result == "emotional_need":
        exact_pred = emotional_model.predict(features)
        exact_need = emotional_le.inverse_transform(exact_pred)[0]

    else:
        exact_need = "Normal"

    final_output = f"{group_result.replace('_',' ').title()} ({exact_need.title()})"

    st.markdown(f"""
    <div class="card" style="margin-top:20px;">
        <div class="title">Prediction Result</div>
        <div style="font-size:22px;color:#2563eb;font-weight:700;margin-top:10px;">
            {final_output.upper()}
        </div>
    </div>
    """, unsafe_allow_html=True)

    os.remove(temp_file)

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