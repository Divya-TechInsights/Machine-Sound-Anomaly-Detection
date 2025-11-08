import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.models import load_model #type: ignore
import librosa.display
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore', category=UserWarning)
tf.get_logger().setLevel('ERROR')


# ======================================
# PAGE CONFIG
# ======================================
st.set_page_config(
    page_title="🎧 Pump Sound Anomaly Detector",
    layout="wide",
    page_icon="🎵"
)

# ======================================
# CUSTOM CSS
# ======================================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
            color: #B2EBF2;
            font-family: 'Segoe UI', sans-serif;
        }

        /* NAVBAR */
        .navbar {
            display: flex;
            justify-content: center;
            gap: 30px;
            background: linear-gradient(90deg, #1f4037, #99f2c8);
            padding: 12px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.3);
        }
        .nav-item {
            font-weight: bold;
            color: white;
            text-decoration: none;
            font-size: 1.1em;
            padding: 8px 15px;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        .nav-item:hover {
            background: rgba(255,255,255,0.2);
            transform: scale(1.05);
        }
        .active {
            background: rgba(255,255,255,0.25);
            border: 1px solid white;
        }

        /* Buttons */
        div.stButton > button {
            background: linear-gradient(45deg, #00adb5, #007580);
            color: white;
            border-radius: 8px;
            padding: 10px 25px;
            border: none;
            transition: all 0.3s ease;
        }
        div.stButton > button:hover {
            background: linear-gradient(45deg, #00ffcc, #00adb5);
            box-shadow: 0 0 15px #00e6e6;
            transform: scale(1.07);
        }

        /* Footer */
        .footer {
            color: #b0bec5;
            text-align: center;
            font-size: 0.9em;
            margin-top: 40px;
            padding-top: 10px;
            border-top: 1px solid #455a64;
        }
        section[data-testid="stSidebar"] {
    background: linear-gradient(45deg, #007580, #00adb5) !important;
    color: #E0F7FA !important;
    border-right: 2px solid #00adb5;
}
     section[data-testid="stSidebar"] .stAlert {
    background: #E0F7FA !important;   /* Light icy blue */
    color: #004D40 !important;        /* Deep teal text for contrast */
    border-radius: 10px;
    box-shadow: 0 0 10px rgba(0, 173, 181, 0.3);
}
       

    </style>
""", unsafe_allow_html=True)

# ======================================
# NAVIGATION STATE
# ======================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

def set_page(page_name):
    st.session_state.page = page_name

# ======================================
# NAVIGATION BAR (REAL FUNCTIONALITY)
# ======================================
cols = st.columns(3)
with cols[0]:
    if st.button("HOME", key="home", use_container_width=True):
        set_page("Home")
with cols[1]:
    if st.button("ANALYZE DATASET", key="analyze", use_container_width=True):
        set_page("Analyze")
with cols[2]:
    if st.button("UPLOAD SOUND", key="upload", use_container_width=True):
        set_page("Upload")

# ======================================
# MODEL LOADING
# ======================================
model_choice = st.sidebar.selectbox(
    "🎛️ Select Model File (.keras):",
    ["Encoder_Model.keras", "Encoder_Model2.keras", "Encoder_Model3.keras"]
)

if os.path.exists(model_choice):
    model = load_model(model_choice)
    input_shape = model.input_shape
    model_input_dim = input_shape[-1]
    st.sidebar.success(f"✅ Loaded: {model_choice}")
else:
    model = None
    model_input_dim = None
    st.sidebar.error(f"⚠️ Model '{model_choice}' not found.")

# ======================================
# FEATURE EXTRACTION
# ======================================
def extract_features(file_path, n_features=18):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_features)
    return np.mean(mfcc, axis=1)

def display_audio_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    db_mel = librosa.power_to_db(mel, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(db_mel, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram", color="white")
    plt.tight_layout()
    st.pyplot(fig)

# ======================================
# PAGE LOGIC
# ======================================
if st.session_state.page == "Home":
    st.title("MACHINE SOUND ANOMALY DETECTION")
    st.subheader("AI-Powered Industrial Sound Analysis using Deep Autoencoder")

    st.markdown("""
    Welcome to the **Pump Sound Anomaly Detection WebApp**!  
    This AI tool detects abnormal pump sounds using deep learning.
    
    ### ✨ Key Features:
    - Real-time anomaly detection  
    - Interactive spectrogram visualization  
    - Deep Autoencoder architecture  
    - Sleek, futuristic dark interface  
    """)

elif st.session_state.page == "Analyze":
    st.header("Analyze Dataset")
    option = st.selectbox("Select Dataset Type:", ["Normal 🔵", "Abnormal 🔴"])
    folder = "normal" if "Normal" in option else "abnormal"

    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(".wav")]
        if files:
            selected_file = st.selectbox("Select an Audio File:", files)
            file_path = os.path.join(folder, selected_file)
            st.audio(file_path, format="audio/wav")
            display_audio_features(file_path)

            if model:
                features = extract_features(file_path, n_features=model_input_dim)
                features = np.expand_dims(features, axis=0)
                reconstruction = model.predict(features)
                mse = np.mean(np.power(features - reconstruction, 2))
                st.write(f"### Reconstruction Error: `{mse:.6f}`")
                if mse > 0.02:
                    st.error("🚨 **Anomaly Detected!**")
                else:
                    st.success("✅ **Normal Sound Detected!**")
        else:
            st.warning("⚠️ No `.wav` files found in the folder.")
    else:
        st.error(f"❌ Folder '{folder}' not found!")

elif st.session_state.page == "Upload":
    st.header("Upload Sound")
    uploaded_file = st.file_uploader("Upload a Pump Sound File (.wav)", type=["wav"])
    if uploaded_file:
        st.audio(uploaded_file, format="audio/wav")
        with open("temp.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        display_audio_features("temp.wav")

        if model:
            features = extract_features("temp.wav", n_features=model_input_dim)
            features = np.expand_dims(features, axis=0)
            reconstruction = model.predict(features)
            mse = np.mean(np.power(features - reconstruction, 2))
            st.write(f"### Reconstruction Error: `{mse:.6f}`")
            if mse > 0.02:
                st.error("🚨 **Anomaly Detected!**")
            else:
                st.success("✅ **Normal Pump Sound Detected!**")
        os.remove("temp.wav")

# ======================================
# FOOTER
# ======================================
st.markdown('<div class="footer">© 2025 | Developed by Divya Tyagi | Powered by Streamlit 💻</div>', unsafe_allow_html=True)
