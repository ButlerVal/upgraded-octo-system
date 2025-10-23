import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU mode (avoid TF GPU init freeze)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # Silence TensorFlow logs

import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    GlobalAveragePooling2D, BatchNormalization, Dropout
)
from keras.applications import EfficientNetB0
import cv2
import requests
import warnings

warnings.filterwarnings("ignore")

# --- 1. URLs and Model Paths ---
CNN_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/cnn_model.keras"
EFF_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/efficientnet_model.keras"
CNN_PATH = "cnn_model.keras"
EFF_PATH = "efficientnet_model.keras"


# --- 2. Helper: Download model with visible progress ---
def download_file_with_progress(url, filename, label):
    """Download file and show live progress bar."""
    if os.path.exists(filename):
        st.success(f"✅ {label} already downloaded.")
        return True

    st.info(f"📦 Downloading {label} ... This might take a few minutes.")
    progress = st.progress(0)

    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            progress.progress(int(downloaded / total_size * 100))
        progress.progress(100)
        st.success(f"✅ {label} downloaded successfully.")
        return True
    except Exception as e:
        st.error(f"❌ Failed to download {label}: {e}")
        return False


# --- 3. Model Architectures ---
def create_cnn_model():
    model = Sequential([
        Input(shape=(96, 96, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    return model


def create_efficientnet_model():
    inputs = Input(shape=(96, 96, 3))
    base = EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs)


# --- 4. Lazy Model Loader ---
def load_models():
    """Load both models after download."""
    ok1 = download_file_with_progress(CNN_URL, CNN_PATH, "CNN Model")
    ok2 = download_file_with_progress(EFF_URL, EFF_PATH, "EfficientNet Model")
    if not (ok1 and ok2):
        st.stop()

    cnn_model = create_cnn_model()
    eff_model = create_efficientnet_model()
    cnn_model.load_weights(CNN_PATH)
    eff_model.load_weights(EFF_PATH)
    return cnn_model, eff_model


# --- 5. Utilities ---
IMG_SIZE = (96, 96)


def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02


def preprocess_image_cnn(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    return np.expand_dims(image, 0)


def preprocess_image_effnet(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return np.expand_dims(image, 0)


# --- 6. Streamlit UI ---
st.set_page_config(page_title="Breast Cancer Detection", layout="centered")
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")
st.caption("Powered by CNN + EfficientNetB0 Ensemble")
st.write("---")

# Lazy model initialization
if "models_loaded" not in st.session_state:
    st.session_state.models_loaded = False
    cnn_model = eff_model = None

if not st.session_state.models_loaded:
    if st.button("🚀 Load Models"):
        with st.spinner("Loading models, please wait..."):
            cnn_model, eff_model = load_models()
            st.session_state.models_loaded = True
            st.session_state.cnn_model = cnn_model
            st.session_state.eff_model = eff_model
        st.success("✅ Models loaded successfully!")
else:
    cnn_model = st.session_state.cnn_model
    eff_model = st.session_state.eff_model
    st.success("✅ Models already loaded.")

st.write("---")

# File upload & prediction
uploaded_file = st.file_uploader("📤 Upload a Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file and st.session_state.models_loaded:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology slide image.")
    else:
        with st.spinner("🧠 Analyzing image..."):
            img_cnn = preprocess_image_cnn(image)
            img_eff = preprocess_image_effnet(image)

            cnn_prob = float(cnn_model.predict(img_cnn, verbose=0)[0][0])
            eff_prob = float(eff_model.predict(img_eff, verbose=0)[0][0])
            ensemble_prob = (cnn_prob + eff_prob) / 2

            label = "Malignant" if ensemble_prob >= 0.5 else "Benign"
            confidence = ensemble_prob if label == "Malignant" else (1 - ensemble_prob)

        if label == "Malignant":
            st.error(f"🩸 Prediction: **{label}** ({confidence*100:.2f}% confidence)")
        else:
            st.success(f"🌿 Prediction: **{label}** ({confidence*100:.2f}% confidence)")

        with st.expander("Show Individual Model Predictions"):
            st.write(f"**CNN Model:** {cnn_prob*100:.2f}% Malignant")
            st.write(f"**EfficientNetB0 Model:** {eff_prob*100:.2f}% Malignant")

