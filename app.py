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
import os
import warnings
import requests

# Disable warnings and force CPU (avoids indefinite CUDA init waits)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
warnings.filterwarnings("ignore")

# --- 1. Define Model URLs and Download Function ---
CNN_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/cnn_model.keras"
EFF_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/efficientnet_model.keras"

CNN_PATH = "cnn_model.keras"
EFF_PATH = "efficientnet_model.keras"


def download_file(url, local_filename):
    """Download file safely with progress updates."""
    if os.path.exists(local_filename):
        st.info(f"✅ {local_filename} already exists. Skipping download.")
        return

    st.warning(f"📦 Downloading `{local_filename}`... This may take a few minutes.")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        block_size = 8192

        with open(local_filename, 'wb') as f:
            for data in response.iter_content(block_size):
                f.write(data)
                downloaded += len(data)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    st.progress(int(percent))
        st.success(f"✅ Downloaded `{local_filename}` successfully!")
    except Exception as e:
        st.error(f"❌ Failed to download {local_filename}: {str(e)}")
        st.stop()


# --- 2. Define Model Architectures ---
def create_cnn_model():
    model = Sequential([
        Input(shape=(96, 96, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
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
    base_model = EfficientNetB0(weights=None, include_top=False, input_tensor=inputs)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model


# --- 3. Load Models Safely ---
@st.cache_resource(show_spinner=False)
def load_all_models():
    # Download weights
    download_file(CNN_URL, CNN_PATH)
    download_file(EFF_URL, EFF_PATH)

    # Build architectures
    cnn_model = create_cnn_model()
    efficient_model = create_efficientnet_model()

    # Load weights
    cnn_model.load_weights(CNN_PATH)
    efficient_model.load_weights(EFF_PATH)

    return cnn_model, efficient_model


# --- 4. Helper Functions ---
IMG_SIZE = (96, 96)


def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02


def preprocess_image_cnn(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image_effnet(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image


# --- 5. Streamlit UI ---
st.set_page_config(page_title="Breast Cancer Detection (Ensemble)", layout="centered")
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")
st.write("---")

# Load models once
with st.spinner("⏳ Loading models (this may take up to 2–3 minutes on first run)..."):
    cnn_model, efficient_model = load_all_models()

st.success("✅ Models loaded successfully!")
st.write("---")

# --- Upload & Predict ---
uploaded_file = st.file_uploader("📤 Upload a Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file and cnn_model is not None and efficient_model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology image. Please upload a valid sample.")
    else:
        with st.spinner("🧠 Analyzing image..."):
            processed_img_cnn = preprocess_image_cnn(image)
            processed_img_effnet = preprocess_image_effnet(image)

            cnn_prob = float(cnn_model.predict(processed_img_cnn, verbose=0)[0][0])
            eff_prob = float(efficient_model.predict(processed_img_effnet, verbose=0)[0][0])

            ensemble_prob = (cnn_prob + eff_prob) / 2
            label = "Malignant" if ensemble_prob >= 0.5 else "Benign"
            confidence = ensemble_prob if label == "Malignant" else (1 - ensemble_prob)

        if label == "Malignant":
            st.error(f"🩸 Prediction: **{label}**")
        else:
            st.success(f"🌿 Prediction: **{label}**")

        st.info(f"**Confidence:** {confidence * 100:.2f}%")

        with st.expander("Show Individual Model Predictions"):
            st.write(f"**CNN Model:** {cnn_prob * 100:.2f}% Malignant")
            st.write(f"**EfficientNet Model:** {eff_prob * 100:.2f}% Malignant")

