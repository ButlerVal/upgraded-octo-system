import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import cv2
import os
import warnings
import requests # For downloading

warnings.filterwarnings("ignore")

# --- 1. Define Model URLs and Download Function ---

# !!! IMPORTANT: REPLACE THESE URLS !!!
# Get your URLs from the "Files" tab in your Hugging Face repo.
# Click on a file (e.g., cnn_model.keras), then click the "download" button.
# Copy that URL here.
# It should look like: https://huggingface.co/YOUR-USERNAME/YOUR-REPO-NAME/resolve/main/cnn_model.keras
CNN_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/cnn_model.keras"
EFF_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/efficientnet_model.keras"

# Local file names
CNN_PATH = "cnn_model.keras"
EFF_PATH = "efficientnet_model.keras"

# Helper function to download files
def download_file(url, local_filename):
    # Check if file already exists
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists. Skipping download.")
        return

    # Download file with a progress bar
    print(f"Downloading {local_filename}...")
    with st.spinner(f"Downloading {local_filename}... (This happens once on first boot)"):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print("Download complete.")

# --- 2. Load Models ---
@st.cache_resource
def load_all_models():
    # Download files from Hugging Face Hub
    download_file(CNN_URL, CNN_PATH)
    download_file(EFF_URL, EFF_PATH)
    
    # Load models from the files we just downloaded
    cnn_model = load_model(CNN_PATH)
    efficient_model = load_model(EFF_PATH)
    
    print("Full .keras models loaded successfully.")
    return cnn_model, efficient_model

cnn_model, efficient_model = load_all_models()

# --- 3. Constants & Helper Functions ---
IMG_SIZE = (96, 96)

def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02

# --- 4. Preprocessing Functions ---
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

uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file and cnn_model is not None and efficient_model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology image. Please upload a valid sample.")
    else:
        processed_img_cnn = preprocess_image_cnn(image)
        processed_img_effnet = preprocess_image_effnet(image)

        cnn_prob = cnn_model.predict(processed_img_cnn)[0][0]
        eff_prob = efficient_model.predict(processed_img_effnet)[0][0]

        ensemble_prob = (cnn_prob + eff_prob) / 2
        label = "Malignant" if ensemble_prob >= 0.5 else "Benign"
        
        if label == "Malignant":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")

        confidence = ensemble_prob if label == "Malignant" else (1 - ensemble_prob)
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        with st.expander("Show Individual Model Predictions"):
            st.write(f"**CNN Model:** {cnn_prob*100:.2f}% Malignant")
            st.write(f"**EfficientNet Model:** {eff_prob*100:.2f}% Malignant")
