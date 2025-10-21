import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

# Load models
cnn_model = load_model("cnn_model_fixed.keras")
efficientnet_model = load_model("efficientnet_model.keras")

# Constants
IMG_SIZE = (96, 96)

# Basic histopathology feature detection (color, texture, etc.)
def is_histopathology_image(img):
    # Convert to HSV for better feature analysis
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)

    # Check for tissue-like pink/purple range
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])

    return pink_ratio > 0.02  # Adjustable threshold

# Image Preprocessing
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.set_page_config(page_title="Breast Cancer Detection (Ensemble)", layout="centered", initial_sidebar_state="collapsed")
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")

uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    # Check image type
    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology image. Please upload a valid sample.")
    else:
        # Preprocess and predict
        processed_img = preprocess_image(image)
        cnn_prob = cnn_model.predict(processed_img)[0][0]
        eff_prob = efficientnet_model.predict(processed_img)[0][0]

        # Ensemble average
        ensemble_prob = (cnn_prob + eff_prob) / 2
        label = "Malignant" if ensemble_prob >= 0.5 else "Benign"

        # Results
        st.success(f"✅ Prediction: **{label}**")
        st.info(f"**Confidence:** {ensemble_prob*100:.2f}%")
