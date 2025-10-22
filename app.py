import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. Load Models ---
# This loads your new, error-free .keras models.
# @st.cache_resource ensures they only load once.
@st.cache_resource
def load_all_models():
    # Make sure you've uploaded 'cnn_model.keras' and 'efficientnet_model.keras'
    try:
        cnn_model = load_model("cnn_model.keras")
        print("CNN model loaded.")
    except Exception as e:
        st.error(f"Error loading cnn_model.keras: {e}")
        return None, None
        
    try:
        # Use the name you saved, e.g., 'efficientnet_best.keras' or 'efficientnet_model.keras'
        efficient_model = load_model("efficientnet_model.keras") 
        print("EfficientNet model loaded.")
    except Exception as e:
        st.error(f"Error loading efficientnet_model.keras: {e}")
        return None, None
        
    return cnn_model, efficient_model

cnn_model, efficient_model = load_all_models()

# --- 2. Constants & Helper Functions ---
IMG_SIZE = (96, 96)

# Helper function to check if the image is a histopathology slide
def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    # Check for pink/purple range, typical of H&E staining
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02 # Threshold, adjustable

# --- 3. Preprocessing Functions ---
# We need TWO different preprocessing functions now.

def preprocess_image_cnn(image):
    """Prepares image for the CNN model (scales to 0-1)."""
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0  # Scale to 0-1
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

def preprocess_image_effnet(image):
    """Prepares image for the EfficientNet model (keeps 0-255)."""
    image = image.resize(IMG_SIZE)
    image = np.array(image)           # No scaling
    image = np.expand_dims(image, axis=0) # Add batch dimension
    return image

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Breast Cancer Detection (Ensemble)", layout="centered")
st.title("🧬 Histopathology Breast Cancer Classifier (Ensemble)")

uploaded_file = st.file_uploader("📤 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file and cnn_model is not None and efficient_model is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="🔍 Uploaded Image", use_container_width=True)

    # Check if the image is valid
    if not is_histopathology_image(image):
        st.error("🚫 This doesn't look like a histopathology image. Please upload a valid sample.")
    else:
        # Process image for each model
        processed_img_cnn = preprocess_image_cnn(image)
        processed_img_effnet = preprocess_image_effnet(image)

        # Predict with both models
        cnn_prob = cnn_model.predict(processed_img_cnn)[0][0]
        eff_prob = efficient_model.predict(processed_img_effnet)[0][0]

        # Ensemble average
        ensemble_prob = (cnn_prob + eff_prob) / 2
        label = "Malignant" if ensemble_prob >= 0.5 else "Benign"

        # --- Display Results ---
        st.subheader("🔬 Analysis Complete")
        
        if label == "Malignant":
            st.error(f"Prediction: **{label}**")
        else:
            st.success(f"Prediction: **{label}**")

        # Confidence score
        confidence = ensemble_prob if label == "Malignant" else (1 - ensemble_prob)
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        # Expander for detailed model outputs
        with st.expander("Show Individual Model Predictions"):
            st.write(f"**CNN Model:** {cnn_prob*100:.2f}% Malignant")
            st.write(f"**EfficientNet Model:** {eff_prob*100:.2f}% Malignant")

else:
    st.info("Waiting for models to load or for image to be uploaded...")
