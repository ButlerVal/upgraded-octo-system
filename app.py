import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from keras.applications import EfficientNetB0
import cv2
import os
import warnings
warnings.filterwarnings("ignore")

# --- 1. Define Model Architectures (from your notebook) ---

def create_cnn_model():
    """
    This is the exact CNN architecture from your ISAMANEproject.ipynb file.
    """
    model = Sequential()
    
    # Keras 3 requires the Input layer to be separate
    model.add(Input(shape=(96, 96, 3)))
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same')) # Padding was missing in your notebook but implied by later layers
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    return model

def create_efficientnet_model():
    """
    This is the exact EfficientNet architecture from your ISAMANEproject.ipynb file.
    """
    # Note: Keras 3 uses 'weights=None' when loading local weights.
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(96, 96, 3))
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) # This layer was in your notebook
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# --- 2. Create Models and Load Weights ---

# This decorator caches the models in memory after loading them once
@st.cache_resource
def load_all_models():
    # Create the model architectures
    cnn_model = create_cnn_model()
    efficientnet_model = create_efficientnet_model()
    
    # Load *only the weights* from your original .h5 files
    # Make sure you have re-uploaded the ORIGINAL .h5 files
    cnn_model.load_weights("cnn_model.h5") 
    efficientnet_model.load_weights("efficientnet_model.h5")
    
    return cnn_model, efficientnet_model

# Load models
cnn_model, efficientnet_model = load_all_models()

# --- 3. Rest of your application (No changes needed here) ---

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
        
        # We don't need eff_preprocess, your notebook preprocesses both the same way
        cnn_prob = cnn_model.predict(processed_img)[0][0]
        eff_prob = efficientnet_model.predict(processed_img)[0][0]

        # Ensemble average
        ensemble_prob = (cnn_prob + eff_prob) / 2
        label = "Malignant" if ensemble_prob >= 0.5 else "Benign"

        # Results
        st.success(f"✅ Prediction: **{label}**")
        st.info(f"**Confidence:** {ensemble_prob*100:.2f}%")
