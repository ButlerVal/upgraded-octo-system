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
import requests # For downloading

warnings.filterwarnings("ignore")

# --- 1. Define Model URLs and Download Function ---

# !!! YOUR HUGGING FACE URLS !!!
CNN_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/cnn_model.keras"
EFF_URL = "https://huggingface.co/ButlerVal/upgraded-octo-system/resolve/main/efficientnet_model.keras"

# Local file names
CNN_PATH = "cnn_model.keras"
EFF_PATH = "efficientnet_model.keras"

def download_file(url, local_filename):
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists. Skipping download.")
        return
    
    print(f"Downloading {local_filename}...")
    with st.spinner(f"Downloading {local_filename}... (This happens once on first boot)"):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    print("Download complete.")

# --- 2. Define Model Architectures (from your notebook) ---
# This is the Keras 3-compatible architecture

def create_cnn_model():
    model = Sequential()
    model.add(Input(shape=(96, 96, 3))) # Keras 3 Input layer
    
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
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
    # Use the architecture from your *second* EfficientNet model
    base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(96, 96, 3))
    base_model.trainable = False 
    
    inputs = Input(shape=(96, 96, 3))
    x = base_model(inputs, training=False) 
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x) # This was in your second definition
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    
    return model

# --- 3. Load Models (The New Way) ---
@st.cache_resource
def load_all_models():
    # Step 1: Download the files
    download_file(CNN_URL, CNN_PATH)
    download_file(EFF_URL, EFF_PATH)
    
    # Step 2: Build the clean Keras 3 architectures
    cnn_model = create_cnn_model()
    efficient_model = create_efficientnet_model()
    
    # Step 3: Load *only the weights* from the downloaded files
    # This skips the buggy config file entirely
    cnn_model.load_weights(CNN_PATH)
    efficient_model.load_weights(EFF_PATH)
    
    print("Models built and weights loaded successfully.")
    return cnn_model, efficient_model

cnn_model, efficient_model = load_all_models()

# --- 4. Constants & Helper Functions ---
IMG_SIZE = (96, 96)

def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02

# --- 5. Preprocessing Functions ---
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

# --- 6. Streamlit UI ---
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
