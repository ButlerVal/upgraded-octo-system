import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
# FIX: Use modern tensorflow.keras imports
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
import cv2
import os
import warnings
import requests # For downloading
import gradio as gr # Import Gradio

warnings.filterwarnings("ignore")

# --- 1. Define Model URLs and Download Function ---
CNN_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/cnn_model.keras"
EFF_URL = "https://huggingface.co/Valisces/iasmane/resolve/main/efficientnet_model.keras"

CNN_PATH = "cnn_model.keras"
EFF_PATH = "efficientnet_model.keras"

def download_file(url, local_filename):
    # FIX: Changed os.path to os.path.exists
    if os.path.exists(local_filename):
        print(f"{local_filename} already exists. Skipping download.")
        return
    
    print(f"Downloading {local_filename}...")
    # This message will show in the server logs
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Download complete: {local_filename}")

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
    x = base_model.output 
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x) # Re-added Dense 128
    x = BatchNormalization()(x) # Re-added BN
    x = Dropout(0.5)(x) 
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# --- 3. Load Models (at global scope) ---
print("--- Initializing App ---")
models_loaded = False
try:
    # Step 1: Download the files
    download_file(CNN_URL, CNN_PATH)
    download_file(EFF_URL, EFF_PATH)
    
    # Step 2: Build the architectures
    cnn_model = create_cnn_model()
    efficient_model = create_efficientnet_model()
    
    # Step 3: Load weights from the downloaded files
    cnn_model.load_weights(CNN_PATH)
    efficient_model.load_weights(EFF_PATH)
    
    models_loaded = True
    print("✅ Models loaded successfully! Ready.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load models: {e}")
    # This will be caught by the predict function

# --- 4. Constants & Helper Functions ---
IMG_SIZE = (96, 96)

def is_histopathology_image(img):
    hsv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2HSV)
    pink_mask = cv2.inRange(hsv, (120, 30, 50), (170, 255, 255))
    pink_ratio = np.sum(pink_mask > 0) / (img.size[0] * img.size[1])
    return pink_ratio > 0.02

# --- 5. Preprocessing Functions ---
def preprocess_image_cnn(image):
    image = image.resize(IMG_SIZE).convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def preprocess_image_effnet(image):
    image = image.resize(IMG_SIZE).convert('RGB')
    image_array = np.array(image)
    image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# --- 6. Gradio Prediction Function ---
def predict_ensemble(input_image):
    """
    Takes a PIL image, runs checks, preprocessing, and ensemble prediction.
    Returns three outputs for the Gradio interface.
    """
    if not models_loaded:
        raise gr.Error("Models are not loaded. Check server logs.")

    # 1. Run image check (Gradio input is already a PIL Image)
    if not is_histopathology_image(input_image):
        raise gr.Error("This does not look like a histopathology image. Please upload a valid sample.")

    # 2. Preprocess for both models
    processed_cnn = preprocess_image_cnn(input_image)
    processed_eff = preprocess_image_effnet(input_image)

    # 3. Predict (using .predict())
    cnn_prob = cnn_model.predict(processed_cnn)[0][0]
    eff_prob = efficient_model.predict(processed_eff)[0][0]

    # 4. Ensemble
    ensemble_prob = (cnn_prob + eff_prob) / 2
    
    if ensemble_prob > 0.5:
        ensemble_label = "Malignant"
        ensemble_conf = ensemble_prob
    else:
        ensemble_label = "Benign"
        ensemble_conf = 1 - ensemble_prob

    # 5. Format outputs
    # Output 1: Label dictionary for the main result
    output_label = {ensemble_label: float(ensemble_conf)}
    
    # Output 2 & 3: Textbox strings for individual results
    output_cnn = f"{'Malignant' if cnn_prob > 0.5 else 'Benign'} ({cnn_prob*100:.2f}%)"
    output_eff = f"{'Malignant' if eff_prob > 0.5 else 'Benign'} ({eff_prob*100:.2f}%)"

    return output_label, output_cnn, output_eff

# --- 7. Gradio UI ---
title = "🧬 Histopathology Breast Cancer Classifier (Ensemble)"
description = (
    "Upload a (96x96) histopathology image. "
    "This app uses an ensemble of two models (a custom CNN and a pre-trained EfficientNetB0) "
    "to predict if the sample is benign or malignant."
)

iface = gr.Interface(
    fn=predict_ensemble,
    inputs=gr.Image(type="pil", label="Upload Histopathology Image"),
    outputs=[
        gr.Label(num_top_classes=2, label="Ensemble Prediction"),
        gr.Textbox(label="Custom CNN Prediction"),
        gr.Textbox(label="EfficientNetB0 Prediction")
    ],
    title=title,
    description=description,
    allow_flagging="never",
    examples=[
        ["https://huggingface.co/spaces/Valisces/iasmane/resolve/main/benign_sample.png"],
        ["https://huggingface.co/spaces/Valisces/iasmane/resolve/main/malignant_sample.png"]
    ]
)

# --- Launch the App ---
if __name__ == "__main__":
    if models_loaded:
        print("Starting Gradio app...")
        iface.launch()
    else:
        print("Gradio app not started due to model loading error.")
