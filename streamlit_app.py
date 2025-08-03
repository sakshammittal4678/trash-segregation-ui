# streamlit_app.py

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- Model Loading and Caching ---
# Use st.cache_resource to load the model only once
@st.cache_resource
def load_my_model():
    """Loads and caches the Keras model."""
    model_path = 'models/20250731-12051753963551-inception_v3-full-image-set-3.h5'
    model = tf.keras.models.load_model(model_path, compile=False)
    # The 'compile=False' argument is often helpful for inference-only models.
    return model

# Define the labels your model can predict
# IMPORTANT: Make sure the order matches your model's training output
UNIQUE_LABELS = ['aerosol_cans', 'aluminum_food_cans', 'aluminum_soda_cans',
       'cardboard_boxes', 'cardboard_packaging', 'clothing',
       'coffee_grounds', 'disposable_plastic_cutlery', 'eggshells',
       'food_waste', 'glass_beverage_bottles',
       'glass_cosmetic_containers', 'glass_food_jars', 'magazines',
       'newspaper', 'office_paper', 'paper_cups', 'plastic_cup_lids',
       'plastic_detergent_bottles', 'plastic_food_containers',
       'plastic_shopping_bags', 'plastic_soda_bottles', 'plastic_straws',
       'plastic_trash_bags', 'plastic_water_bottles', 'shoes',
       'steel_food_cans', 'styrofoam_cups', 'styrofoam_food_containers',
       'tea_bags']

# --- Prediction Function ---
def get_preds(image_data, model):
    """Takes an image, preprocesses it, and returns the top 5 predictions."""
    # Preprocess the image to fit your model's input requirements
    # (e.g., resizing to 224x224, normalizing)
    image = image_data.resize((224, 224)) # Adjust size as per your model
    image = np.array(image)
    image = np.expand_dims(image, axis=0) # Add batch dimension
    image = tf.keras.applications.inception_v3.preprocess_input(image) # Preprocess for InceptionV3

    # Make prediction
    preds = model.predict(image)

    # Get top 5 predictions
    top_5_indexes = np.argsort(preds[0])[-5:][::-1]
    top_5_preds = [UNIQUE_LABELS[i] for i in top_5_indexes]
    return top_5_preds

# --- Streamlit App UI ---
st.title("♻️ Trash Segregation Classifier")
st.write("Upload an image of trash, and the model will predict its category.")

# Load the model
model = load_my_model()

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Classify the image
    st.write("Classifying...")
    top_predictions = get_preds(image, model)

    # Display the results
    st.success(f"**Top Prediction:** {top_predictions[0]}")
    st.write("**Other Possibilities:**")
    for pred in top_predictions[1:]:
        st.write(f"- {pred}")
