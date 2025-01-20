import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from dotenv import load_dotenv
import os
import cv2
import numpy as np


import warnings
warnings.filterwarnings("ignore")

# Define category mapping
category = {
    0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: "Broccoli", 5: 'Cabbage', 
    6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 
    11: 'Potato', 12: 'Pumpkin', 13: "Radish", 14: "Tomato"
}

# Cache the model loading for efficiency
@st.cache_resource
def load_trained_model(path_to_model):
    return load_model(path_to_model)


# Function to predict image category
def predict_image(file, model):
    img_ = image.load_img(file, target_size=(224, 224))
    img_ = ImageEnhance.Color(img_).enhance(1.35)
    img_ = ImageEnhance.Contrast(img_).enhance(1.45)
    
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
 
    st.image(uploaded_image)
    st.markdown(f"<h4 style='text-align: center;'>Prediction - {category[index]}</h4>", unsafe_allow_html=True)

    
    return category[index]

# Function to fetch recipes using the predicted category
def fetch_recipes(predicted_category):


    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    headers = {"Content-Type": "application/json"}
    prompt = f"Recommend 3 recipes using this ingredient: {predicted_category}"
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        return f"Error: {response.status_code} - {response.text}"

# Main Streamlit App
st.title("Image-Based Recipe Recommender")
st.write("Upload an image of a vegetable, and I'll suggest recipes for it!")

# Load the model
# path_to_model = '/home/unthinkable-lap/Desktop/d/model_inceptionV3_epoch3.pkl'
# model = load_trained_model(path_to_model)

import pickle

with open('model_inceptionV3_epoch3.pkl', 'rb') as file:
    model = pickle.load(file)


# Image upload field
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image:
    if st.button("Predict and Get Recipes"):
        with st.spinner("Processing the image..."):
            predicted_category = predict_image(uploaded_image, model)
        
        with st.spinner("Fetching recipes..."):
            response = fetch_recipes(predicted_category)
        
        if isinstance(response, dict):
            try:
                st.subheader("Here are your recommended recipes:")
                candidates = response.get('candidates', [])
                if candidates:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts:
                        for i, part in enumerate(parts, 1):
                            st.write(f"### Recipes:")
                            st.write(part.get('text', 'No recipe details available.'))
                    else:
                        st.error("No parts found in the response.")
                else:
                    st.error("No candidates found in the response.")
            except KeyError:
                st.error("Failed to parse the response. Please check the API output.")
        else:
            st.error(response)
