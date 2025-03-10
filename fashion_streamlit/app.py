# app.py - Streamlit GUI for Fashion Classification Model (Updated)

import os
import io
import base64
import requests
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import json

# Configuration
API_URL = "http://localhost:8000/api"  # Django API URL

st.set_page_config(
    page_title="Fashion Classification App",
    page_icon="👔",
    layout="wide"
)

# Title and description
st.title("Fashion Item Classification")
st.markdown("""
This application analyzes fashion items in images and predicts their:
- Color
- Product Type
- Season
- Gender
""")

# Sidebar
st.sidebar.title("About")
st.sidebar.info(
    "This application uses a deep learning model (ResNet50) "
    "trained on fashion data to classify clothing items."
)

# Load model information
@st.cache_data
def load_model_info():
    try:
        response = requests.get(f"{API_URL}/info/")
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Failed to load model information: {str(e)}")
        return None

# Function to predict image
def predict_image(image):
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    # Send to API
    try:
        response = requests.post(
            f"{API_URL}/predict/",
            data=json.dumps({"image": img_str}),
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()['predictions']
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Failed to get prediction: {str(e)}")
        return None

# Load model info
model_info = load_model_info()
if model_info:
    with st.sidebar.expander("Model Information"):
        st.write(f"**Device:** {model_info.get('device', 'Unknown')}")
        st.write(f"**Number of Colors:** {len(model_info.get('colors', []))}")
        st.write(f"**Number of Product Types:** {len(model_info.get('product_types', []))}")
        st.write(f"**Number of Seasons:** {len(model_info.get('seasons', []))}")
        st.write(f"**Number of Genders:** {len(model_info.get('genders', []))}")

# Image upload
st.subheader("Upload an image of a fashion item")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Sample images option
sample_option = st.checkbox("Or use a sample image")
if sample_option:
    sample_images = ["test.jpg"]
    # Replace with actual sample images if available
    sample_selection = st.selectbox("Select a sample image:", sample_images)

# Process image
if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    
    with col1:
        # Using use_container_width instead of the deprecated use_column_width
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Make prediction
    with st.spinner("Analyzing image..."):
        prediction = predict_image(image)
    
    if prediction:
        with col2:
            st.subheader("Prediction Results")
            
            # Create metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Color", prediction['color'])
                st.metric("Product Type", prediction['product_type'])
            with col_b:
                st.metric("Season", prediction['season'])
                st.metric("Gender", prediction['gender'])
            
            # Create a DataFrame for the results
            results_df = pd.DataFrame([prediction])
            st.dataframe(results_df)
            
            # Download option
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="fashion_prediction.csv",
                mime="text/csv",
            )
        
elif sample_option:
    # Load and display sample image
    try:
        # Update this path to where sample images are stored
        image_path = f"/sample_images/{sample_selection}"
        image = Image.open(image_path).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Using use_container_width instead of the deprecated use_column_width
            st.image(image, caption=f"Sample Image: {sample_selection}", use_container_width=True)
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            prediction = predict_image(image)
        
        if prediction:
            with col2:
                st.subheader("Prediction Results")
                
                # Create metrics
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Color", prediction['color'])
                    st.metric("Product Type", prediction['product_type'])
                with col_b:
                    st.metric("Season", prediction['season'])
                    st.metric("Gender", prediction['gender'])
                
                # Create a DataFrame for the results
                results_df = pd.DataFrame([prediction])
                st.dataframe(results_df)
    except Exception as e:
        st.error(f"Error loading or processing sample image: {str(e)}")

# Additional Sections
st.markdown("---")
st.subheader("How It Works")
st.write("""
This application uses a deep learning model (ResNet50) trained on fashion data to classify clothing items.
The model was trained to recognize multiple attributes simultaneously:

1. **Upload Image**: Submit an image of a fashion item
2. **Image Processing**: The image is resized and normalized
3. **Feature Extraction**: The ResNet50 model extracts visual features
4. **Multi-task Classification**: Separate classifiers predict various attributes
5. **Results Display**: Color, product type, season, and gender predictions
""")

if model_info:
    # Show available classifications
    with st.expander("Available Classifications"):
        tab1, tab2, tab3, tab4 = st.tabs(["Colors", "Product Types", "Seasons", "Genders"])
        
        with tab1:
            st.write(model_info.get('colors', []))
        with tab2:
            st.write(model_info.get('product_types', []))
        with tab3:
            st.write(model_info.get('seasons', []))
        with tab4:
            st.write(model_info.get('genders', []))