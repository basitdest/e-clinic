import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# Load the pretrained model
model = load_model('Best_Model2.h5')

# Page 1: Skin Disease Classifier
def page1():
    st.title("Skin Disease Classifier")
    st.header("Upload an image for classification")
    
    # Upload an image
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Resize the image to 124x124
        image = Image.open(uploaded_image)
        image = image.resize((124, 124))
        
        st.image(image, caption="Uploaded Image (Resized)", use_column_width=True)

        # Predict the class
        image = np.array(image)
        image = image / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)

        # Define class labels (modify as per your model)
        class_labels = ['Eczema','Melanoma','Acne','Basal Cell Carcinoma','Benign Keratosis']

        # Show the predicted class
        st.subheader("Predicted Disease:")
        st.write(class_labels[np.argmax(prediction)])

import streamlit as st

# Page 2: About Us
def page2():
    st.title("About Us")
    st.header("Meet the Team")

    # Data for team members
    team_data = [
        {"name": "Abdul Basit", "Roll No": "28", "image_url": "images/member1.jpeg"},
        {"name": "Manzoor Ahmed", "Roll No": "29", "image_url": "images/member2.jpeg"},
        {"name": "Mujahid Ali", "Roll No": "39", "image_url": "images/member3.jpeg"},
        {"name": "Asif Ali", "Roll No": "42", "image_url": "images/member4.jpeg"},
        {"name": "Muzamil Hussain", "Roll No": "50", "image_url": "images/member5.jpeg"},
    ]

    # Create a row layout for displaying team members
    for member in team_data:
        # Add a horizontal line between team members
        st.write("---")
        
        # Display the image with a caption and resize it to 256x256
        st.image(member["image_url"], use_column_width=True, width=256, caption=member["name"])
        
        # Center-align the name and roll number
        st.markdown(f'<div style="text-align: center;">Name: {member["name"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">Roll No: {member["Roll No"]}</div>', unsafe_allow_html=True)

# Create a navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Skin Disease Classifier", "About Us"])

# Render the selected page
if page == "Skin Disease Classifier":
    page1()
elif page == "About Us":
    page2()