import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

working_dir = os.path.dirname(os.path.abspath(__file__))

# Form the correct model path using os.path.join
model_path = os.path.join(working_dir, 'trained_model', 'plant_disease_prediction_model.h5')

# Ensure the model path is correct
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Form the correct path for class_indices.json
class_indices_path = os.path.join(working_dir, 'class_indices.json')

# Ensure the class_indices.json path is correct
if not os.path.exists(class_indices_path):
    raise FileNotFoundError(f"Class indices file not found at {class_indices_path}")

# Load the class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)
# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name


# Main title
st.title('Plant Disease Classifier')

# File uploader for image upload
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

    with col2:
        # Button to trigger classification
        if st.button('Classify'):
            # Call function to classify the uploaded image
            prediction = classify_image(uploaded_image)
            # Display the prediction result
            st.success(f'Prediction: {prediction}')

