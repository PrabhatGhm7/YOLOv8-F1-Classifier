import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Loading the trained model
model = YOLO("best_model.pt")  

# Streamlit app title
st.title('F1 Liveries Classifier')

# Upload image through Streamlit
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Open the uploaded image
    img = Image.open(uploaded_image)
    
    # Perform prediction
    results = model(img)  
    
    # Display the image
    st.image(img, caption="Uploaded Image.", use_column_width=True)
    
    # Extract the results
    names_dict = results[0].names
    probs = results[0].probs.data.tolist()
    
    # Geting  predicted class and confidence
    predicted_class = names_dict[np.argmax(probs)]
    confidence_score = max(probs) * 100
    
    # Display prediction result
    st.write(f"Predicted Class: {predicted_class}")
    st.write(f"Confidence: {confidence_score:.2f}%")
