
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Define the path to your .pkl files
model_path = '/content/drive/MyDrive/Colab Notebooks/GradientBoosting.pkl'
scaler_path = '/content/drive/MyDrive/scaler.pkl'

# Load the model
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Load the scaler
@st.cache_resource
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    return scaler

# Function to preprocess input data
def preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions, scaler):
    # Convert preferred_foot to binary (assuming 'Left' is 1 and 'Right' is 0)
    preferred_foot_binary = 1 if preferred_foot.lower() == 'left' else 0

    # Prepare input as numpy array
    input_data = np.array([potential, age, shooting, passing, physic, movement_reactions, preferred_foot_binary]).reshape(1, -1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    return input_data_scaled

# Function to handle prediction and display result
def predict_rating(model, scaler, preferred_foot, potential, age, shooting, passing, physic, movement_reactions):
    input_data = preprocess_input(preferred_foot, potential, age, shooting, passing, physic, movement_reactions, scaler)
    st.write(f"Preprocessed and scaled input data shape: {input_data.shape}, data: {input_data}")

    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit application
def main():
    st.title('Football Player Overall Rating Predictor')
    st.markdown('Enter the details of the football player to predict the overall rating.')

    # Load the model and scaler
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    # Input fields
    preferred_foot = st.selectbox('Preferred Foot', ['Left', 'Right'])
    potential = st.slider('Potential', min_value=50, max_value=100, value=80)
    age = st.slider('Age', min_value=16, max_value=40, value=25)
    shooting = st.slider('Shooting', min_value=50, max_value=100, value=70)
    passing = st.slider('Passing', min_value=50, max_value=100, value=70)
    physic = st.slider('Physic', min_value=50, max_value=100, value=70)
    movement_reactions = st.slider('Movement Reactions', min_value=50, max_value=100, value=70)

    # Predict button
    if st.button('Predict'):
        try:
            prediction = predict_rating(model, scaler, preferred_foot, potential, age, shooting, passing, physic, movement_reactions)
            st.success(f'Predicted Overall Rating: {prediction:.2f}')
        except Exception as e:
            st.error(f'Error predicting: {e}')

if __name__ == '__main__':
    main()
