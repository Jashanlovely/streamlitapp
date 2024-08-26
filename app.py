import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Define the Streamlit app
st.title('Model Prediction')

# Create input fields for features
feature1 = st.text_input('Feature 1:')
feature2 = st.text_input('Feature 2:')
# Add more input fields as required

# When the predict button is clicked
if st.button('Predict'):
    # Convert input values to a numpy array
    features = np.array([[float(feature1), float(feature2)]])
    # Make a prediction
    prediction = model.predict(features)
    # Display the prediction
    st.write(f'Predicted Class: {prediction[0]}')
