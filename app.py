import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('best_random_forest_model.pkl')

# Streamlit UI
st.title('Model Prediction')

# Input fields for each feature
feature1 = st.text_input('Feature 1')
feature2 = st.text_input('Feature 2')
# Add more input fields as needed for each feature

# Prediction button
if st.button('Predict'):
    # Convert input to the appropriate format
    feature_values = [float(feature1), float(feature2)]
    features = np.array([feature_values])

    # Make a prediction
    prediction = model.predict(features)

    # Display the result
    st.success(f'Predicted Class: {prediction[0]}')
