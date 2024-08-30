import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer

# Load the model
model_path = 'best_random_forest_model.pkl'
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
else:
    st.error(f"Model file not found at {model_path}")
    model = None

# Define the feature columns expected by the model
training_columns = ['temperature', 'humidity', 'precipIntensity', 'precipProbability', 'windSpeed']

# Define the appliance columns
appliance_columns = [
    'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 
    'Fridge [kW]', 'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]', 
    'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]', 
    'Living room [kW]', 'Solar [kW]'
]

# Define the internal/external variable columns
variable_columns = [
    'temperature', 'humidity', 'precipIntensity', 'precipProbability', 'windSpeed', 
    'pressure', 'visibility', 'apparentTemperature', 'cloudCover', 'windBearing', 
    'dewPoint', 'precipProbability'
]

# Preprocess data
def preprocess_data(sensor_df, training_columns):
    # Ensure all required columns exist in the dataframe
    for col in training_columns:
        if col not in sensor_df.columns:
            sensor_df[col] = 0  # Fill missing columns with zeros
    
    sensor_df = sensor_df[training_columns]  # Keep only the relevant columns
    imputer = SimpleImputer(strategy='mean')
    sensor_data_imputed = imputer.fit_transform(sensor_df)
    sensor_data_scaled = pd.DataFrame(sensor_data_imputed, columns=training_columns)
    return sensor_data_scaled

# Outlier detection
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Prediction function
def predict_failure(processed_data):
    if processed_data.isna().any().any():
        st.error("Processed data contains NaN values.")
        return None
    try:
        prediction = model.predict(processed_data)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Set up Streamlit app layout
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# Sidebar: File uploader and input sliders
st.sidebar.title("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
temperature_input = st.sidebar.slider("Select Temperature", min_value=-10.0, max_value=50.0, value=20.0)
humidity_input = st.sidebar.slider("Select Humidity", min_value=0.0, max_value=100.0, value=50.0)

# Tab setup
tab1, tab2, tab3 = st.tabs(["Data Insights", "Predictions", "Trends"])

if uploaded_file:
    sensor_df = pd.read_csv(uploaded_file)

    # Add temperature and humidity input from sidebar sliders
    sensor_df['temperature'] = temperature_input
    sensor_df['humidity'] = humidity_input

    # **Tab 1: Data Insights**
    with tab1:
        st.subheader("Electric Appliances")
        # Ensure all appliance columns are present in the data
        existing_appliance_columns = [col for col in appliance_columns if col in sensor_df.columns]
        appliance_df = sensor_df[existing_appliance_columns]
        st.dataframe(appliance_df.describe().transpose())

        # Internal/External Variables section
        st.subheader("Internal/External Variables")
        existing_variable_columns = [col for col in variable_columns if col in sensor_df.columns]
        variable_df = sensor_df[existing_variable_columns]
        st.dataframe(variable_df.describe().transpose())

    # **Tab 2: Predictions**
    with tab2:
        st.subheader("Failure Predictions")
        if set(training_columns).issubset(sensor_df.columns):
            processed_data = preprocess_data(sensor_df, training_columns)
            if st.sidebar.button("Predict Failure"):
                prediction = predict_failure(processed_data)
                if prediction is not None:
                    st.write(f"Failure Prediction: {prediction}")
                    if np.any(prediction == 1):
                        st.write("Recommendation: Schedule maintenance soon.")
                    else:
                        st.write("No immediate maintenance required.")
        else:
            st.error("Uploaded file doesn't contain all the required columns for prediction.")

    # **Tab 3: Trends and Insights**
    with tab3:
        st.subheader("Sensor Data Trends")
        fig = px.line(sensor_df, x=sensor_df.index, y='temperature', title='Temperature Over Time')
        st.plotly_chart(fig)

        # Correlation heatmap
        st.subheader("Correlation Heatmap")
        if len(existing_variable_columns) > 1:
            correlation_matrix = variable_df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else:
            st.write("Not enough variables for correlation analysis.")

else:
    st.sidebar.write("Upload a CSV file to get started.")
