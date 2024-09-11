import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.express as px
import time

model_path = 'best_random_forest_model.pkl'  # Adjust this if needed
if os.path.exists(model_path):
    with open(model_path, 'rb') as file:
        model = joblib.load(file)
else:
    st.error(f"Model file not found at {model_path}")
    model = None

# Define the feature columns expected by the model
training_columns = [
    'temperature',  # 1. Current temperature
    'humidity',  # 2. Current humidity
    'precipIntensity',  # 3. Current precipitation intensity
    'precipProbability',  # 4. Probability of precipitation
    'windSpeed',  # 5. Wind speed
]

# Preprocess data
def preprocess_data(sensor_df, training_columns):
    try:
        # Ensure columns match model training columns
        for col in training_columns:
            if col not in sensor_df.columns:
                sensor_df[col] = 0  # Add missing columns

        # Reorder columns
        sensor_df = sensor_df[training_columns]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        sensor_data_imputed = imputer.fit_transform(sensor_df)

        # Return processed data as DataFrame
        sensor_data_scaled = pd.DataFrame(sensor_data_imputed, columns=training_columns)
        return sensor_data_scaled

    except Exception as e:
        st.error(f"Data preprocessing failed: {e}")
        return None

# Outlier detection function
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Predict function
def predict_failure(processed_data):
    if processed_data is None or processed_data.isna().any().any():
        st.error("Processed data contains NaN values. Prediction aborted.")
        return None

    if model:
        try:
            prediction = model.predict(processed_data)
            return prediction
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            return None
    else:
        st.error("Model not loaded.")
        return None

# Log data issues
def log_data_issues(df, training_columns):
    missing_cols = [col for col in training_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns: {missing_cols}")
    st.write(f"Data sample: {df.head()}")

# Streamlit app setup
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# Sidebar for navigation and input
st.sidebar.title("Predictive Maintenance System")

# File uploader in sidebar
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# User input sliders in sidebar (will only affect Data Insights and Predictions)
temperature_input = st.sidebar.slider("Select Temperature", min_value=-10.0, max_value=50.0, value=20.0)
humidity_input = st.sidebar.slider("Select Humidity", min_value=0.0, max_value=100.0, value=50.0)

# Display different sections in tabs
tab1, tab2, tab3 = st.tabs(["Data Insights", "Predictions", "Trends"])

if uploaded_file is not None:
    try:
        sensor_df = pd.read_csv(uploaded_file)

        if sensor_df.empty:
            st.error("Uploaded file is empty.")
        else:
            # Define columns for analysis
            appliance_columns = [
                'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 
                'Fridge [kW]', 'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]', 
                'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 'Microwave [kW]', 
                'Living room [kW]', 'Solar [kW]'
            ]

            variable_columns = [
                'temperature', 'humidity', 'precipIntensity', 'precipProbability', 'windSpeed', 
                'pressure', 'visibility', 'apparentTemperature', 'cloudCover', 'windBearing', 
                'dewPoint', 'precipProbability'
            ]

            # Tab 1: Data Insights
            with tab1:
                st.subheader("Electric Appliances")

                # Apply slider values for Data Insights
                sensor_df_insights = sensor_df.copy()
                sensor_df_insights['temperature'] = temperature_input
                sensor_df_insights['humidity'] = humidity_input

                if all(col in sensor_df.columns for col in appliance_columns):
                    appliance_df = sensor_df_insights[appliance_columns]
                    st.dataframe(appliance_df.describe().transpose())
                    
                    # Handle low mean values by filtering zeros
                    non_zero_appliance_df = appliance_df[(appliance_df > 0).any(axis=1)]
                    st.subheader("Non-Zero Appliance Usage Statistics")
                    st.dataframe(non_zero_appliance_df.describe().transpose())
                else:
                    st.error("Data does not contain required appliance columns.")

                st.subheader("Internal/External Variables")
                if all(col in sensor_df.columns for col in variable_columns):
                    variable_df = sensor_df_insights[variable_columns].drop_duplicates()  # Remove duplicates
                    st.dataframe(variable_df.describe().transpose())
                else:
                    st.error("Data does not contain required variable columns.")

            # Tab 2: Predictions
            with tab2:
                st.subheader("Failure Predictions")

                # Apply slider values for Predictions
                sensor_df_pred = sensor_df.copy()
                sensor_df_pred['temperature'] = temperature_input
                sensor_df_pred['humidity'] = humidity_input

                if set(training_columns).issubset(sensor_df_pred.columns):
                    processed_data = preprocess_data(sensor_df_pred, training_columns)

                    if processed_data is not None:
                        if st.sidebar.button("Predict Failure"):
                            with st.spinner('Running prediction...'):
                                failure_prediction = predict_failure(processed_data)
                                if failure_prediction is not None:
                                    st.write(f"Failure Prediction: {failure_prediction}")
                                    if np.any(failure_prediction == 1):
                                        st.write("Recommendation: Please schedule maintenance as soon as possible.")
                                    else:
                                        st.write("No immediate maintenance is required.")
                            st.success('Prediction complete!')

                        # Outlier detection
                        st.subheader("Outlier Detection")
                        outliers = detect_outliers(sensor_df_pred[training_columns])
                        if not outliers.empty:
                            st.write("Outliers detected in the dataset:")
                            st.dataframe(outliers)
                        else:
                            st.write("No outliers detected.")
                else:
                    st.error("Uploaded file does not contain all the required columns.")

            # Tab 3: Trends and Insights
            with tab3:
                st.subheader("Sensor Data Trends")

                # Use the actual dataset values for Trends (no slider input)
                if 'time' in sensor_df.columns:
                    sensor_df['time'] = pd.to_datetime(sensor_df['time'], errors='coerce')
                    sensor_df = sensor_df.dropna(subset=['time'])  # Remove rows with invalid time values
                    
                    # Drop rows with NaN in 'temperature' column to ensure valid plotting
                    sensor_df = sensor_df.dropna(subset=['temperature'])
                    
                    # Plot temperature trends over time using the actual dataset values
                    fig = px.line(sensor_df, x='time', y='temperature', title='Temperature Over Time')
                    st.plotly_chart(fig)
                else:
                    st.error("Time data is missing for trend analysis.")

                    st.subheader("Humidity Over Time")
                if 'humidity' in sensor_df.columns:
                  fig = px.line(sensor_df, x='time', y='humidity', title='Humidity Over Time')
                  st.plotly_chart(fig)
                else:
                    st.error("Humidity data is missing for trend analysis.")

                    st.subheader("Precipitation Intensity Over Time")
                if 'precipIntensity' in sensor_df.columns:
                  fig = px.line(sensor_df, x='time', y='precipIntensity', title='Precipitation Intensity Over Time')
                  st.plotly_chart(fig)
                else:
                  st.error("Precipitation intensity data is missing for trend analysis.")

                  st.subheader("Distribution of Key Variables")
                if set(variable_columns).issubset(sensor_df.columns):
                 fig = px.histogram(sensor_df, x='temperature', title='Temperature Distribution', nbins=50)
                 st.plotly_chart(fig)
     
                 fig = px.histogram(sensor_df, x='humidity', title='Humidity Distribution', nbins=50)
                 st.plotly_chart(fig)
                else:
                 st.error("Key variable data is missing.")

                 st.subheader("Wind Speed Over Time")
                if 'windSpeed' in sensor_df.columns:
                 fig = px.line(sensor_df, x='time', y='windSpeed', title='Wind Speed Over Time')
                 st.plotly_chart(fig)
                else:
                 st.error("Wind speed data is missing for trend analysis.")


                # Correlation heatmap for internal/external variables
                st.subheader("Correlation Heatmap")
                if set(variable_columns).issubset(sensor_df.columns):
                    numeric_df = sensor_df[variable_columns].select_dtypes(include=[np.number])
                    correlation_matrix = numeric_df.corr()
                    fig, ax = plt.subplots()
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                    st.pyplot(fig)
                else:
                    st.error("Data does not contain all required variable columns.")

                # Optionally add a download button for processed data
                st.subheader("Download Processed Data")
                csv = sensor_df.to_csv(index=False)
                st.download_button(label="Download CSV", data=csv, file_name='processed_data.csv', mime='text/csv')
    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.sidebar.write("Upload a CSV file to get started.")
