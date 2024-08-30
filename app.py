import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
import plotly.express as px

# Load the model
model_file = os.path.join(model_path, 'best_random_forest_model.pkl')

# Load the model if it exists
model_file = 'best_random_forest_model.pkl'

# Check if the model file exists and load it
if os.path.exists(model_file):
    with open(model_file, 'rb') as file:
        model = joblib.load(file)
else:
    st.error(f"Model file not found at {model_file}")
    model = None

# Define the expected feature columns (those used in training)
training_columns = [
    'temperature',  # 1. Current temperature
    'humidity',  # 2. Current humidity
    'precipIntensity',  # 3. Current precipitation intensity
    'precipProbability',  # 4. Probability of precipitation
    'windSpeed',  # 5. Wind speed
]

# Preprocess the uploaded data
def preprocess_data(sensor_df):
    try:
        # Add missing columns with zeros and reorder them
        for col in training_columns:
            if col not in sensor_df.columns:
                sensor_df[col] = 0

        sensor_df = sensor_df[training_columns]

        # Impute missing values
        imputer = SimpleImputer(strategy='mean')
        sensor_data_imputed = imputer.fit_transform(sensor_df)
        sensor_data_scaled = pd.DataFrame(sensor_data_imputed, columns=training_columns)

        return sensor_data_scaled
    except Exception as e:
        st.error(f"Data preprocessing failed: {e}")
        return None

# Predict function
def predict_failure(processed_data):
    if processed_data is None or processed_data.isna().any().any():
        st.error("Processed data contains NaN values. Prediction aborted.")
        return None

    try:
        prediction = model.predict(processed_data)
        return prediction
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# Outlier detection function
def detect_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    return df[((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Streamlit app layout and features
st.set_page_config(page_title="Predictive Maintenance System", layout="wide")

# Sidebar for inputs
st.sidebar.title("Control Panel")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# User input sliders
temperature_input = st.sidebar.slider("Select Temperature", min_value=-10.0, max_value=50.0, value=20.0)
humidity_input = st.sidebar.slider("Select Humidity", min_value=0.0, max_value=100.0, value=50.0)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Data Insights", "Predictions", "Trends"])

if uploaded_file is not None:
    sensor_df = pd.read_csv(uploaded_file)

    # Override columns with user inputs
    sensor_df['temperature'] = temperature_input
    sensor_df['humidity'] = humidity_input

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
        appliance_df = sensor_df[appliance_columns]
        st.dataframe(appliance_df.describe().transpose())

        non_zero_appliance_df = appliance_df[(appliance_df > 0).any(axis=1)]
        st.subheader("Non-Zero Appliance Usage Statistics")
        st.dataframe(non_zero_appliance_df.describe().transpose())

        st.subheader("Internal/External Variables")
        variable_df = sensor_df[variable_columns].drop_duplicates()
        st.dataframe(variable_df.describe().transpose())

    # Tab 2: Predictions
    with tab2:
        st.subheader("Failure Predictions")

        if set(training_columns).issubset(sensor_df.columns):
            processed_data = preprocess_data(sensor_df)

            if processed_data is not None and st.sidebar.button("Predict Failure"):
                with st.spinner('Running prediction...'):
                    failure_prediction = predict_failure(processed_data)
                    if failure_prediction is not None:
                        st.write(f"Failure Prediction: {failure_prediction}")
                        if np.any(failure_prediction == 1):
                            st.write("Recommendation: Please schedule maintenance.")
                        else:
                            st.write("No immediate maintenance required.")
                st.success('Prediction complete!')

            st.subheader("Outlier Detection")
            outliers = detect_outliers(sensor_df[training_columns])
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
        fig = px.line(sensor_df, x=sensor_df.index, y='temperature', title='Temperature Over Time')
        st.plotly_chart(fig)

        st.subheader("Correlation Heatmap")
        numeric_df = sensor_df[variable_columns].select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        fig, ax = plt.subplots()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.subheader("Download Processed Data")
        csv = sensor_df.to_csv(index=False)
        st.download_button(label="Download CSV", data=csv, file_name='processed_data.csv', mime='text/csv')
else:
    st.sidebar.write("Upload a CSV file to get started.")
