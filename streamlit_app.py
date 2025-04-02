import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Load and prepare the dataset
file_path = 'data.csv'  # Update with your dataset path
data = pd.read_csv(file_path)
data = pd.get_dummies(data, columns=['Transmission', 'Fuel Type', 'Vehicle Class'], drop_first=True)

# Streamlit UI setup
st.title("Fuel Consumption Prediction App")
st.sidebar.title("Input Parameters")

# Define numeric_data for use across different modes
numeric_data = data.select_dtypes(include=[np.number])

# Selection for display mode: EDA or Prediction
mode = st.sidebar.radio("Choose Mode:", ["EDA", "Prediction"])

# Display the first few rows of the dataset
if mode == "EDA":
    st.write("## Data Overview")
    st.dataframe(data.head())

    st.write("## Exploratory Data Analysis")
    # Showing correlation matrix only in EDA mode
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    st.pyplot()

# Prepare data for the model
X = numeric_data.drop('Fuel Consumption(Comb (L/100 km))', axis=1, errors='ignore')
y = data['Fuel Consumption(Comb (L/100 km))']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and Evaluate in Prediction mode
if mode == "Prediction":
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    st.write("## Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")

    st.write("## Predict Fuel Consumption")
    with st.sidebar.form("prediction_form"):
        engine_capacity = st.number_input("Engine Capacity (Liters)", value=2.0, step=0.1)
        number_of_cylinders = st.number_input("Number of Cylinders", value=4, step=1)
        gearbox_type = st.selectbox("Gearbox Type", options=['AM8', 'AS10', 'A8', 'A9', 'AM7', 'AS8', 'M6', 'AS6', 'AV', 'AS9', 'A10', 'A6', 'M5', 'M7', 'AV7', 'AV1', 'AM6', 'AS7', 'AV8', 'AV6', 'AV10', 'AS5', 'A7'])
        fuel_variant = st.selectbox("Fuel Variant", options=['Z', 'X', 'D', 'E'])
        vehicle_category = st.selectbox("Vehicle Category", options=['Compact', 'SUV: Small', 'Mid-size', 'Minicompact', 'SUV: Standard', 'Two-seater', 'Subcompact', 'Station wagon: Small', 'Station wagon: Mid-size', 'Full-size', 'Pickup truck: Small', 'Pickup truck: Standard', 'Minivan', 'Special purpose vehicle'])
        year_of_manufacture = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2021)
        drivetrain = st.selectbox("Drivetrain Type", ['FWD', 'RWD', 'AWD'])
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        input_data = pd.DataFrame({
            'Engine Size(L)': [engine_capacity],
            'Cylinders': [number_of_cylinders],
            'Year': [year_of_manufacture]
        })
        # Set all other columns to 0
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Set selected options to 1
        input_data[f'Transmission_{gearbox_type}'] = 1
        input_data[f'Fuel Type_{fuel_variant}'] = 1
        input_data[f'Vehicle Class_{vehicle_category}'] = 1
        # Assume there are columns for year and drivetrain in your model
        input_data['Drivetrain'] = drivetrain

        # Reorder columns to match training data
        input_data = input_data.reindex(columns=X_train.columns, fill_value=0)

        predicted_consumption = model.predict(input_data)
        st.write(f"Predicted Fuel Consumption (Comb): {predicted_consumption[0]:.2f} L/100 km")
