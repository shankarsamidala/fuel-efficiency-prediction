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
st.title("Advanced Vehicle Fuel Consumption Predictor")
st.markdown("""
This comprehensive tool predicts the combined fuel consumption of vehicles based on an extensive range of features. Input your vehicle’s specifications below.
""")

# Data Overview Section
if st.expander("Data Overview", expanded=False):
    st.dataframe(data.head())

# Exploratory Data Analysis Section
if st.expander("Exploratory Data Analysis", expanded=False):
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    st.pyplot()

# Model Training and Evaluation Section
model = LinearRegression()
X = data.select_dtypes(include=[np.number]).drop('Fuel Consumption(Comb (L/100 km))', axis=1)
y = data['Fuel Consumption(Comb (L/100 km))']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

if st.expander("Model Training and Evaluation", expanded=False):
    st.write(f"Mean Squared Error of the model is: {mse:.2f}")

# Prediction Section
with st.expander("Predict Fuel Consumption", expanded=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        engine_capacity = st.number_input("Engine Capacity (Liters)", value=2.0, step=0.1)
        number_of_cylinders = st.number_input("Cylinders Count", value=4, step=1)
        vehicle_weight = st.number_input("Vehicle Weight (kg)", min_value=800, max_value=4000, value=1500)
    with col2:
        gear_type = st.selectbox("Gearbox Type", options=['Automatic', 'Manual', 'Semi-Automatic', 'CVT', 'Dual-Clutch'])
        fuel_variant = st.selectbox("Fuel Variant", options=['Petrol', 'Diesel', 'Electric', 'Hybrid'])
        drive_type = st.selectbox("Drive Train", options=['FWD', 'RWD', 'AWD'])
    with col3:
        car_class = st.selectbox("Car Category", options=['Compact', 'SUV: Small', 'Mid-size', 'Minicompact', 'SUV: Standard', 'Two-seater', 'Subcompact', 'Station wagon: Small', 'Station wagon: Mid-size', 'Full-size', 'Pickup truck: Small', 'Pickup truck: Standard', 'Minivan', 'Special purpose vehicle'])
        manufacturing_year = st.number_input("Manufacture Year", min_value=1990, max_value=2025, value=2021)
        emissions_rating = st.selectbox("Emissions Rating", options=['Euro 3', 'Euro 4', 'Euro 5', 'Euro 6', 'Euro 6d'])
    with col4:
        color_preference = st.selectbox("Vehicle Color", options=['White', 'Black', 'Silver', 'Red', 'Blue', 'Green'])
        air_conditioning = st.selectbox("Air Conditioning", options=['Yes', 'No'])
        parking_sensors = st.selectbox("Parking Sensors", options=['Front', 'Rear', 'Both', 'None'])
    
    submit_button = st.form_submit_button("Submit")

    if submit_button:
        input_data = pd.DataFrame({'Engine Size(L)': [engine_capacity], 'Cylinders': [number_of_cylinders], 'Year': [manufacturing_year], 'Weight(kg)': [vehicle_weight]})
        # Set all other columns to 0
        for col in X_train.columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Set selected options to 1
        input_data[f'Transmission_{gear_type}'] = 1
        input_data[f'Fuel Type_{fuel_variant}'] = 1
        input_data[f'Vehicle Class_{car_class}'] = 1
        # Assume the model could handle categorical data conversion for new features in the future
        input_data[f'Drive Type_{drive_type}'] = 1
        input_data[f'Emissions_{emissions_rating}'] = 1
        input_data[f'Color_{color_preference}'] = 1
        input_data[f'AC_{air_conditioning}'] = 1
        input_data[f'Parking Sensors_{parking_sensors}'] = 1

        # Reorder columns to match training data
        input_data = input_data[X_train.columns]

        predicted_consumption = model.predict(input_data)
        st.write(f"Predicted Fuel Consumption (Comb): {predicted_consumption[0]:.2f} L/100 km")
