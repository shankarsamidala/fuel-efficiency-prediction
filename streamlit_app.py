import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and prepare the dataset
def load_data(filepath):
    logging.info("Loading data...")
    data = pd.read_csv(filepath)
    data = pd.get_dummies(data, columns=['Transmission', 'Fuel Type', 'Vehicle Class'], drop_first=True)
    return data

data = load_data('data.csv')  # Update with your dataset path

# Streamlit UI setup
st.set_page_config(page_title="Fuel Consumption Predictor", layout="wide")
st.title("Enterprise-Level Vehicle Fuel Consumption Prediction System")

# Sidebar for navigation
with st.sidebar:
    st.title("Navigation")
    page = st.radio("Select a Page", ["Overview", "Data Analysis", "Model Training", "Prediction"])

# Main Page Layout
def show_overview(data):
    st.subheader("Data Overview")
    st.dataframe(data.head())
    st.markdown("### Description")
    st.text("Summary stats, data types, and more descriptive information can be placed here.")

def show_data_analysis(data):
    st.subheader("Exploratory Data Analysis")
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
    st.pyplot()

def train_model(data):
    st.subheader("Model Training and Evaluation")
    X = data.drop('Fuel Consumption(Comb (L/100 km))', axis=1)
    y = data['Fuel Consumption(Comb (L/100 km))']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write("Mean Squared Error: {:.2f}".format(mse))
    return model

def prediction_interface(model, X_train):
    st.subheader("Predict Fuel Consumption")
    # Input forms and prediction logic goes here
    # This should include input validation and possibly session management for user inputs

# Routing logic
if page == "Overview":
    show_overview(data)
elif page == "Data Analysis":
    show_data_analysis(data)
elif page == "Model Training":
    model = train_model(data)
elif page == "Prediction":
    prediction_interface(model, data)

st.markdown("## Additional Information")
st.markdown("""
This system is designed to seamlessly integrate with existing enterprise resource planning (ERP) and other analytical tools to enhance decision-making processes.
""")
