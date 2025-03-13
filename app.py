import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Load the trained model, scaler, and label encoder
model = joblib.load('../stacked_model.pkl')
scaler = joblib.load('../scaler.pkl')
label_encoder = joblib.load('../label_encoder.pkl')

# Function to make predictions
def predict_stability(hr, n):
    input_data = np.array([[hr, n]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of class 1
    
    # Decode the prediction to original label
    status = label_encoder.inverse_transform(prediction)[0]
    return status, probability

# Function to generate isoprobability contours
def generate_isoprobability_contour(hr, n):
    # Create a grid for hydraulic radius (HR) and stability number (N)
    hr_min, hr_max = 0, 10  # Example range for HR
    n_min, n_max = 0, 50    # Example range for N
    grid_hr, grid_n = np.meshgrid(
        np.linspace(hr_min, hr_max, 100),
        np.linspace(n_min, n_max, 100)
    )
    
    # Combine grid points
    grid_points = np.c_[grid_hr.ravel(), grid_n.ravel()]
    
    # Scale the grid points
    grid_points_scaled = scaler.transform(grid_points)
    
    # Predict probabilities on the grid
    grid_probs = model.predict_proba(grid_points_scaled)[:, 1].reshape(grid_hr.shape)
    
    # Plot isoprobability contours
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        grid_hr,
        grid_n,
        grid_probs,
        levels=np.linspace(0, 1, 11),
        cmap="coolwarm",
        alpha=0.7
    )
    plt.colorbar(label="Probability of Unstable Status")
    plt.scatter(hr, n, color='red', label="Input Point", s=100, edgecolor='black')  # Highlight user input
    plt.title("Isoprobability Contour for Stope Stability")
    plt.xlabel("Hydraulic Radius (HR)")
    plt.ylabel("Stability Number (N)")
    plt.legend()
    return plt

# Function to generate a report using Gemini API
def generate_report(status, probability):
    # Configure the Gemini API
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    
    # Define the prompt for Gemini
    prompt = f"""
    Generate a detailed report for the stope stability prediction.
    Prediction: {status}
    Probability of Unstable Status: {probability:.2%}
    Provide insights into the stability condition and recommend actions.
    """
    
    # Call Gemini API
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    
    # Extract the generated report
    if response.text:
        return response.text
    else:
        st.error(f"Error generating report: No response from Gemini API.")
        return None

# Streamlit App Design
st.set_page_config(page_title="Stope Stability Prediction", layout="wide")

# Sidebar for Inputs
with st.sidebar:
    st.title("Input Parameters")
    st.markdown("Enter the values below:")
    hr = st.number_input("Hydraulic Radius (HR)", min_value=0.0, step=0.1, key="hr_input")
    n = st.number_input("Stability Number (N)", min_value=0.0, step=1.0, key="n_input")
    predict_button = st.button("Predict Stability")

# Main Panel for Results
st.title("Stope Stability Prediction Results")
if predict_button:
    if all([hr, n]):
        # Get prediction and probability
        status, probability = predict_stability(hr, n)
        
        # Display prediction result
        st.success(f"Predicted Status: **{status}**")
        st.info(f"Probability of Unstable Status: **{probability:.2%}**")
        
        # Generate isoprobability contour
        st.subheader("Isoprobability Contour")
        contour_plot = generate_isoprobability_contour(hr, n)
        st.pyplot(contour_plot)
        
        # Generate report using Gemini API
        st.subheader("Detailed Report")
        report = generate_report(status, probability)
        if report:
            st.markdown(report)
    else:
        st.warning("Please enter all required parameters.")
else:
    st.info("Enter parameters in the sidebar and click 'Predict Stability' to see results.")