import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os
import google.generativeai as genai

# Load the trained model, scaler, and label encoder
model = joblib.load('stacked_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

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
def generate_isoprobability_contour():
    hr_min, hr_max = 0, 10  
    n_min, n_max = 0, 50    
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
    # Set up Gemini API key
    # api_key = os.getenv("AIzaSyCStM_MT_1GdnIQ2qTGi0sHgR5VVVlXQLE")  # Replace with your actual API key
    
    # Configure the Gemini API
    genai.configure(api_key="AIzaSyCtGH81_C-37XjGodNR4SHpmfgUz3LelRM")
    
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
st.title("Stope Stability Prediction")
st.write("Enter the parameters to predict stope stability.")

# Input widgets
hr = st.number_input("HR (Hydraulic Radius)", min_value=0.0, step=0.1)
n = st.number_input("Stability Number", min_value=0.0, step=1.0)

# Predict on button click
if st.button("Predict Stability"):
    if all([hr, n]):
        # Get prediction and probability
        status, probability = predict_stability(hr, n)
        
        # Display prediction result
        st.success(f"Predicted Status: **{status}**")
        st.info(f"Probability of Unstable Status: **{probability:.2%}**")
        
        # Generate isoprobability contour
        st.subheader("Isoprobability Contour")
        contour_plot = generate_isoprobability_contour()
        st.pyplot(contour_plot)
        
        # Generate report using Gemini API
        st.subheader("Detailed Report")
        report = generate_report(status, probability)
        if report:
            st.markdown(report)
    else:
        st.warning("Please enter all required parameters.")