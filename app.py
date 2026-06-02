import streamlit as st
import joblib
import numpy as np

# Set up page configurations
st.set_page_config(page_title="Study Hours to Marks Predictor", page_icon="🎓", layout="centered")

# App Header
st.title("🎓 Student Marks Predictor App")
st.write("Enter your daily study hours below to predict your final examination score using pre-trained Machine Learning models.")

# Load the pre-trained models
@st.cache_resource
def load_models():
    lr = joblib.load('linear_model.pkl')
    knn = joblib.load('knn_model.pkl')
    return lr, knn

try:
    linear_model, knn_model = load_models()
except Exception as e:
    st.error("Error loading model files. Please make sure 'linear_model.pkl' and 'knn_model.pkl' are present in the repository.")
    st.stop()

# Layout layout split into inputs
st.markdown("### 🛠️ Input Parameters")
study_hours = st.slider("Select Study Hours per Day:", min_value=1.0, max_value=12.0, value=5.0, step=0.1)

# Model Selection dropdown
model_choice = st.selectbox("Select Prediction Model:", ["Linear Regression", "K-Nearest Neighbors (KNN)"])

st.markdown("---")

# Prediction Trigger
if st.button("Predict Score 🎯", use_container_width=True):
    # Reshape input to a 2D array as required by scikit-learn
    input_data = np.array([[study_hours]])
    
    # Predict based on choice
    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_data)[0]
    else:
        prediction = knn_model.predict(input_data)[0]
        
    # Cap the results logically between 0% and 100%
    prediction = max(0.0, min(100.0, prediction))
    
    # Display Result
    st.balloons()
    st.success(f"📚 Estimated Score using **{model_choice}**: **{prediction:.2f}%**")