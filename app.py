import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# -----------------------------------------------------------------------------
# 1. Page Configuration & Style
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Student Marks Predictor",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Performance Predictor")
st.markdown("Predict final marks based on habits, schedule, and course choice.")
st.write("---")

# -----------------------------------------------------------------------------
# 2. Data Loading & Cleaning Function
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_clean_data():
    # Load dataset
    df = pd.read_csv("student_study_hour_dataset.csv")
    
    # Clean Outliers/Typos based on dataset anomalies:
    # Caps Sleep_Hours at 24 and Study_Hours at 24 (fixing entries like 45)
    if 'Sleep_Hours' in df.columns:
        df.loc[df['Sleep_Hours'] > 24, 'Sleep_Hours'] = df['Sleep_Hours'] / 10
    if 'Study_Hours' in df.columns:
        df.loc[df['Study_Hours'] > 24, 'Study_Hours'] = df['Study_Hours'] / 10
        
    # Cap Marks at 100 (fixing entries like 500)
    if 'Marks' in df.columns:
        df = df[df['Marks'] <= 100]
        
    return df

# -----------------------------------------------------------------------------
# 3. Model Training Pipeline
# -----------------------------------------------------------------------------
@st.cache_resource
def train_model(df):
    # Features and Target
    X = df[['Course', 'Study_Hours', 'Sleep_Hours', 'Attendance_Percentage', 'Screen_Time_Hours']]
    y = df['Marks']
    
    # Handle missing values & encode categorical data ('Course')
    numeric_features = ['Study_Hours', 'Sleep_Hours', 'Attendance_Percentage', 'Screen_Time_Hours']
    categorical_features = ['Course']
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Create the final Linear Regression pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    # Train the model
    model_pipeline.fit(X, y)
    return model_pipeline, df['Course'].unique()

# Load data and train model
try:
    df = load_and_clean_data()
    model, unique_courses = train_model(df)
except Exception as e:
    st.error(f"Error loading dataset or training model: {e}")
    st.info("Make sure 'student_study_hour_dataset.csv' is in the same folder as this app.py file.")
    st.stop()

# -----------------------------------------------------------------------------
# 4. User Interface Input Form
# -----------------------------------------------------------------------------
st.subheader("Enter Student Details:")

col1, col2 = st.columns(2)

with col1:
    course = st.selectbox("Select Course", options=sorted(unique_courses))
    study_hours = st.slider("Daily Study Hours", min_value=0.0, max_value=16.0, value=4.0, step=0.5)
    sleep_hours = st.slider("Daily Sleep Hours", min_value=3.0, max_value=12.0, value=7.0, step=0.5)

with col2:
    attendance = st.slider("Attendance Percentage", min_value=0, max_value=100, value=85, step=1)
    screen_time = st.slider("Daily Screen Time (Hours)", min_value=0.0, max_value=16.0, value=3.5, step=0.5)

# -----------------------------------------------------------------------------
# 5. Prediction Logic
# -----------------------------------------------------------------------------
st.write("---")
if st.button("Predict Expected Marks", type="primary"):
    # Create dataframe for the input row
    input_data = pd.DataFrame([{
        'Course': course,
        'Study_Hours': study_hours,
        'Sleep_Hours': sleep_hours,
        'Attendance_Percentage': attendance,
        'Screen_Time_Hours': screen_time
    }])
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    # Bound the results realistically between 0 and 100
    final_score = max(0, min(100, prediction))
    
    # Display the result
    st.balloons()
    st.metric(label="Predicted Score", value=f"{final_score:.2f}%")
    
    # Contextual message
    if final_score >= 75:
        st.success("Great projection! Keep up these habits.")
    elif final_score >= 40:
        st.warning("Passable score, but balancing screen time or increasing study hours could boost performance.")
    else:
        st.error("Warning: Projected performance is low. Consider adjustments to study habits and attendance.")