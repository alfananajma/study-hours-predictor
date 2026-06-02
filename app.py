import streamlit as st
import joblib
import numpy as np

# 1. Set page config with a classy dark/light responsive layout
st.set_page_config(
    page_title="EduPredict • Premium Grade Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. Inject Custom CSS for an ultra-modern, elegant aesthetic
st.markdown("""
    <style>
    /* Import premium font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main app background tweak */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    @media (prefers-color-scheme: dark) {
        .stApp {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        }
    }

    /* Custom elegant cards for results */
    .metric-card {
        background-color: padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 20px;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    .lr-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .knn-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    .card-title {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
        opacity: 0.9;
        margin-bottom: 10px;
    }

    .card-value {
        font-size: 42px;
        font-weight: 700;
        margin: 0;
    }
    
    /* Elegant Title Styling */
    .hero-title {
        font-size: 48px;
        font-weight: 700;
        background: linear-gradient(45deg, #4f46e5, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .hero-subtitle {
        font-size: 16px;
        color: #6b7280;
        margin-top: 5px;
        margin-bottom: 35px;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Load ML models safely
@st.cache_resource
def load_models():
    try:
        lr = joblib.load('linear_model.pkl')
        knn = joblib.load('knn_model.pkl')
        return lr, knn
    except FileNotFoundError:
        return None, None

linear_model, knn_model = load_models()

# If models are missing, show a beautiful error state
if linear_model is None or knn_model is None:
    st.error("📥 Model files ('linear_model.pkl' or 'knn_model.pkl') were not found.