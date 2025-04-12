import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_and_preprocess_data
from data_analysis import perform_eda, get_key_insights
from ml_models import train_model, predict
from visualizations import plot_distributions, plot_time_series, plot_comparisons

# Page configuration
st.set_page_config(
    page_title="Food Balance Sheet Analysis",
    page_icon="🍲",
    layout="wide"
)

# Title and introduction
st.title("Food Balance Sheet Analysis - Europe")
st.markdown("""
This application analyzes food balance sheet data for European countries,
provides insights into food production and consumption patterns, and offers
predictive analytics using machine learning models.
""")

# Load data
@st.cache_data
def load_data():
    return load_and_preprocess_data()

data = load_data()

# Sidebar for navigation and options
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Analysis", "Machine Learning", "Recommendations"])

# Data Overview page
if page == "Data Overview":
    st.header("Data Overview")
    st.write(data.head())
    st.write(f"Dataset shape: {data.shape}")
    
    # Basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())
    
    # Data distributions
    st.subheader("Data Distributions")
    plot_distributions(data)

# Data Analysis page
elif page == "Data Analysis":
    st.header("Data Analysis")
    
    # Country selection
    countries = sorted(data['Country'].unique())
    selected_countries = st.multiselect("Select countries to analyze", countries, default=countries[:3])
    
    if selected_countries:
        # Filter data
        filtered_data = data[data['Country'].isin(selected_countries)]
        
        # Time series analysis
        st.subheader("Food Production Over Time")
        plot_time_series(filtered_data)
        
        # Comparisons
        st.subheader("Country Comparisons")
        plot_comparisons(filtered_data)
        
        # Key insights
        st.subheader("Key Insights")
        insights = get_key_insights(filtered_data)
        for insight in insights:
            st.write(f"• {insight}")

# Machine Learning page
elif page == "Machine Learning":
    st.header("Machine Learning Models")
    
    # Feature and target selection
    features = st.multiselect("Select features", data.columns[2:], default=data.columns[2:5])
    target = st.selectbox("Select target variable", data.columns[2:])
    
    if features and target and target not in features:
        # Model selection
        model_type = st.selectbox("Select model type", ["Linear Regression", "Random Forest", "XGBoost"])
        
        # Train model button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, metrics = train_model(data, features, target, model_type)
                
                # Display metrics
                st.subheader("Model Performance")
                st.write(f"R² Score: {metrics['r2']:.4f}")
                st.write(f"MAE: {metrics['mae']:.4f}")
                st.write(f"RMSE: {metrics['rmse']:.4f}")
                
                # Feature importance
                st.subheader("Feature Importance")
                # Display feature importance plot
                
                # Predictions
                st.subheader("Predictions vs Actual")
                # Display predictions plot
    else:
        st.warning("Please select features and a different target variable")

# Recommendations page
elif page == "Recommendations":
    st.header("Data-Driven Recommendations")
    
    # Generate recommendations based on data analysis and ML insights
    st.subheader("Food Production Recommendations")
    st.write("1. Based on the analysis of trends in food consumption...")
    st.write("2. Countries like X could benefit from increasing production of...")
    st.write("3. Our predictive models suggest that...")
    
    st.subheader("Food Security Insights")
    st.write("1. The data indicates potential food security challenges in...")
    st.write("2. To address these challenges, we recommend...")
    
    # Additional custom recommendations

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Created as part of a Machine Learning project")