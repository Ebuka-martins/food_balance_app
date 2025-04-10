import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_distributions(df, columns=None):
    """
    Plot distributions of numeric columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of columns to plot (if None, use all numeric columns)
    """
    if columns is None:
        # Select numeric columns with reasonable ranges (exclude years)
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = [col for col in columns if 'Year' not in col]
    
    # Limit to first 6 columns to avoid overwhelming visuals
    if len(columns) > 6:
        columns = columns[:6]
    
    for col in columns:
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def plot_time_series(df, countries=None, metric='Production'):
    """
    Plot time series data for selected countries
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to plot
    metric : str
        Metric to plot
    """
    if 'Year' not in df.columns or metric not in df.columns:
        st.error(f"Required columns not found: 'Year' or '{metric}'")
        return
    
    if countries is None and 'Country' in df.columns:
        # Use all countries, but limit to top 10 by metric
        top_countries = df.groupby('Country')[metric].sum().sort_values(ascending=False).head(10).index.tolist()
        df_filtered = df[df['Country'].isin(top_countries)]
    elif countries is not None and 'Country' in df.columns:
        df_filtered = df[df['Country'].isin(countries)]
    else:
        df_filtered = df
    
    # Group by year and country
    if 'Country' in df.columns:
        df_grouped = df_filtered.groupby(['Year', 'Country'])[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric, color='Country', 
                      title=f"{metric} Over Time by Country")
    else:
        df_grouped = df_filtered.groupby('Year')[metric].sum().reset_index()
        fig = px.line(df_grouped, x='Year', y=metric, 
                      title=f"{metric} Over Time")
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_comparisons(df, countries=None, metrics=None):
    """
    Plot country comparisons for selected metrics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to compare
    metrics : list
        List of metrics to compare
    """
    if 'Country' not in df.columns:
        st.error("Required column not found: 'Country'")
        return
    
    if countries is None:
        # Use top 10 countries by Production
        if 'Production' in df.columns:
            countries = df.groupby('Country')['Production'].sum().sort_values(ascending=False).head(10).index.tolist()
        else:
            countries = df['Country'].unique().tolist()[:10]
    
    if metrics is None:
        # Use common food balance sheet metrics
        potential_metrics = ['Production', 'Import Quantity', 'Export Quantity', 'Food', 'Feed', 'Losses']
        metrics = [m for m in potential_metrics if m in df.columns][:3]  # Limit to first 3
    
    # Filter data
    df_filtered = df[df['Country'].isin(countries)]
    
    # Aggregate by country
    df_agg = df_filtered.groupby('Country')[metrics].sum().reset_index()
    
    # Create grouped bar chart
    fig = px.bar(df_agg, x='Country', y=metrics, barmode='group',
                 title='Country Comparison by Key Metrics')
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def plot_food_balance(df, country):
    """
    Plot food balance composition for a selected country
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    country : str
        Country to plot
    """
    if not all(col in df.columns for col in ['Country', 'Food', 'Feed', 'Seed', 'Losses']):
        st.error("Required columns not found for food balance chart")
        return
    
    # Filter for the selected country
    df_country = df[df['Country'] == country]