import pandas as pd
import numpy as np
import os
import streamlit as st
import re
from datetime import datetime

def create_directory_if_not_exists(directory):
    """
    Create a directory if it doesn't exist
    
    Parameters:
    -----------
    directory : str
        Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def sanitize_string(s):
    """
    Remove special characters and make string suitable for filenames
    
    Parameters:
    -----------
    s : str
        Input string
        
    Returns:
    --------
    str
        Sanitized string
    """
    # Replace spaces and special characters with underscores
    return re.sub(r'[^a-zA-Z0-9]', '_', s)

def format_number(number, precision=2):
    """
    Format number with comma as thousand separator and specified precision
    
    Parameters:
    -----------
    number : float or int
        Number to format
    precision : int
        Number of decimal places
        
    Returns:
    --------
    str
        Formatted number
    """
    if pd.isna(number):
        return "N/A"
    elif isinstance(number, (int, float)):
        if number == int(number):
            return f"{int(number):,}"
        else:
            return f"{number:,.{precision}f}"
    else:
        return str(number)

def calculate_growth_rate(start_value, end_value, periods=1):
    """
    Calculate compound annual growth rate
    
    Parameters:
    -----------
    start_value : float
        Starting value
    end_value : float
        Ending value
    periods : int
        Number of periods
        
    Returns:
    --------
    float
        Growth rate as a percentage
    """
    if start_value <= 0 or end_value <= 0:
        return None
    
    return ((end_value / start_value) ** (1 / periods) - 1) * 100

def aggregate_by_categories(df, group_columns, value_columns, agg_func='sum'):
    """
    Aggregate data by specified categories
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    group_columns : list
        Columns to group by
    value_columns : list
        Columns to aggregate
    agg_func : str or dict
        Aggregation function(s)
        
    Returns:
    --------
    pd.DataFrame
        Aggregated dataframe
    """
    # Check if all columns exist
    for col in group_columns + value_columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
    
    # Apply aggregation
    if isinstance(agg_func, str):
        agg_func = {col: agg_func for col in value_columns}
    
    return df.groupby(group_columns).agg(agg_func).reset_index()

def get_percentile_rank(series, value):
    """
    Get percentile rank of a value in a series
    
    Parameters:
    -----------
    series : pd.Series
        Series of values
    value : float
        Value to find percentile for
        
    Returns:
    --------
    float
        Percentile rank (0-100)
    """
    return stats.percentileofscore(series.dropna(), value)

def streamlit_download_button(object_to_download, download_filename, button_text):
    """
    Generate a download button for any object
    
    Parameters:
    -----------
    object_to_download : pd.DataFrame, dict, str, etc.
        Object to download
    download_filename : str
        File name for download
    button_text : str
        Text to display on button
    """
    if isinstance(object_to_download, pd.DataFrame):
        # DataFrame -> CSV
        object_to_download = object_to_download.to_csv(index=False)
        mime = "text/csv"
        extension = "csv"
    elif isinstance(object_to_download, dict):
        # Dict -> JSON
        object_to_download = json.dumps(object_to_download)
        mime = "application/json"
        extension = "json"
    else:
        # String -> TXT
        mime = "text/plain"
        extension = "txt"
    
    # Add extension if not present
    if not download_filename.endswith(f".{extension}"):
        download_filename = f"{download_filename}.{extension}"
    
    st.download_button(
        label=button_text,
        data=object_to_download,
        file_name=download_filename,
        mime=mime
    )

def generate_timestamp():
    """
    Generate a timestamp string
    
    Returns:
    --------
    str
        Current timestamp formatted as YYYYMMDD_HHMMSS
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def color_scale(val, min_val, max_val, reverse=False):
    """
    Generate a color scale based on value (for styling dataframes)
    
    Parameters:
    -----------
    val : float
        Value to generate color for
    min_val : float
        Minimum value in range
    max_val : float
        Maximum value in range
    reverse : bool
        If True, reverse color scale (higher values get cooler colors)
        
    Returns:
    --------
    str
        CSS background-color property
    """
    if pd.isna(val):
        return "background-color: #f8f9fa"
    
    # Normalize value to 0-1 range
    normalized = (val - min_val) / (max_val - min_val) if max_val > min_val else 0.5
    
    if reverse:
        normalized = 1 - normalized
    
    # Generate color from blue (cool) to red (hot)
    r = int(255 * normalized)
    b = int(255 * (1 - normalized))
    g = int(100 + 100 * (1 - abs(normalized - 0.5) * 2))  # Peaks at 200 when normalized = 0.5
    
    return f"background-color: rgba({r}, {g}, {b}, 0.2)"

def get_data_summary(df):
    """
    Generate a comprehensive data summary
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Summary statistics and information
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": df.isna().sum().to_dict(),
        "missing_percentage": (df.isna().sum() / len(df) * 100).to_dict(),
        "dtypes": df.dtypes.astype(str).to_dict()
    }
    
    # Add numeric column statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()
    
    # Add categorical column info
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if cat_cols:
        summary["categorical_counts"] = {col: df[col].value_counts().to_dict() for col in cat_cols}
        summary["categorical_unique"] = {col: df[col].nunique() for col in cat_cols}
        
    return summary

def get_food_balance_indicators(df):
    """
    Calculate food balance sheet indicators
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with food balance indicators by country
    """
    if not all(col in df.columns for col in ['Country', 'Production', 'Import Quantity', 'Export Quantity', 'Total Supply']):
        return pd.DataFrame()
    
    # Group by country
    country_df = df.groupby('Country').agg({
        'Production': 'sum',
        'Import Quantity': 'sum',
        'Export Quantity': 'sum',
        'Total Supply': 'sum'
    }).reset_index()
    
    # Calculate indicators
    country_df['Self_Sufficiency_Ratio'] = country_df['Production'] / country_df['Total Supply']
    country_df['Import_Dependency_Ratio'] = country_df['Import Quantity'] / country_df['Total Supply']
    country_df['Export_Ratio'] = country_df['Export Quantity'] / country_df['Production']
    country_df['Trade_Balance'] = country_df['Export Quantity'] - country_df['Import Quantity']
    
    return country_df

def create_date_features(df, date_column):
    """
    Create date-related features from a date column
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_column : str
        Date column name
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional date features
    """
    if date_column not in df.columns:
        return df
    
    result_df = df.copy()
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(result_df[date_column]):
        result_df[date_column] = pd.to_datetime(result_df[date_column], errors='coerce')
    
    # Extract date components
    result_df[f'{date_column}_year'] = result_df[date_column].dt.year
    result_df[f'{date_column}_month'] = result_df[date_column].dt.month
    result_df[f'{date_column}_quarter'] = result_df[date_column].dt.quarter
    
    return result_df