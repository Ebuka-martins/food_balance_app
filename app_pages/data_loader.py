import pandas as pd
import numpy as np
import os

def load_and_preprocess_data(file_path="data/food_balance_sheet_europe.csv"):
    """
    Load and preprocess the Food Balance Sheet dataset
    
    Parameters:
    -----------
    file_path : str
        Path to the dataset
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed dataframe
    """
    try:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        # Remove duplicate rows
        df = df.drop_duplicates()
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Convert data types if needed
        df = convert_data_types(df)
        
        # Create additional features
        df = create_features(df)
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(df):
    """
    Handle missing values in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with handled missing values
    """
    # Check for missing values
    print(f"Missing values before handling: {df.isna().sum().sum()}")
    
    # For numeric columns, fill with mean or median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    print(f"Missing values after handling: {df.isna().sum().sum()}")
    
    return df

def convert_data_types(df):
    """
    Convert data types for appropriate columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with converted data types
    """
    # Convert Year to int if it exists
    if 'Year' in df.columns:
        df['Year'] = df['Year'].astype(int)
    
    # Convert any date columns if necessary
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col])
    
    return df

def create_features(df):
    """
    Create additional features for analysis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with additional features
    """
    # Example: Calculate per capita values if population data exists
    if 'Population' in df.columns and 'Total Supply' in df.columns:
        df['Supply Per Capita'] = df['Total Supply'] / df['Population']
    
    # Example: Calculate year-over-year changes if time series data
    if 'Year' in df.columns:
        # Group by relevant dimensions and calculate YoY changes
        if 'Country' in df.columns and 'Item' in df.columns:
            for metric in df.select_dtypes(include=[np.number]).columns:
                if metric not in ['Year', 'Population']:
                    df[f'{metric}_YoY_Change'] = df.groupby(['Country', 'Item'])[metric].pct_change()
    
    # Example: Create categorical features based on continuous values
    if 'Production' in df.columns:
        df['Production_Category'] = pd.qcut(df['Production'].fillna(0), 4, labels=['Low', 'Medium', 'High', 'Very High'])
    
    return df

def get_country_list(df):
    """
    Get the list of countries in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of countries
    """
    if 'Country' in df.columns:
        return sorted(df['Country'].unique().tolist())
    return []

def get_item_list(df):
    """
    Get the list of food items in the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of food items
    """
    if 'Item' in df.columns:
        return sorted(df['Item'].unique().tolist())
    return []

def filter_data(df, countries=None, items=None, years=None):
    """
    Filter data based on selected countries, items, and years
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to filter by
    items : list
        List of food items to filter by
    years : list
        List of years to filter by
        
    Returns:
    --------
    pd.DataFrame
        Filtered dataframe
    """
    filtered_df = df.copy()
    
    if countries and 'Country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Country'].isin(countries)]
        
    if items and 'Item' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Item'].isin(items)]
        
    if years and 'Year' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['Year'].isin(years)]
        
    return filtered_df