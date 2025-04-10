import pandas as pd
import numpy as np
from scipy import stats

def perform_eda(df):
    """
    Perform exploratory data analysis on the dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    dict
        Dictionary containing EDA results
    """
    eda_results = {}
    
    # Basic statistics
    eda_results['shape'] = df.shape
    eda_results['columns'] = df.columns.tolist()
    eda_results['data_types'] = df.dtypes.to_dict()
    eda_results['missing_values'] = df.isna().sum().to_dict()
    eda_results['summary_stats'] = df.describe().to_dict()
    
    # Count unique values for categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    eda_results['categorical_counts'] = {col: df[col].value_counts().to_dict() 
                                        for col in categorical_columns}
    
    return eda_results

def get_key_insights(df):
    """
    Generate key insights from the data
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of key insights as strings
    """
    insights = []
    
    # Example: Top producing countries
    if all(col in df.columns for col in ['Country', 'Production']):
        top_producers = df.groupby('Country')['Production'].sum().sort_values(ascending=False)
        insights.append(f"The top producing country is {top_producers.index[0]} with {top_producers.iloc[0]:.2f} units")
        insights.append(f"The bottom producing country is {top_producers.index[-1]} with {top_producers.iloc[-1]:.2f} units")
    
    # Example: Production trends over time
    if all(col in df.columns for col in ['Year', 'Production']):
        yearly_production = df.groupby('Year')['Production'].sum()
        if len(yearly_production) > 1:
            first_year = yearly_production.index[0]
            last_year = yearly_production.index[-1]
            change_pct = ((yearly_production.iloc[-1] - yearly_production.iloc[0]) / yearly_production.iloc[0]) * 100
            insights.append(f"From {first_year} to {last_year}, total production changed by {change_pct:.2f}%")
    
    # Example: Food item analysis
    if all(col in df.columns for col in ['Item', 'Production']):
        top_items = df.groupby('Item')['Production'].sum().sort_values(ascending=False)
        insights.append(f"The most produced food item is {top_items.index[0]} with {top_items.iloc[0]:.2f} units")
        insights.append(f"The least produced food item is {top_items.index[-1]} with {top_items.iloc[-1]:.2f} units")
    
    # Example: Supply and consumption patterns
    if all(col in df.columns for col in ['Country', 'Total Supply', 'Food']):
        df_agg = df.groupby('Country').agg({'Total Supply': 'sum', 'Food': 'sum'})
        df_agg['Food_Supply_Ratio'] = df_agg['Food'] / df_agg['Total Supply']
        highest_ratio = df_agg['Food_Supply_Ratio'].sort_values(ascending=False).index[0]
        insights.append(f"{highest_ratio} has the highest proportion of supply used for food consumption")
    
    # Example: Production sustainability
    if all(col in df.columns for col in ['Country', 'Production', 'Total Supply']):
        df_agg = df.groupby('Country').agg({'Production': 'sum', 'Total Supply': 'sum'})
        df_agg['Production_Supply_Ratio'] = df_agg['Production'] / df_agg['Total Supply']
        sustainable = df_agg[df_agg['Production_Supply_Ratio'] >= 1].index.tolist()
        if sustainable:
            insights.append(f"{len(sustainable)} countries are self-sustainable (production >= supply): {', '.join(sustainable[:3])}{' and more' if len(sustainable) > 3 else ''}")
        else:
            insights.append("No countries in the dataset are completely self-sustainable in food production")

    # Example: Correlation insights
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        high_corrs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corrs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        
        if high_corrs:
            strongest = max(high_corrs, key=lambda x: abs(x[2]))
            insights.append(f"Strong {' positive' if strongest[2] > 0 else ' negative'} correlation ({strongest[2]:.2f}) between {strongest[0]} and {strongest[1]}")
    
    return insights

def calculate_food_security_index(df):
    """
    Calculate a food security index for each country
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with food security index
    """
    if not all(col in df.columns for col in ['Country', 'Production', 'Total Supply', 'Import Quantity']):
        return pd.DataFrame()
    
    # Aggregate data by country
    country_data = df.groupby('Country').agg({
        'Production': 'sum',
        'Total Supply': 'sum',
        'Import Quantity': 'sum'
    }).reset_index()
    
    # Calculate metrics
    country_data['Self_Sufficiency'] = country_data['Production'] / country_data['Total Supply']
    country_data['Import_Dependency'] = country_data['Import Quantity'] / country_data['Total Supply']
    
    # Create composite index (simplified)
    country_data['Food_Security_Index'] = (
        0.6 * country_data['Self_Sufficiency'] +
        0.4 * (1 - country_data['Import_Dependency'])
    )
    
    # Normalize to 0-100 scale
    min_val = country_data['Food_Security_Index'].min()
    max_val = country_data['Food_Security_Index'].max()
    country_data['Food_Security_Score'] = 100 * (country_data['Food_Security_Index'] - min_val) / (max_val - min_val)
    
    # Rank countries
    country_data['Food_Security_Rank'] = country_data['Food_Security_Score'].rank(ascending=False)
    
    return country_data[['Country', 'Self_Sufficiency', 'Import_Dependency', 'Food_Security_Score', 'Food_Security_Rank']]

def detect_anomalies(df, column, method='zscore', threshold=3):
    """
    Detect anomalies in a specific column using different methods
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name to detect anomalies
    method : str
        Method to use ('zscore', 'iqr')
    threshold : float
        Threshold for outlier detection
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with anomalies
    """
    if column not in df.columns:
        return pd.DataFrame()
    
    result_df = df.copy()
    
    if method == 'zscore':
        # Z-score method
        z_scores = np.abs(stats.zscore(result_df[column].fillna(result_df[column].median())))
        result_df['is_anomaly'] = z_scores > threshold
    
    elif method == 'iqr':
        # IQR method
        Q1 = result_df[column].quantile(0.25)
        Q3 = result_df[column].quantile(0.75)
        IQR = Q3 - Q1
        result_df['is_anomaly'] = (result_df[column] < (Q1 - threshold * IQR)) | (result_df[column] > (Q3 + threshold * IQR))
    
    # Extract anomalies
    anomalies = result_df[result_df['is_anomaly']].drop(columns=['is_anomaly'])
    
    return anomalies

def time_series_analysis(df, countries, item, metric='Production'):
    """
    Perform time series analysis for selected countries and food item
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    countries : list
        List of countries to analyze
    item : str
        Food item to analyze
    metric : str
        Metric to analyze
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with time series data
    """
    if not all(col in df.columns for col in ['Country', 'Item', 'Year', metric]):
        return pd.DataFrame()
    
    # Filter data
    filtered_df = df[(df['Country'].isin(countries)) & (df['Item'] == item)]
    
    # Create time series dataframe
    ts_df = filtered_df.pivot_table(index='Year', columns='Country', values=metric)
    
    # Calculate year-over-year growth
    growth_df = ts_df.pct_change() * 100
    
    # Calculate rolling mean (3-year average)
    rolling_df = ts_df.rolling(window=3).mean()
    
    return {
        'raw_data': ts_df,
        'growth': growth_df,
        'rolling_avg': rolling_df
    }