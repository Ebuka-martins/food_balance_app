import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as XGBRegressor
import joblib
import os

def prepare_data_for_modeling(df, features, target, test_size=0.2, random_state=42):
    """
    Prepare data for modeling by splitting into train and test sets
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns
    target : str
        Target column
    test_size : float
        Test set proportion
    random_state : int
        Random state for reproducibility
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Check if all features and target exist in the dataframe
    for col in features + [target]:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")
    
    # Extract features and target
    X = df[features]
    y = df[target]
    
    # Identify numeric and categorical features
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(df, features, target, model_type='Linear Regression'):
    """
    Train a machine learning model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    features : list
        List of feature columns
    target : str
        Target column
    model_type : str
        Type of model to train
        
    Returns:
    --------
    tuple
        Trained model and performance metrics
    """
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_modeling(df, features, target)
        
        # Define model based on model_type
        if model_type == 'Linear Regression':
            model = LinearRegression()
            param_grid = {}
        elif model_type == 'Ridge':
            model = Ridge()
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'Lasso':
            model = Lasso()
            param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
        elif model_type == 'Random Forest':
            model = RandomForestRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif model_type == 'XGBoost':
            model = XGBRegressor.XGBRegressor(random_state=42)
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1]
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Use grid search if param_grid is not empty
        if param_grid:
            grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
            grid_search.fit(X_train, y_train)
            best_pipeline = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            best_pipeline = pipeline
            best_pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_pipeline.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
        }
        
        # Get feature importance if available
        feature_importance = get_feature_importance(best_pipeline, features, preprocessor)
        
        return best_pipeline, metrics, feature_importance
    
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None

def get_feature_importance(pipeline, features, preprocessor):
    """
    Get feature importance from the model if available
    
    Parameters:
    -----------
    pipeline : Pipeline
        Trained scikit-learn pipeline
    features : list
        List of feature columns
    preprocessor : ColumnTransformer
        Preprocessor used in the pipeline
        
    Returns:
    --------
    pd.DataFrame
        Feature importance dataframe
    """
    model = pipeline.named_steps['model']
    
    # Get feature names after preprocessing
    if hasattr(preprocessor, 'get_feature_names_out'):
        feature_names = preprocessor.get_feature_names_out()
    else:
        # Fallback for older scikit-learn versions
        feature_names = [f"feature_{i}" for i in range(len(features))]
    
    # Extract feature importance if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        return importance_df
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_
        if coefficients.ndim > 1:
            coefficients = coefficients[0]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).sort_values('Coefficient', ascending=False)
        return importance_df
    else:
        return pd.DataFrame()

def save_model(model, filename='models/food_balance_model.joblib'):
    """
    Save the trained model to a file
    
    Parameters:
    -----------
    model : Pipeline
        Trained scikit-learn pipeline
    filename : str
        Filename to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save the model
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename='models/food_balance_model.joblib'):
    """
    Load a trained model from a file
    
    Parameters:
    -----------
    filename : str
        Filename to load the model from
        
    Returns:
    --------
    Pipeline
        Loaded scikit-learn pipeline
    """
    try:
        model = joblib.load(filename)
        print(f"Model loaded from {filename}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict(model, X):
    """
    Make predictions using a trained model
    
    Parameters:
    -----------
    model : Pipeline
        Trained scikit-learn pipeline
    X : pd.DataFrame
        Input features
        
    Returns:
    --------
    np.ndarray
        Predictions
    """
    try:
        return model.predict(X)
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

def forecast_food_production(df, country, item, years_to_forecast=5):
    """
    Forecast food production for a specific country and item
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    country : str
        Country to forecast for
    item : str
        Food item to forecast for
    years_to_forecast : int
        Number of years to forecast
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with forecasted values
    """
    if not all(col in df.columns for col in ['Country', 'Item', 'Year', 'Production']):
        return pd.DataFrame()
    
    # Filter data for the specified country and item
    filtered_df = df[(df['Country'] == country) & (df['Item'] == item)]
    
    if filtered_df.empty:
        return pd.DataFrame()
    
    # Sort by year
    filtered_df = filtered_df.sort_values('Year')
    
    # Extract features and target
    X = filtered_df[['Year']]
    y = filtered_df['Production']
    
    # Train a simple model
    model = LinearRegression()
    model.fit(X, y)
    
    # Get the last year in the dataset
    last_year = X['Year'].max()
    
    # Create future years
    future_years = pd.DataFrame({'Year': range(last_year + 1, last_year + years_to_forecast + 1)})
    
    # Make predictions
    future_predictions = model.predict(future_years)
    
    # Create result dataframe
    result = pd.DataFrame({
        'Year': future_years['Year'],
        'Country': country,
        'Item': item,
        'Forecasted_Production': future_predictions
    })
    
    return result