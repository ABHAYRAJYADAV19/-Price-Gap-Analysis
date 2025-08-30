"""
This module contains all functions related to data loading,
cleaning, and preprocessing for the sales dashboard.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Any

# --- DATA LOADING ---
@st.cache_data
def load_data(uploaded_file: Any) -> Optional[pd.DataFrame]:
    """
    Loads data from a file uploaded via Streamlit.
    Supports CSV and Excel formats.
    """
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='latin1')
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# --- DATA CLEANING & PREPROCESSING FUNCTIONS ---

def handle_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """Removes rows with negative Quantity or zero Price."""
    return df[(df['Quantity'] > 0) & (df['Price'] > 0)].copy()

def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Removes outliers from specified columns using the IQR method."""
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handles missing values for 'Customer ID' and 'Description'."""
    df_imputed = df.copy()
    df_imputed['Customer ID'].fillna('Unknown', inplace=True)
    stockcode_map = df_imputed.dropna(subset=['Description']).set_index('StockCode')['Description'].to_dict()
    df_imputed['Description'] = df_imputed['Description'].fillna(df_imputed['StockCode'].map(stockcode_map))
    df_imputed['Description'].fillna('Unknown', inplace=True)
    return df_imputed

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates new features like Revenue and time-based attributes."""
    df_featured = df.copy()
    df_featured['Revenue'] = df_featured['Quantity'] * df_featured['Price']
    df_featured['InvoiceDate'] = pd.to_datetime(df_featured['InvoiceDate'])
    df_featured['InvoiceYearMonth'] = df_featured['InvoiceDate'].dt.strftime('%Y-%m')
    df_featured['InvoiceWeekday'] = df_featured['InvoiceDate'].dt.day_name()
    df_featured['InvoiceHour'] = df_featured['InvoiceDate'].dt.hour
    return df_featured

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates the entire data preprocessing workflow."""
    with st.spinner("Getting your data ready for analysis..."):
        st.write("Step 1: Removing returns and invalid entries...")
        df = handle_negative_values(df)
        st.write("Step 2: Removing extreme, unusual values...")
        df = remove_outliers_iqr(df, ['Quantity', 'Price'])
        st.write("Step 3: Filling in missing information...")
        df = impute_missing_values(df)
        st.write("Step 4: Creating new data points for analysis...")
        df = engineer_features(df)
    return df
