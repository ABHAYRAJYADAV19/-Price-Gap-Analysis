"""
This module contains all functions for the Exploratory Data Analysis (EDA)
section of the sales dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import shared resources
from utilities import COLOR_PALETTE

def plot_monthly_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for monthly sales."""
    monthly_sales = df.groupby('InvoiceYearMonth')['Revenue'].sum().reset_index()
    fig = px.bar(
        monthly_sales, x='InvoiceYearMonth', y='Revenue',
        title=f'<b>{title_prefix}Monthly Sales Trend</b>',
        labels={'InvoiceYearMonth': 'Month', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_daily_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for daily sales."""
    daily_sales = df.groupby('InvoiceWeekday')['Revenue'].sum().reset_index()
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_sales['InvoiceWeekday'] = pd.Categorical(daily_sales['InvoiceWeekday'], categories=weekday_order, ordered=True)
    daily_sales = daily_sales.sort_values('InvoiceWeekday')
    fig = px.bar(
        daily_sales, x='InvoiceWeekday', y='Revenue',
        title=f'<b>{title_prefix}Sales by Day of the Week</b>',
        labels={'InvoiceWeekday': 'Day of the Week', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['secondary']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_hourly_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a bar chart for hourly sales."""
    hourly_sales = df.groupby('InvoiceHour')['Revenue'].sum().reset_index()
    fig = px.bar(
        hourly_sales, x='InvoiceHour', y='Revenue',
        title=f'<b>{title_prefix}Sales by Hour of the Day</b>',
        labels={'InvoiceHour': 'Hour of the Day (24h format)', 'Revenue': 'Total Sales'},
        color_discrete_sequence=[COLOR_PALETTE['info']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def plot_geographical_sales(df: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Creates a choropleth map of sales by country."""
    country_sales = df.groupby('Country')['Revenue'].sum().reset_index()
    fig = px.choropleth(
        country_sales,
        locations="Country",
        locationmode='country names',
        color="Revenue",
        hover_name="Country",
        color_continuous_scale=px.colors.sequential.Blues,
        title=f'<b>{title_prefix}Geographical Sales Distribution</b>'
    )
    return fig

def plot_top_products(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Creates a horizontal bar chart of top N products by revenue."""
    top_products = df.groupby('Description')['Revenue'].sum().nlargest(top_n).sort_values(ascending=True)
    fig = px.bar(
        top_products,
        x=top_products.values,
        y=top_products.index,
        orientation='h',
        title=f'<b>What are your {top_n} best-selling products?</b>',
        labels={'x': 'Total Sales', 'y': 'Product'},
        color_discrete_sequence=[COLOR_PALETTE['success']]
    )
    return fig

def plot_worst_performers(df: pd.DataFrame, bottom_n: int = 10) -> go.Figure:
    """Creates horizontal bar charts of bottom N products and countries by revenue."""
    worst_products = df[df['Description'] != 'Unknown'].groupby('Description')['Revenue'].sum().nsmallest(bottom_n).sort_values(ascending=False)
    worst_countries = df.groupby('Country')['Revenue'].sum().nsmallest(bottom_n).sort_values(ascending=False)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"<b>Bottom {bottom_n} Products by Sales</b>", f"<b>Bottom {bottom_n} Countries by Sales</b>")
    )

    fig.add_trace(go.Bar(
        x=worst_products.values, y=worst_products.index, orientation='h',
        marker_color=COLOR_PALETTE['danger'], name='Worst Products'
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=worst_countries.values, y=worst_countries.index, orientation='h',
        marker_color=COLOR_PALETTE['info'], name='Worst Countries'
    ), row=1, col=2)
    
    fig.update_layout(title_text=f'<b>Which products and countries are underperforming?</b>', showlegend=False, template='plotly_white')
    return fig

def plot_new_vs_returning_customers(df: pd.DataFrame) -> go.Figure:
    """Creates a pie chart showing revenue from new vs. returning customers."""
    known_customers_df = df[df['Customer ID'] != 'Unknown']
    invoice_counts = known_customers_df.groupby('Customer ID')['Invoice'].nunique().reset_index()
    invoice_counts.columns = ['Customer ID', 'InvoiceCount']
    df_with_counts = pd.merge(known_customers_df, invoice_counts, on='Customer ID')
    df_with_counts['Customer_Category'] = np.where(df_with_counts['InvoiceCount'] == 1, 'New Customers', 'Returning Customers')
    revenue_summary = df_with_counts.groupby('Customer_Category')['Revenue'].sum().reset_index()
    fig = px.pie(
        revenue_summary, values='Revenue', names='Customer_Category',
        title='<b>Who brings in more money: New or Returning Customers?</b>',
        color_discrete_map={'Returning Customers': COLOR_PALETTE['primary'], 'New Customers': COLOR_PALETTE['light']}
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_average_order_value(df: pd.DataFrame) -> go.Figure:
    """Creates a line chart showing the trend of Average Order Value (AOV)."""
    order_revenue = df.groupby(['InvoiceYearMonth', 'Invoice'])['Revenue'].sum().reset_index()
    aov_trend = order_revenue.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()
    fig = px.line(
        aov_trend, x='InvoiceYearMonth', y='Revenue',
        title='<b>Are customers spending more per order over time?</b>',
        labels={'InvoiceYearMonth': 'Month', 'Revenue': 'Average Order Value ($)'},
        markers=True, color_discrete_sequence=[COLOR_PALETTE['danger']]
    )
    fig.update_layout(template='plotly_white')
    return fig

def analyze_market_basket(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Analyzes and returns a heatmap figure for market basket analysis."""
    top_products = df['Description'].value_counts().nlargest(top_n).index
    df_top = df[df['Description'].isin(top_products)]
    crosstab = pd.crosstab(df_top['Invoice'], df_top['Description'])
    crosstab[crosstab > 0] = 1
    co_occurrence_matrix = crosstab.T.dot(crosstab)
    np.fill_diagonal(co_occurrence_matrix.values, 0)
    fig = go.Figure(data=go.Heatmap(
        z=co_occurrence_matrix.values, x=co_occurrence_matrix.columns, y=co_occurrence_matrix.index, colorscale='Blues'))
    fig.update_layout(title=f'<b>Which of Your Top {top_n} Products are Bought Together?</b>',
                      xaxis_title="Products", yaxis_title="Products")
    return fig

def display_eda_insights(df: pd.DataFrame):
    """Generates and displays a summary of business insights from the EDA."""
    st.header("💡 Key Business Takeaways from the Overview")
    
    monthly_sales = df.groupby('InvoiceYearMonth')['Revenue'].sum()
    daily_sales = df.groupby('InvoiceWeekday')['Revenue'].sum()
    hourly_sales = df.groupby('InvoiceHour')['Revenue'].sum()
    top_product = df.groupby('Description')['Revenue'].sum().nlargest(1)
    top_country = df.groupby('Country')['Revenue'].sum().nlargest(1)
    
    st.subheader("Performance Highlights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Busiest Month", monthly_sales.idxmax(), f"${monthly_sales.max():,.0f} in sales")
    col2.metric("Busiest Day", daily_sales.idxmax())
    col3.metric("Busiest Hour", f"{hourly_sales.idxmax()}:00 - {hourly_sales.idxmax()+1}:00")
    
    st.subheader("Your Star Performers")
    col1, col2 = st.columns(2)
    col1.metric("Top Product", top_product.index[0], f"${top_product.values[0]:,.0f} in sales")
    col2.metric("Top Country", top_country.index[0], f"${top_country.values[0]:,.0f} in sales")
    
    st.subheader("Actionable Advice")
    st.markdown(f"""
    - **Seasonal Strategy:** Your sales peak in **{monthly_sales.idxmax()}**. Plan your marketing campaigns and stock levels to take full advantage of this period.
    - **Weekly Promotions:** **{daily_sales.idxmax()}** is your strongest sales day. Consider running special promotions or flash sales on slower days to even out weekly revenue.
    - **Focus on Winners:** Your top product is **'{top_product.index[0]}'**. Ensure it's always in stock and consider bundling it with less popular items to boost their sales.
    - **Market Focus:** **{top_country.index[0]}** is your biggest market. Think about targeted advertising or country-specific deals to further grow this key area.
    """)
