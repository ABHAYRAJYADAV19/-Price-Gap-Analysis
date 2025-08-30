"""
This module contains all functions for customer segmentation,
including both simple RFM and advanced K-Means clustering.
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict


# Import shared resources
from utilities import COLOR_PALETTE

# --- RFM & K-MEANS ANALYSIS FUNCTIONS ---

def calculate_rfm_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates Recency, Frequency, and Monetary values."""
    analysis_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda date: (analysis_date - date.max()).days,
        'Invoice': 'nunique',
        'Revenue': 'sum'
    })
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    return rfm

def segment_customers(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns RFM scores and segments customers."""
    rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rf_score_str = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str)
    segment_map = {
        r'[1-2][1-2]': 'Sleeping', r'[1-2][3-4]': 'At Risk', r'[1-2]5': "Can't Lose Them",
        r'3[1-2]': 'Fading', r'33': 'Need Attention', r'[3-4][4-5]': 'Loyal Customers',
        r'41': 'Promising', r'51': 'New Customers', r'[4-5][2-3]': 'Potential Loyalists',
        r'5[4-5]': 'Champions'
    }
    rfm_df['Segment'] = rf_score_str.replace(segment_map, regex=True)
    return rfm_df

def assign_business_actions(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Assigns suggested business actions to each segment."""
    business_actions = {
        "Sleeping": "Try to win them back with special offers or reminders.",
        "Loyal Customers": "Reward them with exclusive deals and loyalty points.",
        "Champions": "Treat them like VIPs! Offer early access and special gifts.",
        "At Risk": "Send personalized emails and discounts to bring them back.",
        "Potential Loyalists": "Encourage them to join a membership or loyalty program.",
        "Fading": "Send 'we miss you' emails with a special offer.",
        "Need Attention": "Ask for their feedback and suggest popular products.",
        "Can't Lose Them": "Offer a great deal to keep them from leaving.",
        "Promising": "Suggest other products they might like.",
        "New Customers": "Give them a great first experience and a welcome discount."
    }
    rfm_df["Action"] = rfm_df["Segment"].map(business_actions)
    return rfm_df

def plot_rfm_distribution(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart showing the distribution of customers across RFM segments."""
    segment_counts = rfm_df['Segment'].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig = px.bar(
        segment_counts, 
        x='Count', 
        y='Segment', 
        orientation='h',
        title='<b>How many customers are in each group?</b>',
        labels={'Count': 'Number of Customers', 'Segment': 'Customer Group'},
        color_discrete_sequence=[COLOR_PALETTE['primary']]
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_rfm_sales_by_segment(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates a bar chart showing total sales by RFM segment."""
    segment_sales = rfm_df.groupby('Segment')['Monetary'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        segment_sales,
        x='Segment',
        y='Monetary',
        title='<b>Which customer groups generate the most sales?</b>',
        labels={'Monetary': 'Total Sales', 'Segment': 'Customer Group'},
        color_discrete_sequence=[COLOR_PALETTE['success']]
    )
    return fig

def plot_rfm_pie_charts(rfm_df: pd.DataFrame) -> go.Figure:
    """Creates pie charts for customer count and revenue share by RFM segment."""
    summary = rfm_df.groupby('Segment').agg(
        Customer_Count=('Recency', 'count'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()
    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=("Share of Total Customers", "Share of Total Sales")
    )
    fig.add_trace(go.Pie(
        labels=summary['Segment'], 
        values=summary['Customer_Count'], 
        name="Customers"
    ), 1, 1)
    fig.add_trace(go.Pie(
        labels=summary['Segment'], 
        values=summary['Total_Revenue'], 
        name="Sales"
    ), 1, 2)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="<b>How big and valuable is each customer group?</b>",
        legend_orientation="h"
    )
    return fig

def generate_business_summary(rfm_df: pd.DataFrame) -> pd.DataFrame:
    """Generates a summary table for each segment."""
    summary = rfm_df.groupby('Segment').agg(
        Customer_Count=('Recency', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Revenue=('Monetary', 'mean')
    ).round(2)
    total_customers = summary['Customer_Count'].sum()
    total_revenue = summary['Total_Revenue'].sum()
    summary['%_of_Customers'] = ((summary['Customer_Count'] / total_customers) * 100).round(2)
    summary['%_of_Revenue'] = ((summary['Total_Revenue'] / total_revenue) * 100).round(2)
    return summary.sort_values(by='Total_Revenue', ascending=False).reset_index()

def display_rfm_insights(rfm_df: pd.DataFrame):
    """Generates a text summary of insights from the RFM analysis."""
    st.header("💡 Insights from Simple Customer Groups")
    summary = rfm_df.groupby('Segment')['Monetary'].agg(['count', 'sum']).sort_values(by='sum', ascending=False)
    total_revenue = rfm_df['Monetary'].sum()
    total_customers = len(rfm_df)
    
    st.subheader(f"Your Most Valuable Group: **{summary.index[0]}**")
    st.markdown(f"""
    - The **'{summary.index[0]}'** group is your business's powerhouse.
    - Although they make up only **{summary.iloc[0]['count'] / total_customers:.1%}** of your customers, they generate **${summary.iloc[0]['sum']:,.0f}**, which is **{summary.iloc[0]['sum'] / total_revenue:.1%}** of your total sales!
    - **Action:** These are your VIPs. Focus your efforts on keeping them happy with exclusive rewards and personalized attention.
    """)
    
    st.subheader(f"Group to Re-engage: **{summary.index[-1]}**")
    st.markdown(f"""
    - The **'{summary.index[-1]}'** group represents customers who haven't purchased in a while and may be at risk of leaving.
    - **Action:** Launch a targeted 'we miss you' campaign with a special offer to win them back before they're gone for good.
    """)

def find_optimal_clusters(df: pd.DataFrame) -> go.Figure:
    """Calculates and plots the elbow curve to find the optimal number of clusters."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)
    
    wcss = []
    k_range = range(1, 11)
    for i in k_range:
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(rfm_scaled)
        wcss.append(kmeans.inertia_)
    
    fig = px.line(x=list(k_range), y=wcss, markers=True,
                  title='Finding the Best Number of Groups (Elbow Method)',
                  labels={'x': 'Number of Groups', 'y': 'WCSS (A measure of grouping quality)'},
                  color_discrete_sequence=[COLOR_PALETTE['primary']])
    return fig

def perform_kmeans_clustering(df: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Performs K-Means clustering on the RFM data."""
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)
    
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    clusters = kmeans.fit_predict(rfm_scaled)
    
    df_clustered = df.copy()
    df_clustered['Cluster'] = clusters
    return df_clustered

def get_cluster_names(df_clustered: pd.DataFrame) -> Dict:
    """Assigns descriptive names to K-Means clusters."""
    cluster_summary = df_clustered.groupby('Cluster').agg({
        'Recency': 'mean', 'Frequency': 'mean', 'Monetary': 'mean'
    })
    
    cluster_names = {}
    sorted_monetary = cluster_summary['Monetary'].sort_values().index
    sorted_recency = cluster_summary['Recency'].sort_values(ascending=False).index
    
    if len(sorted_monetary) > 0:
        cluster_names[sorted_monetary[-1]] = "High-Value Champions"
    if len(sorted_monetary) > 1:
        cluster_names[sorted_monetary[0]] = "Low-Spenders / New"
    if len(sorted_recency) > 0:
        if sorted_recency[0] not in cluster_names:
            cluster_names[sorted_recency[0]] = "At-Risk / Lapsed"
    
    remaining_clusters = [c for c in cluster_summary.index if c not in cluster_names.keys()]
    if len(remaining_clusters) > 0:
        sorted_freq_remaining = cluster_summary.loc[remaining_clusters]['Frequency'].sort_values(ascending=False).index
        if len(sorted_freq_remaining) > 0:
            cluster_names[sorted_freq_remaining[0]] = "Potential Loyalists"
        for i, cid in enumerate(sorted_freq_remaining[1:]):
             cluster_names[cid] = f"Standard Group {i+1}"
    
    return cluster_names

def generate_kmeans_summary_table(df_clustered: pd.DataFrame):
    """Generates a detailed summary table for K-Means clusters."""
    summary = df_clustered.groupby('Cluster_Name').agg(
        Customer_Count=('Recency', 'count'),
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Total_Revenue=('Monetary', 'sum'),
        Avg_Revenue=('Monetary', 'mean')
    ).round(2).reset_index()
    
    total_customers = summary['Customer_Count'].sum()
    total_revenue = summary['Total_Revenue'].sum()
    
    summary['% of Customers'] = (summary['Customer_Count'] / total_customers * 100).round(2)
    summary['% of Revenue'] = (summary['Total_Revenue'] / total_revenue * 100).round(2)
    
    return summary.sort_values(by='Total_Revenue', ascending=False)

def plot_kmeans_sales_by_segment(df_clustered: pd.DataFrame) -> go.Figure:
    """Creates a bar chart showing total sales by K-Means segment."""
    segment_sales = df_clustered.groupby('Cluster_Name')['Monetary'].sum().sort_values(ascending=False).reset_index()
    fig = px.bar(
        segment_sales,
        x='Cluster_Name',
        y='Monetary',
        title='<b>Which smart groups generate the most sales?</b>',
        labels={'Monetary': 'Total Sales', 'Cluster_Name': 'Customer Group'},
        color_discrete_sequence=[COLOR_PALETTE['success']]
    )
    return fig

def plot_kmeans_pie_charts(df_clustered: pd.DataFrame) -> go.Figure:
    """Creates pie charts for customer count and revenue share by cluster."""
    summary = df_clustered.groupby('Cluster_Name').agg(
        Customer_Count=('Recency', 'count'),
        Total_Revenue=('Monetary', 'sum')
    ).reset_index()

    fig = make_subplots(
        rows=1, cols=2, 
        specs=[[{'type':'domain'}, {'type':'domain'}]],
        subplot_titles=("Share of Total Customers", "Share of Total Sales")
    )

    fig.add_trace(go.Pie(
        labels=summary['Cluster_Name'], 
        values=summary['Customer_Count'], 
        name="Customers"
    ), 1, 1)

    fig.add_trace(go.Pie(
        labels=summary['Cluster_Name'], 
        values=summary['Total_Revenue'], 
        name="Sales"
    ), 1, 2)

    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(
        title_text="<b>How big and valuable is each customer group?</b>",
        legend_orientation="h"
    )
    return fig

def plot_kmeans_bar_charts(df_clustered: pd.DataFrame) -> go.Figure:
    """Creates bar charts comparing the average R, F, and M for each cluster."""
    summary = df_clustered.groupby('Cluster_Name').agg(
        Avg_Recency=('Recency', 'mean'),
        Avg_Frequency=('Frequency', 'mean'),
        Avg_Monetary=('Monetary', 'mean')
    ).reset_index()

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            "Days Since Last Purchase<br>(Lower is Better)", 
            "Number of Purchases<br>(Higher is Better)", 
            "Amount Spent<br>(Higher is Better)"
        )
    )

    fig.add_trace(go.Bar(
        x=summary['Cluster_Name'], y=summary['Avg_Recency'], name='Avg Recency', marker_color=COLOR_PALETTE['primary']
    ), 1, 1)

    fig.add_trace(go.Bar(
        x=summary['Cluster_Name'], y=summary['Avg_Frequency'], name='Avg Frequency', marker_color=COLOR_PALETTE['secondary']
    ), 1, 2)
    
    fig.add_trace(go.Bar(
        x=summary['Cluster_Name'], y=summary['Avg_Monetary'], name='Avg Monetary', marker_color=COLOR_PALETTE['success']
    ), 1, 3)

    fig.update_layout(
        title_text="<b>What are the buying habits of each group?</b>",
        showlegend=False,
        barmode='group'
    )
    return fig


def display_kmeans_business_insights(df: pd.DataFrame, cluster_names: Dict):
    """Generates and displays business insights for K-Means clusters."""
    st.header("📊 Insights from Your Smart Customer Groups")
    cluster_summary = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Cluster': 'count'
    }).rename(columns={'Cluster': 'Customer_Count'}).round(2)

    st.write("#### Meet Your Customer Groups:")
    for i, row in cluster_summary.iterrows():
        cluster_name = cluster_names.get(i, f"Group {i}")
        st.subheader(f"👤 {cluster_name}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Days Since Last Purchase", f"{row['Recency']:.1f}")
        col2.metric("Avg. Number of Purchases", f"{row['Frequency']:.1f}")
        col3.metric("Avg. Amount Spent", f"${row['Monetary']:,.2f}")

        interpretation = ""
        strategy = ""
        if "Champions" in cluster_name:
            interpretation = "These are your best customers: they buy often, spend the most, and have bought recently."
            strategy = "Engage with VIP treatment, early access to new products, and loyalty rewards. Turn them into brand ambassadors."
        elif "Low-Spenders" in cluster_name:
            interpretation = "This group consists of new or infrequent customers with low spending."
            strategy = "Help them grow! Send welcome offers, product recommendations, and deals to encourage a second purchase."
        elif "At-Risk" in cluster_name or "Lapsed" in cluster_name:
            interpretation = "These customers used to be valuable but haven't purchased in a long time. They might not come back."
            strategy = "Launch a special campaign to win them back. Offer a great discount or show them what's new to get their attention."
        elif "Potential" in cluster_name:
             interpretation = "This is a group of mid-tier customers who are starting to buy more often."
             strategy = "Encourage higher spending and frequency through bundling, cross-selling, and multi-buy offers."
        else:
            interpretation = "This is a group of your typical, everyday customers."
            strategy = "Keep them engaged with regular newsletters and promotions on popular products."
        
        st.markdown(f"**Who they are:** {interpretation}")
        st.markdown(f"**How to engage them:** {strategy}")
        st.markdown("---")
