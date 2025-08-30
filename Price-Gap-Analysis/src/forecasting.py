"""
This module contains all functions for time-series forecasting
and dynamic pricing.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import holidays
from typing import Optional, Dict, Tuple

# Import shared resources
from utilities import COLOR_PALETTE

# --- MODEL ARCHITECTURES ---
class LSTMModel(nn.Module):
    """Defines the structure of the LSTM neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class GRUModel(nn.Module):
    """Defines the structure of the GRU neural network."""
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.4):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# --- FORECASTING HELPER FUNCTIONS ---

def prepare_and_engineer_features_forecast(
    df: pd.DataFrame,
    product_stock_code: str,
    competitor_df: Optional[pd.DataFrame] = None,
    customer_segment_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """Filters, prepares, merges external data, and engineers features for forecasting."""
    
    product_df = df[df['StockCode'] == product_stock_code].copy()
    if product_df.empty:
        st.error(f"Error: No data found for Product ID {product_stock_code}.")
        return None

    product_df.set_index('InvoiceDate', inplace=True)
    daily_sales_df = product_df.resample('D').agg(
        Quantity=('Quantity', 'sum'),
        our_price=('Price', 'mean')
    )
    daily_sales_df['our_price'].replace(0, np.nan, inplace=True)
    daily_sales_df['our_price'] = daily_sales_df['our_price'].ffill().bfill()
    daily_sales_df.fillna(0, inplace=True) # Fill Quantity NaNs for days with no sales

    if competitor_df is not None:
        try:
            competitor_df['Date'] = pd.to_datetime(competitor_df['Date'], errors='coerce')
            competitor_df.set_index('Date', inplace=True)
            daily_comp_prices = competitor_df.resample('D').mean()
            
            daily_sales_df = daily_sales_df.merge(daily_comp_prices, left_index=True, right_index=True, how='left', suffixes=('', '_comp'))

            if 'our_price_comp' in daily_sales_df.columns:
                daily_sales_df['our_price'] = daily_sales_df['our_price_comp'].fillna(daily_sales_df['our_price'])
                daily_sales_df.drop(columns=['our_price_comp'], inplace=True)
            
            price_cols = ['competitor_A', 'competitor_B', 'competitor_C']
            for col in price_cols:
                if col in daily_sales_df.columns:
                    daily_sales_df[col].ffill(inplace=True)
                    daily_sales_df[col].bfill(inplace=True)
        except Exception as e:
            st.warning(f"Could not use the competitor price data. Error: {e}")

    if customer_segment_df is not None:
        try:
            customer_segment_df['Date'] = pd.to_datetime(customer_segment_df['Date'], errors='coerce')
            daily_segment_sales = customer_segment_df.pivot_table(index='Date', columns='Segment', values='Quantity', aggfunc='sum')
            daily_segment_sales = daily_segment_sales.resample('D').sum()
            daily_sales_df = daily_sales_df.merge(daily_segment_sales, left_index=True, right_index=True, how='left')
            daily_sales_df[daily_segment_sales.columns] = daily_sales_df[daily_segment_sales.columns].fillna(0)
        except Exception as e:
            st.warning(f"Could not use the customer segment data. Skipping. Error: {e}")

    df_featured = daily_sales_df.copy()
    df_featured['day_of_week'] = df_featured.index.dayofweek
    df_featured['month'] = df_featured.index.month
    df_featured['week_of_year'] = df_featured.index.isocalendar().week.astype(int)
    
    for i in [1, 7, 14, 30]:
        df_featured[f'lag_{i}_days'] = df_featured['Quantity'].shift(i)
    
    for window in [7, 14, 30]:
        df_featured[f'rolling_mean_{window}_days'] = df_featured['Quantity'].shift(1).rolling(window=window).mean()
        df_featured[f'rolling_std_{window}_days'] = df_featured['Quantity'].shift(1).rolling(window=window).std()

    uk_holidays = holidays.UK(years=df_featured.index.year.unique())
    df_featured['is_holiday'] = df_featured.index.map(lambda x: 1 if x in uk_holidays else 0)
    
    df_featured.fillna(0, inplace=True)
    return df_featured

def scale_and_create_sequences(daily_df: pd.DataFrame, seq_length: int, forecast_horizon: int):
    """Scales data and creates sequences for model training."""
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(daily_df)
    target_col_idx = daily_df.columns.get_loc('Quantity')
    X, y = [], []
    for i in range(len(scaled_features) - seq_length - forecast_horizon + 1):
        X.append(scaled_features[i:i+seq_length])
        y.append(scaled_features[i+seq_length:i+seq_length+forecast_horizon, target_col_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), scaler, target_col_idx

def split_data_and_create_loaders(X: np.ndarray, y: np.ndarray, train_split_ratio: float, val_split_ratio: float, batch_size: int = 32):
    """Splits data and creates PyTorch DataLoaders."""
    total_samples = len(X)
    train_size = int(total_samples * train_split_ratio)
    val_size = int(total_samples * val_split_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, y_test, X_test

def train_model(train_loader: DataLoader, val_loader: DataLoader, model: nn.Module, training_params: Dict) -> nn.Module:
    """Generic model training function with early stopping."""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    best_val_loss = float('inf')
    patience_counter = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(training_params['num_epochs']):
        model.train()
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                val_loss += criterion(model(batch_X), batch_y).item()
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1

        progress_bar.progress((epoch + 1) / training_params['num_epochs'])
        status_text.text(f"Teaching the AI... Step {epoch+1} of {training_params['num_epochs']} | Learning Progress: {avg_val_loss:.5f}")

        if patience_counter >= training_params['patience']:
            st.info(f"AI has learned enough after {epoch+1} steps.")
            break
    
    progress_bar.empty()
    status_text.empty()
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def evaluate_model(model: nn.Module, test_loader: DataLoader, scaler: MinMaxScaler, y_test: np.ndarray, target_col_idx: int, num_features: int) -> Tuple[pd.DataFrame, Dict]:
    """Evaluates the model and returns predictions and metrics."""
    model.eval()
    all_predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            all_predictions.append(model(batch_X).cpu().numpy())
    predictions_scaled = np.concatenate(all_predictions)
    predictions_full = np.zeros((predictions_scaled.shape[0], num_features))
    predictions_full[:, target_col_idx] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(predictions_full)[:, target_col_idx]
    y_test_full = np.zeros((len(y_test), num_features))
    y_test_full[:, target_col_idx] = y_test.flatten()
    y_test_actual = scaler.inverse_transform(y_test_full)[:, target_col_idx]
    mae = mean_absolute_error(y_test_actual, predictions)
    rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
    metrics = {'MAE': mae, 'RMSE': rmse}
    results_df = pd.DataFrame({'Actual': y_test_actual, 'Predicted': predictions})
    return results_df, metrics

def generate_future_forecasts(model: nn.Module, daily_df: pd.DataFrame, scaler: MinMaxScaler, seq_length: int, target_col_idx: int, num_features: int, num_days: int) -> pd.DataFrame:
    """Generates forecasts for a specified number of future days."""
    model.eval()
    last_sequence_full = daily_df.tail(seq_length).values
    last_sequence_scaled = scaler.transform(last_sequence_full)
    current_sequence = torch.from_numpy(last_sequence_scaled).unsqueeze(0).float()
    future_predictions_scaled = []
    with torch.no_grad():
        for _ in range(num_days):
            prediction = model(current_sequence)
            future_predictions_scaled.append(prediction.item())
            new_row_scaled = current_sequence.numpy().squeeze()[-1].copy()
            new_row_scaled[target_col_idx] = prediction.item()
            new_sequence_np = np.vstack([current_sequence.numpy().squeeze()[1:], new_row_scaled])
            current_sequence = torch.from_numpy(new_sequence_np).unsqueeze(0).float()
    future_predictions_full = np.zeros((len(future_predictions_scaled), num_features))
    future_predictions_full[:, target_col_idx] = future_predictions_scaled
    future_predictions = scaler.inverse_transform(future_predictions_full)[:, target_col_idx]
    last_date = daily_df.index[-1]
    future_dates = pd.to_datetime([last_date + pd.DateOffset(days=i) for i in range(1, num_days + 1)])
    future_df = pd.DataFrame({'Date': future_dates, 'Future_Forecast': future_predictions})
    future_df.set_index('Date', inplace=True)
    return future_df

def plot_focused_forecast(future_df: pd.DataFrame, results_df: pd.DataFrame, product_stock_code: str) -> go.Figure:
    """Creates a forecast chart showing actual vs. predicted sales and the future forecast."""
    fig = go.Figure()

    # Actual Sales on Test Data
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Actual'], name='Actual Sales (Recent)',
        mode='lines', line=dict(color=COLOR_PALETTE['primary'], width=2)
    ))

    # Predicted Sales on Test Data
    fig.add_trace(go.Scatter(
        x=results_df.index, y=results_df['Predicted'], name='Predicted Sales (Recent)',
        mode='lines', line=dict(color=COLOR_PALETTE['danger'], dash='dash', width=2)
    ))

    # Future Forecast
    fig.add_trace(go.Scatter(
        x=future_df.index, y=future_df['Future_Forecast'], name='Future Forecast',
        mode='lines', line=dict(color=COLOR_PALETTE['success'], dash='dot', width=3)
    ))

    fig.update_layout(
        title=f'<b>Sales Forecast for Product {product_stock_code}</b>',
        template='plotly_white',
        yaxis_title='Items Sold per Day',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

def plot_forecast_breakdown(future_df: pd.DataFrame, daily_df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Creates pie and bar charts to break down the forecast."""
    # Pie Chart: Forecast by Day of the Week
    future_df['Weekday'] = future_df.index.day_name()
    weekday_sales = future_df.groupby('Weekday')['Future_Forecast'].sum().reset_index()
    fig_pie = px.pie(
        weekday_sales,
        values='Future_Forecast',
        names='Weekday',
        title='<b>Which days will be busiest next month?</b>',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    # Bar Chart: Past vs. Future Average Sales
    past_avg = daily_df['Quantity'].tail(30).mean()
    future_avg = future_df['Future_Forecast'].mean()
    comparison_df = pd.DataFrame({
        'Period': ['Past 30 Days', 'Next 30 Days (Predicted)'],
        'Average Daily Sales': [past_avg, future_avg]
    })
    fig_bar = px.bar(
        comparison_df,
        x='Period',
        y='Average Daily Sales',
        title='<b>Is growth predicted for next month?</b>',
        text_auto='.2s',
        color='Period',
        color_discrete_map={'Past 30 Days': COLOR_PALETTE['light'], 'Next 30 Days (Predicted)': COLOR_PALETTE['success']}
    )
    fig_bar.update_layout(showlegend=False)
    return fig_pie, fig_bar

def style_future_sales_table(future_df: pd.DataFrame):
    """Applies a color gradient to the future sales DataFrame."""
    styled_df = future_df[['Future_Forecast']].copy()
    styled_df['Future_Forecast'] = styled_df['Future_Forecast'].round(0).astype(int)
    return styled_df.style.background_gradient(cmap='Greens')

def display_forecast_insights(daily_df: pd.DataFrame, metrics: Dict, future_df: pd.DataFrame):
    """Generates and displays actionable business insights."""
    st.header("💡 Actionable Insights from the Prediction")

    st.subheader("1. How Accurate is the Prediction? (And what to do about it)")
    st.markdown(f"- The prediction is typically off by about **{metrics['MAE']:.2f} items** per day.")
    st.markdown(f"- **ACTION:** To be safe and avoid running out of stock, consider keeping at least **{metrics['MAE']:.0f} extra items** on hand. This is your 'safety stock'.")
    
    forecast_next_7_days = future_df['Future_Forecast'].iloc[:7].sum()
    st.markdown(f"- **Inventory Planning:** The prediction says you'll sell about **{forecast_next_7_days:.0f} items** in the next 7 days. Use this number to plan your next order!")

    st.subheader("2. When to Run Promotions & Schedule Staff")
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_of_week_sales = future_df.groupby(future_df.index.day_name())['Future_Forecast'].sum()
    if not day_of_week_sales.empty:
        best_day_name = day_of_week_sales.idxmax()
        worst_day_name = day_of_week_sales.idxmin()
        st.markdown(f"- **Predicted Weekly Pattern:** The forecast suggests your busiest day will be **{best_day_name}**, and your slowest will be **{worst_day_name}**.")
        st.markdown(f"- **ACTION:** Plan to have more staff and stock ready for **{best_day_name}s**. Consider running promotions on **{worst_day_name}s** to drive traffic and balance out the week.")
        
    last_30_days_avg = daily_df['Quantity'].tail(30).mean()
    next_30_days_avg = future_df['Future_Forecast'].mean()
    growth_pct = ((next_30_days_avg - last_30_days_avg) / last_30_days_avg) * 100 if last_30_days_avg > 0 else 0
    
    st.subheader("3. What's the Big Picture?")
    if growth_pct > 5:
        st.success(f"**Good News!** The model predicts sales will grow by about **{growth_pct:.1f}%** over the next 30 days. Get ready for more customers!")
    elif growth_pct < -5:
        st.warning(f"**Heads Up!** The model predicts sales might dip by **{abs(growth_pct):.1f}%** over the next 30 days. It might be a good time for a marketing push.")
    else:
        st.info("**Steady As She Goes:** The model predicts sales will remain stable for the next 30 days.")

# --- DYNAMIC PRICING FUNCTIONS ---

def recommend_optimal_price(model: nn.Module, daily_df: pd.DataFrame, scaler: MinMaxScaler, seq_length: int, target_col_idx: int, price_col_idx: int, num_features: int, num_days_to_simulate: int) -> Tuple[pd.Series, pd.DataFrame]:
    """Simulates different prices over a future period to find the one that maximizes total revenue."""
    model.eval()
    
    original_columns = daily_df.columns
    last_sequence_df = daily_df.tail(seq_length)
    
    current_price = last_sequence_df.iloc[-1, price_col_idx]
    if current_price == 0:
        st.warning("Current price is zero, cannot simulate price changes. Please check your data.")
        return None, None
        
    price_range = np.linspace(current_price * 0.8, current_price * 1.2, 20)
    
    results = []
    with torch.no_grad():
        for price in price_range:
            future_quantities = []
            temp_sequence_df = last_sequence_df.copy()
            
            temp_sequence_df.iloc[:, price_col_idx] = price
            
            scaled_sequence = scaler.transform(temp_sequence_df)
            current_sequence_tensor = torch.from_numpy(scaled_sequence).unsqueeze(0).float()

            for _ in range(num_days_to_simulate):
                predicted_quantity_scaled = model(current_sequence_tensor).item()
                
                dummy_df = pd.DataFrame(np.zeros((1, num_features)), columns=original_columns)
                dummy_df.iloc[0, target_col_idx] = predicted_quantity_scaled
                
                predicted_quantity = scaler.inverse_transform(dummy_df)[0, target_col_idx]
                predicted_quantity = max(0, predicted_quantity)
                future_quantities.append(predicted_quantity)

                new_row_scaled = current_sequence_tensor.numpy().squeeze()[-1].copy()
                new_row_scaled[target_col_idx] = predicted_quantity_scaled
                
                new_sequence_np = np.vstack([current_sequence_tensor.numpy().squeeze()[1:], new_row_scaled])
                current_sequence_tensor = torch.from_numpy(new_sequence_np).unsqueeze(0).float()

            total_predicted_quantity = sum(future_quantities)
            total_predicted_revenue = price * total_predicted_quantity
            results.append({'Price': price, 'Total_Predicted_Quantity': total_predicted_quantity, 'Total_Predicted_Revenue': total_predicted_revenue})

    results_df = pd.DataFrame(results)
    optimal_row = results_df.loc[results_df['Total_Predicted_Revenue'].idxmax()]
    return optimal_row, results_df

def plot_price_recommendation(results_df: pd.DataFrame, optimal_row: pd.Series, num_days_to_simulate: int) -> go.Figure:
    """Visualizes the price vs. revenue curve and highlights the optimal point."""
    fig = px.line(results_df, x='Price', y='Total_Predicted_Revenue',
                  title=f'<b>How Changing the Price Might Affect Your Total Sales</b>',
                  labels={'Price': 'Price ($)', 'Total_Predicted_Revenue': f'Predicted Total Sales in the next {num_days_to_simulate} Days ($)'},
                  markers=True, color_discrete_sequence=[COLOR_PALETTE['primary']])
    
    fig.add_trace(go.Scatter(
        x=[optimal_row['Price']],
        y=[optimal_row['Total_Predicted_Revenue']],
        mode='markers',
        marker=dict(color=COLOR_PALETTE['danger'], size=12, symbol='star'),
        name='Best Price'
    ))
    
    fig.add_annotation(
        x=optimal_row['Price'],
        y=optimal_row['Total_Predicted_Revenue'],
        text=f"Best Price: ${optimal_row['Price']:.2f}<br>Total Sales: ${optimal_row['Total_Predicted_Revenue']:.2f}",
        showarrow=True,
        arrowhead=1,
        ax=20,
        ay=-40
    )
    return fig

def display_pricing_insights(optimal_row: pd.Series, current_price: float, num_days_to_simulate: int, results_df: pd.DataFrame):
    """Generates and displays business insights for the pricing recommendation."""
    st.header("📈 Smart Pricing Insights")
    
    if optimal_row is None or results_df is None:
        return

    revenue_at_current_price_row = results_df.iloc[(results_df['Price']-current_price).abs().argsort()[:1]]
    if revenue_at_current_price_row.empty:
        st.warning("Could not calculate revenue uplift.")
        return
        
    revenue_at_current_price = revenue_at_current_price_row['Total_Predicted_Revenue'].values[0]
    potential_uplift = optimal_row['Total_Predicted_Revenue'] - revenue_at_current_price
    uplift_pct = (potential_uplift / revenue_at_current_price) * 100 if revenue_at_current_price > 0 else 0

    st.subheader("Recommendation Summary:")
    price_change_pct = ((optimal_row['Price'] - current_price) / current_price) * 100
    change_direction = "increase" if price_change_pct > 0 else "decrease"
    
    col1, col2 = st.columns(2)
    col1.metric(label="Recommended Best Price", value=f"${optimal_row['Price']:.2f}", delta=f"{price_change_pct:.2f}% vs. Current")
    col2.metric(label=f"Potential Extra Sales ({num_days_to_simulate} days)", value=f"${potential_uplift:,.2f}", help=f"An estimated increase of {uplift_pct:.2f}% compared to keeping the current price.")
    
    col3, col4 = st.columns(2)
    col3.metric(label=f"Expected Items Sold", value=f"{optimal_row['Total_Predicted_Quantity']:.0f} items")
    col4.metric(label=f"Expected Total Sales", value=f"${optimal_row['Total_Predicted_Revenue']:,.2f}")

    st.subheader("Your Action Plan:")
    st.markdown(f"""
    - **The Action:** The model suggests you **{change_direction}** the price by **{abs(price_change_pct):.1f}%**, from **${current_price:.2f}** to **${optimal_row['Price']:.2f}**.
    - **Why?** This new price is the 'sweet spot' that is predicted to make you the most money over the next **{num_days_to_simulate} days**, potentially increasing your sales by **${potential_uplift:,.2f}**.
    - **What this means:** If a price `{change_direction}` leads to higher total sales, it suggests customers are **{'not very sensitive' if change_direction == 'increase' else 'very sensitive'}** to price changes for this product right now.
    - **How to test it:** Try out the new price for a week or two to see if it works as predicted before making it permanent. Keep an eye on sales and customer feedback.
    """)

# --- MAIN FORECASTING PIPELINE ---

def run_forecasting_pipeline(
    model_type: str,
    df: pd.DataFrame,
    product_stock_code: str,
    competitor_df: Optional[pd.DataFrame] = None,
    customer_segment_df: Optional[pd.DataFrame] = None,
    future_forecast_days: int = 30,
    seq_length: int = 60,
    train_split_ratio: float = 0.7,
    val_split_ratio: float = 0.15,
    seed: int = 42
):
    """Orchestrates the complete time-series forecasting pipeline."""
    model_name = "Model 1" if model_type == "LSTM" else "Model 2"
    st.info(f"Starting Prediction with {model_name} for Product: {product_stock_code}")
    torch.manual_seed(seed); np.random.seed(seed); 
    
    daily_sales_df = prepare_and_engineer_features_forecast(df, product_stock_code, competitor_df, customer_segment_df)
    if daily_sales_df is None or daily_sales_df.empty:
        st.error("Prediction stopped because the data could not be prepared.")
        return

    X, y, scaler, target_col_idx = scale_and_create_sequences(daily_sales_df, seq_length, 1)
    train_loader, val_loader, test_loader, y_test, X_test = split_data_and_create_loaders(X, y, train_split_ratio, val_split_ratio)
    
    if len(X_test) == 0:
        st.error("There isn't enough data for this product to make a reliable prediction. Please try a different product or upload more data.")
        return

    model_params = {'input_size': X.shape[2], 'hidden_size': 128, 'num_layers': 2, 'output_size': 1}
    training_params = {'num_epochs': 50, 'learning_rate': 0.005, 'patience': 10}
    
    model = LSTMModel(**model_params) if model_type == 'LSTM' else GRUModel(**model_params)
    
    st.write(f"Teaching {model_name} to understand your sales data...")
    model = train_model(train_loader, val_loader, model, training_params)
    st.success(f"{model_name} is ready to make predictions!")

    results_df, metrics = evaluate_model(model, test_loader, scaler, y_test, target_col_idx, daily_sales_df.shape[1])
    results_df.index = daily_sales_df.index[-len(results_df):]

    future_df = generate_future_forecasts(model, daily_sales_df, scaler, seq_length, target_col_idx, daily_sales_df.shape[1], future_forecast_days)

    st.subheader(f"{model_name} Performance")
    st.metric("Average Prediction Error", f"± {metrics['MAE']:.2f} items / day")
    
    st.plotly_chart(plot_focused_forecast(future_df, results_df, product_stock_code), use_container_width=True)
    
    st.subheader("Forecast Breakdown")
    fig_pie, fig_bar = plot_forecast_breakdown(future_df, daily_sales_df)
    st.plotly_chart(fig_pie, use_container_width=True)
    st.plotly_chart(fig_bar, use_container_width=True)
    
    display_forecast_insights(daily_sales_df, metrics, future_df)
    
    st.subheader(f"Future Prediction Data ({model_name})")
    st.table(style_future_sales_table(future_df))
    st.info(f"Prediction complete for Product: {product_stock_code}")
    
    st.session_state.trained_model = model
    st.session_state.daily_sales_df = daily_sales_df
    st.session_state.scaler = scaler
    st.session_state.seq_length = seq_length
    st.session_state.target_col_idx = target_col_idx
    st.session_state.model_trained = True

