from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import matplotlib.pyplot as plt
import io
import base64
import os

# Initialize Flask app
app = Flask(__name__, template_folder='../templates')
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def load_and_prepare_data(file_path):
    """Loads and prepares the data from the given file path."""
    try:
        # Load the data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding='latin1')
        else:
            df = pd.read_excel(file_path)

        # Basic data cleaning
        df.dropna(subset=['Invoice', 'StockCode', 'Description', 'Quantity', 'Price', 'Customer ID', 'Country'], inplace=True)
        df = df[df['Quantity'] > 0]
        df = df[df['Price'] > 0]

        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

        # Create 'Seller_Listed' and 'Customer_Paid' columns
        df['Seller_Listed'] = df['Price']
        df['Customer_Paid'] = df['Price'] * np.random.uniform(0.8, 1.0, size=len(df))
        df['Price_Gap'] = df['Seller_Listed'] - df['Customer_Paid']

        return df

    except Exception as e:
        print(f"Error loading or preparing data: {e}")
        return None

def forecast_prices(df):
    """Forecasts future prices using Holt-Winters model."""
    try:
        daily_prices = df.set_index('InvoiceDate')['Seller_Listed'].resample('D').mean().fillna(method='ffill')
        model = ExponentialSmoothing(daily_prices, seasonal='add', seasonal_periods=7).fit()
        forecast = model.forecast(30)

        fig, ax = plt.subplots(figsize=(12, 6))
        daily_prices.plot(ax=ax, label='Observed')
        forecast.plot(ax=ax, label='Forecast')
        ax.set_title('Price Forecast (next 30 days)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        plt.grid(True)

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
        plt.close()

        return plot_url, forecast.mean()

    except Exception as e:
        print(f"Error in forecasting: {e}")
        return None, None


@app.route('/')
def index():
    """Renders the index page."""
    return render_template('index.html')

@app.route('/api/price_analysis', methods=['POST'])
def price_analysis():
    """API endpoint for price analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        df = load_and_prepare_data(file_path)
        if df is None:
            return jsonify({'error': 'Could not process data'}), 500

        forecast_plot, avg_forecast_price = forecast_prices(df)

        fair_price = df['Customer_Paid'].mean()
        price_gap_stats = {
            'mean': df['Price_Gap'].mean(),
            'median': df['Price_Gap'].median(),
            'std_dev': df['Price_Gap'].std()
        }

        return jsonify({
            'forecast_plot': forecast_plot,
            'avg_forecast_price': avg_forecast_price,
            'fair_price_recommendation': fair_price,
            'price_gap_stats': price_gap_stats
        })

if __name__ == '__main__':
    app.run(debug=True)
