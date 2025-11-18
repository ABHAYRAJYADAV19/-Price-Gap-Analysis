Price Gap Analysis Dashboard
A web-based analytical tool that helps businesses optimize pricing strategies by analyzing the difference between seller-listed prices and actual customer-paid prices, with AI-powered forecasting capabilities.

Python Flask License

🎯 Overview
The Price Gap Analysis Dashboard analyzes retail transaction data to:

Identify price gaps between listed and paid prices
Forecast future pricing trends for the next 30 days
Recommend fair prices based on customer behavior
Visualize patterns with interactive charts
✨ Features
Core Functionality
✅ Upload CSV/Excel transaction data via drag-and-drop
✅ Automated data cleaning and validation
✅ Time series forecasting using Holt-Winters Exponential Smoothing
✅ Statistical analysis (mean, median, standard deviation)
✅ Interactive visualizations with Matplotlib
✅ Real-time price gap calculations
UI/UX Highlights
🎨 Modern, responsive design with smooth animations
💡 Tooltips explaining every metric
📱 Mobile-friendly interface
🚀 Scroll-triggered animations
⚡ Fast, client-side interactions
🎭 Three analysis modes (Basic, Advanced, Comprehensive)
🚀 Quick Start
Prerequisites
Python 3.8 or higher
pip package manager
Installation
Clone or download the project

cd Price-Gap-Analysis
Install dependencies

pip install -r src/requirements.txt
Running the Application
Navigate to the src directory

cd src
Start the Flask server

python app.py
Open your browser

http://127.0.0.1:5000
The application will start in debug mode with auto-reload enabled.

📁 Project Structure
Price-Gap-Analysis/
├── src/
│   ├── app.py                    # Main Flask application
│   ├── customer_segmentation.py  # Customer clustering module
│   ├── data_processing.py        # Data cleaning utilities
│   ├── eda.py                    # Exploratory data analysis
│   ├── forecasting.py            # Time series forecasting
│   ├── utilities.py              # Helper functions
│   └── requirements.txt          # Python dependencies
├── templates/
│   └── index.html                # Main dashboard UI
├── uploads/                      # Temporary file storage
└── requirements.txt              # Root dependencies
📊 How It Works
1. Data Upload
Upload retail transaction data in CSV or Excel format with the following required columns:

Column	Description	Type	Example
Invoice	Invoice number	String/Number	"536365"
StockCode	Product code	String/Number	"85123A"
Description	Product name	String	"WHITE HANGING HEART"
Quantity	Items purchased	Number (>0)	6
InvoiceDate	Transaction date	Date/Datetime	"12/1/2010 8:26"
Price	Unit price	Number (>0)	2.55
Customer ID	Customer identifier	String/Number	17850
Country	Customer location	String	"United Kingdom"
2. Data Processing Pipeline
Upload → Validation → Cleaning → Feature Engineering → Analysis → Visualization
Validation: Checks file format, size (max 16MB), and required columns
Cleaning: Removes nulls, filters invalid quantities/prices
Feature Engineering: Creates Seller_Listed, Customer_Paid, Price_Gap columns
Analysis: Applies Holt-Winters forecasting with 7-day seasonality
Visualization: Generates interactive charts and statistics
3. Price Forecasting
Uses Holt-Winters Exponential Smoothing:

Seasonal component: Additive
Seasonal period: 7 days (weekly patterns)
Forecast horizon: 30 days
4. Results Display
Fair Price Recommendation

Based on average customer-paid prices
Represents market willingness to pay
Price Gap Statistics

Mean Gap: Average difference
Median Gap: Middle value (outlier-resistant)
Standard Deviation: Variability measure
Forecast Chart

Blue line: Historical observed prices
Orange line: 30-day price forecast
🛠️ Technology Stack
Backend
Flask - Python web framework
Pandas - Data manipulation
NumPy - Numerical computing
Statsmodels - Statistical modeling & time series
Matplotlib - Data visualization
Frontend
HTML5/CSS3 - Modern web standards
Vanilla JavaScript - No framework dependencies
Font Awesome - Icon library
CSS Animations - Smooth transitions & effects
Additional Libraries
openpyxl - Excel file support
scikit-learn - Machine learning utilities
seaborn - Statistical visualizations
plotly - Interactive charts
📖 Usage Guide
Step 1: Choose Analysis Type
Select from three analysis modes:

Basic: Quick price gap overview
Advanced: Detailed forecasting
Comprehensive: Full analysis with AI insights
Step 2: Upload Data
Drag & drop your file, or click "Browse Files"
Supported formats: .csv, .xlsx, .xls
Maximum file size: 16MB
Step 3: Analyze
Click "Analyze Data" to process your file. The system will:

Clean and validate data
Calculate price gaps
Generate forecasts
Create visualizations
Step 4: Review Results
Examine:

Price forecast chart (30-day prediction)
Fair price recommendation
Statistical metrics (hover for explanations)
🎨 UI Features
Interactive Elements
Hover Effects: All cards, buttons, and options respond to mouse interaction
Tooltips: Helpful explanations on metrics
Animations: Smooth transitions, scroll-triggered effects
Tab Navigation: Organized content sections
Responsive Design: Works on all screen sizes
Visual Feedback
Upload area highlights on drag-over
Selected options pulse with color
Loading spinner with animated dots
Results fade in with stagger effect
Error messages with icons
⚙️ Configuration
File Size Limit
Default: 16MB. To change, edit in templates/index.html:

if (file.size > 16 * 1024 * 1024) { // Change value here
Forecast Period
Default: 30 days. To modify, edit in src/app.py:

forecast = model.forecast(30)  # Change number of days
Seasonal Period
Default: 7 days. To adjust, edit in src/app.py:

model = ExponentialSmoothing(daily_prices, seasonal='add', seasonal_periods=7)
🔍 API Endpoint
POST /api/price_analysis
Request:

Method: POST
Content-Type: multipart/form-data
Body:
file: CSV or Excel file
analysis_type: "basic" | "advanced" | "comprehensive"
Response:

{
  "forecast_plot": "base64_encoded_image",
  "avg_forecast_price": 2.45,
  "fair_price_recommendation": 2.38,
  "price_gap_stats": {
    "mean": 0.22,
    "median": 0.2,
    "std_dev": 0.15
  }
}
Error Response:

{
  "error": "Error message description"
}
🐛 Troubleshooting
Common Issues
"No file part" error

Ensure you've selected a file before clicking "Analyze Data"
"Could not process data" error

Verify your file has all required columns
Check for missing values in critical columns
Ensure Quantity and Price values are positive
File size error

Files must be under 16MB
Consider filtering data to recent transactions
Forecast generation fails

Ensure you have sufficient data (at least 100 transactions)
Check that dates span multiple days
📈 Use Cases
E-Commerce
Optimize dynamic pricing
Identify underpriced/overpriced products
Track seasonal trends
Retail
Benchmark against competitors
Plan promotional pricing
Forecast demand-based pricing
Market Analysis
Study price elasticity
Analyze customer behavior
Compare regions/categories
🤝 Contributing
Contributions are welcome! Areas for enhancement:

Additional forecasting models (ARIMA, Prophet)
Real-time data integration
Multi-product comparison
Export functionality (PDF reports)
Advanced filtering options
📝 License
This project is licensed under the MIT License.

🙏 Acknowledgments
Built with Flask and modern web technologies
Uses Holt-Winters algorithm from Statsmodels
UI inspired by modern dashboard designs
📞 Support
For issues or questions:

Check the FAQ tab in the dashboard
Review the File Requirements section
Ensure your data meets format specifications
Made with ❤️ for data-driven pricing decisions
