# dashboard.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import joblib
import os
from keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
    }
    .positive {
        color: #00cc00;
    }
    .negative {
        color: #ff0000;
    }
    .prediction-box {
        background: #1f1f1f;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #333;
    }
    .model-status {
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .model-available {
        background-color: #1a3a1a;
        border: 1px solid #00cc00;
    }
    .model-not-available {
        background-color: #3a1a1a;
        border: 1px solid #ff0000;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">Global Financial Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## Stock Search")
st.sidebar.markdown("---")

# Date range selector
st.sidebar.markdown("## Date Range")
end_date = datetime.datetime.now()
start_date = st.sidebar.date_input(
    "Select Start Date",
    value=end_date - timedelta(days=365),
    max_value=end_date
)

# Country selection for top stocks
st.sidebar.markdown("---")
st.sidebar.markdown("## Country Selection")
country = st.sidebar.selectbox(
    "Select Country",
    ["USA", "India", "UK", "Germany", "Japan", "China", "Canada", "Australia", "France", "South Korea"]
)

# Model info
st.sidebar.markdown("---")
st.sidebar.markdown("## ML Model Status")

# Define the 15 stocks that we trained
TRAINED_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "VTI",
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    "HSBA.L", "AZN.L",
    "SAP.DE"
]

# Check which models are available
def get_available_models():
    """Get list of available trained models"""
    if not os.path.exists('models'):
        return []
    
    models = []
    for file in os.listdir('models'):
        if file.endswith('_model.h5'):
            symbol = file.replace('_model.h5', '')
            models.append(symbol)
    return models

available_models = get_available_models()

if available_models:
    st.sidebar.success(f"Models available: {len(available_models)} stocks")
    st.sidebar.info(f"Examples: {', '.join(available_models[:5])}")
else:
    st.sidebar.warning("No models found! Run train_models.py first")

# Top companies by country
def get_top_companies(country):
    companies = {
        "USA": [
            {"symbol": "AAPL", "name": "Apple Inc."},
            {"symbol": "MSFT", "name": "Microsoft Corp."},
            {"symbol": "GOOGL", "name": "Alphabet Inc."},
            {"symbol": "AMZN", "name": "Amazon.com Inc."},
            {"symbol": "NVDA", "name": "NVIDIA Corp."},
            {"symbol": "META", "name": "Meta Platforms Inc."},
            {"symbol": "TSLA", "name": "Tesla Inc."},
            {"symbol": "BRK-B", "name": "Berkshire Hathaway"},
            {"symbol": "VTI", "name": "Vanguard Total Stock Market"},
            {"symbol": "JPM", "name": "JPMorgan Chase"}
        ],
        "India": [
            {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
            {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
            {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
            {"symbol": "INFY.NS", "name": "Infosys"},
            {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
            {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
            {"symbol": "ITC.NS", "name": "ITC Limited"},
            {"symbol": "SBIN.NS", "name": "State Bank of India"},
            {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
            {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"}
        ],
        "UK": [
            {"symbol": "HSBA.L", "name": "HSBC Holdings"},
            {"symbol": "AZN.L", "name": "AstraZeneca"},
            {"symbol": "BP.L", "name": "BP plc"},
            {"symbol": "SHEL.L", "name": "Shell plc"},
            {"symbol": "GSK.L", "name": "GSK plc"},
            {"symbol": "DGE.L", "name": "Diageo"},
            {"symbol": "BATS.L", "name": "British American Tobacco"},
            {"symbol": "GLEN.L", "name": "Glencore"},
            {"symbol": "RIO.L", "name": "Rio Tinto"},
            {"symbol": "ULVR.L", "name": "Unilever"}
        ],
        "Germany": [
            {"symbol": "SAP.DE", "name": "SAP SE"},
            {"symbol": "SIE.DE", "name": "Siemens AG"},
            {"symbol": "BAS.DE", "name": "BASF SE"},
            {"symbol": "ALV.DE", "name": "Allianz SE"},
            {"symbol": "DTE.DE", "name": "Deutsche Telekom"},
            {"symbol": "BMW.DE", "name": "BMW"},
            {"symbol": "VOW3.DE", "name": "Volkswagen AG"},
            {"symbol": "BAYN.DE", "name": "Bayer AG"},
            {"symbol": "DBK.DE", "name": "Deutsche Bank"},
            {"symbol": "MUV2.DE", "name": "Munich Re"}
        ],
        "Japan": [
            {"symbol": "7203.T", "name": "Toyota Motor"},
            {"symbol": "9984.T", "name": "SoftBank Group"},
            {"symbol": "6758.T", "name": "Sony Group"},
            {"symbol": "8306.T", "name": "Mitsubishi UFJ Financial"},
            {"symbol": "9432.T", "name": "Nippon Telegraph & Telephone"},
            {"symbol": "6501.T", "name": "Hitachi"},
            {"symbol": "6861.T", "name": "Keyence"},
            {"symbol": "7974.T", "name": "Nintendo"},
            {"symbol": "4519.T", "name": "Chugai Pharmaceutical"},
            {"symbol": "8058.T", "name": "Mitsubishi Corp"}
        ],
        "China": [
            {"symbol": "BABA", "name": "Alibaba Group"},
            {"symbol": "TCEHY", "name": "Tencent Holdings"},
            {"symbol": "JD", "name": "JD.com"},
            {"symbol": "PDD", "name": "Pinduoduo"},
            {"symbol": "BIDU", "name": "Baidu"},
            {"symbol": "NTES", "name": "NetEase"},
            {"symbol": "NIO", "name": "NIO Inc."},
            {"symbol": "XPEV", "name": "XPeng Inc."},
            {"symbol": "LI", "name": "Li Auto"},
            {"symbol": "TAL", "name": "TAL Education Group"}
        ],
        "Canada": [
            {"symbol": "RY.TO", "name": "Royal Bank of Canada"},
            {"symbol": "TD.TO", "name": "Toronto-Dominion Bank"},
            {"symbol": "SHOP.TO", "name": "Shopify Inc."},
            {"symbol": "BNS.TO", "name": "Bank of Nova Scotia"},
            {"symbol": "ENB.TO", "name": "Enbridge Inc."},
            {"symbol": "CNQ.TO", "name": "Canadian Natural Resources"},
            {"symbol": "BCE.TO", "name": "BCE Inc."},
            {"symbol": "TRP.TO", "name": "TC Energy Corp."},
            {"symbol": "FTS.TO", "name": "Fortis Inc."},
            {"symbol": "T.TO", "name": "Telus Corp."}
        ],
        "Australia": [
            {"symbol": "CBA.AX", "name": "Commonwealth Bank"},
            {"symbol": "BHP.AX", "name": "BHP Group"},
            {"symbol": "CSL.AX", "name": "CSL Limited"},
            {"symbol": "WBC.AX", "name": "Westpac Banking Corp"},
            {"symbol": "NAB.AX", "name": "National Australia Bank"},
            {"symbol": "ANZ.AX", "name": "ANZ Group"},
            {"symbol": "WES.AX", "name": "Wesfarmers"},
            {"symbol": "TLS.AX", "name": "Telstra Group"},
            {"symbol": "WOW.AX", "name": "Woolworths Group"},
            {"symbol": "GMG.AX", "name": "Goodman Group"}
        ],
        "France": [
            {"symbol": "MC.PA", "name": "LVMH Moet Hennessy"},
            {"symbol": "TTE.PA", "name": "TotalEnergies SE"},
            {"symbol": "SAN.PA", "name": "Sanofi"},
            {"symbol": "OR.PA", "name": "L'Oreal"},
            {"symbol": "AI.PA", "name": "Airbus SE"},
            {"symbol": "SU.PA", "name": "Schneider Electric"},
            {"symbol": "DANO.PA", "name": "Danone"},
            {"symbol": "CAP.PA", "name": "Capgemini SE"},
            {"symbol": "CS.PA", "name": "AXA SA"},
            {"symbol": "ENGI.PA", "name": "Engie SA"}
        ],
        "South Korea": [
            {"symbol": "005930.KS", "name": "Samsung Electronics"},
            {"symbol": "000660.KS", "name": "SK Hynix Inc."},
            {"symbol": "005380.KS", "name": "Hyundai Motor"},
            {"symbol": "006400.KS", "name": "Samsung SDI"},
            {"symbol": "035420.KS", "name": "NAVER Corp."},
            {"symbol": "051910.KS", "name": "LG Chem"},
            {"symbol": "012330.KS", "name": "Hyundai Mobis"},
            {"symbol": "005490.KS", "name": "POSCO Holdings"},
            {"symbol": "015760.KS", "name": "Korea Electric Power"},
            {"symbol": "032830.KS", "name": "Samsung Life Insurance"}
        ]
    }
    return companies.get(country, companies["USA"])

@st.cache_data(ttl=300)
def get_stock_data(symbol, start_date, end_date):
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(start=start_date, end=end_date)
        info = stock.info
        return data, info
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None, None

@st.cache_data(ttl=300)
def get_multiple_stocks(symbols, start_date, end_date):
    data_dict = {}
    for symbol in symbols:
        data, _ = get_stock_data(symbol, start_date, end_date)
        if data is not None and not data.empty:
            data_dict[symbol] = data['Close']
    return pd.DataFrame(data_dict)

@st.cache_resource
def load_pretrained_model(symbol):
    """Load a pretrained model for a specific stock"""
    model_dir = 'models'
    model_path = os.path.join(model_dir, f'{symbol}_model.h5')
    scaler_path = os.path.join(model_dir, f'{symbol}_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except:
        return None, None

def predict_future_prices(model, scaler, data, lookback_days=60, days=3):
    """Predict future prices using loaded model"""
    if model is None or scaler is None:
        return None
    
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.transform(close_prices)
    
    last_sequence = scaled_data[-lookback_days:].reshape(1, lookback_days, 1)
    
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        next_pred = model.predict(current_sequence, verbose=0)
        predictions.append(next_pred[0, 0])
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, 0] = next_pred[0, 0]
    
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def calculate_technical_indicators(data):
    df = data.copy()
    
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + 2 * bb_std
    df['BB_Lower'] = df['BB_Middle'] - 2 * bb_std
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal']
    
    return df

def plot_stock_price_with_predictions(data, symbol, name="", predictions=None):
    fig = go.Figure()
    
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    if 'MA20' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA20'],
            line=dict(color='orange', width=1),
            name='MA 20'
        ))
    if 'MA50' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA50'],
            line=dict(color='blue', width=1),
            name='MA 50'
        ))
    if 'MA200' in data.columns:
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['MA200'],
            line=dict(color='red', width=1),
            name='MA 200'
        ))
    
    if predictions is not None:
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=len(predictions))
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            line=dict(color='green', width=2, dash='dash'),
            marker=dict(size=10, color='green'),
            name='Predicted'
        ))
        
        std_dev = data['Close'].std() * 0.1
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions + std_dev,
            mode='lines',
            line=dict(width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions - std_dev,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.2)',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f'{name} ({symbol}) - Price Chart with Predictions',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def plot_volume(data, symbol):
    colors = ['green' if close >= open else 'red' for close, open in zip(data['Close'], data['Open'])]
    
    fig = go.Figure(data=[go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        name='Volume'
    )])
    
    fig.update_layout(
        title=f'{symbol} - Trading Volume',
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_dark',
        height=300
    )
    
    return fig

def plot_rsi(data, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['RSI'],
        line=dict(color='purple', width=2),
        name='RSI'
    ))
    
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    
    fig.update_layout(
        title=f'{symbol} - Relative Strength Index (RSI)',
        xaxis_title='Date',
        yaxis_title='RSI',
        template='plotly_dark',
        height=300
    )
    
    return fig

def plot_macd(data, symbol):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MACD'],
        line=dict(color='blue', width=2),
        name='MACD'
    ))
    
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Signal'],
        line=dict(color='red', width=2),
        name='Signal'
    ))
    
    colors = ['green' if val >= 0 else 'red' for val in data['MACD_Histogram']]
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['MACD_Histogram'],
        marker_color=colors,
        name='Histogram'
    ))
    
    fig.update_layout(
        title=f'{symbol} - MACD',
        xaxis_title='Date',
        yaxis_title='MACD',
        template='plotly_dark',
        height=300
    )
    
    return fig

def plot_stock_comparison(data_dict):
    fig = go.Figure()
    
    for symbol, data in data_dict.items():
        normalized = (data / data.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=normalized.index,
            y=normalized,
            name=symbol,
            mode='lines'
        ))
    
    fig.update_layout(
        title='Stock Performance Comparison (%)',
        xaxis_title='Date',
        yaxis_title='Percentage Change (%)',
        template='plotly_dark',
        height=400,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Stock Search", "ML Predictions", "Top Companies", "Market Overview"])

# Tab 1: Stock Search
with tab1:
    st.markdown("## Stock Search")
    st.markdown("Select a country and search for a company from the Top Companies tab.")
    st.info("Go to the **Top Companies** tab to browse the 10 companies for the selected country and search for a specific stock.")

# Tab 2: ML Predictions (NEW)
with tab2:
    st.markdown("## Machine Learning Predictions")
    st.markdown("Predict next 3 days stock prices using LSTM models trained on 5 years of data")
    
    # Check if models are available
    available_models = get_available_models()
    
    if not available_models:
        st.warning("No trained models found! Please run the training script first.")
        st.info("""
        To train models:
        1. Run `colab_train_15_stocks.ipynb` on Google Colab
        2. Download the `models` folder
        3. Place it in the same directory as this dashboard
        """)
    else:
        # Stock selector - only show stocks that have trained models
        stock_options = []
        for stock in TRAINED_STOCKS:
            if stock in available_models:
                stock_options.append(stock)
        
        if not stock_options:
            st.warning("No trained models found for the defined stocks!")
        else:
            selected_stock = st.selectbox(
                "Select Stock for Prediction",
                stock_options,
                key="ml_stock"
            )
            
            if selected_stock:
                # Fetch data for the selected stock
                data, info = get_stock_data(selected_stock, start_date, end_date)
                
                if data is not None and not data.empty:
                    # Display current info
                    current_price = data['Close'].iloc[-1]
                    company_name = info.get('longName', selected_stock)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Current Price", f"${current_price:.2f}")
                    
                    with col2:
                        prev_close = data['Close'].iloc[-2] if len(data) > 1 else current_price
                        change = ((current_price - prev_close) / prev_close) * 100
                        st.metric("Daily Change", f"{change:.2f}%")
                    
                    with col3:
                        volume = data['Volume'].iloc[-1]
                        st.metric("Volume", f"{volume:,.0f}")
                    
                    # Predict button
                    if st.button("Predict Next 3 Days", key="predict_ml"):
                        with st.spinner("Loading model and making predictions..."):
                            # Load model
                            model, scaler = load_pretrained_model(selected_stock)
                            
                            if model is None or scaler is None:
                                st.error("Model not found! Please train the model first.")
                            else:
                                # Make predictions
                                predictions = predict_future_prices(model, scaler, data)
                                
                                if predictions is not None:
                                    # Create future dates
                                    last_date = data.index[-1]
                                    future_dates = pd.date_range(
                                        start=last_date + timedelta(days=1),
                                        periods=len(predictions)
                                    )
                                    
                                    # Show predictions in a nice layout
                                    st.subheader("Predicted Prices")
                                    
                                    # Create 3 columns for day predictions
                                    pred_col1, pred_col2, pred_col3 = st.columns(3)
                                    
                                    with pred_col1:
                                        change_d1 = ((predictions[0] - current_price) / current_price) * 100
                                        st.markdown(f"""
                                            <div class="prediction-box">
                                                <h3 style="margin: 0;">Day 1</h3>
                                                <h2 style="margin: 0.5rem 0; color: #00cc00;">${predictions[0]:.2f}</h2>
                                                <p style="color: {'green' if change_d1 >= 0 else 'red'}; margin: 0;">
                                                    {change_d1:+.2f}%
                                                </p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with pred_col2:
                                        change_d2 = ((predictions[1] - current_price) / current_price) * 100
                                        st.markdown(f"""
                                            <div class="prediction-box">
                                                <h3 style="margin: 0;">Day 2</h3>
                                                <h2 style="margin: 0.5rem 0; color: #00cc00;">${predictions[1]:.2f}</h2>
                                                <p style="color: {'green' if change_d2 >= 0 else 'red'}; margin: 0;">
                                                    {change_d2:+.2f}%
                                                </p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    with pred_col3:
                                        change_d3 = ((predictions[2] - current_price) / current_price) * 100
                                        st.markdown(f"""
                                            <div class="prediction-box">
                                                <h3 style="margin: 0;">Day 3</h3>
                                                <h2 style="margin: 0.5rem 0; color: #00cc00;">${predictions[2]:.2f}</h2>
                                                <p style="color: {'green' if change_d3 >= 0 else 'red'}; margin: 0;">
                                                    {change_d3:+.2f}%
                                                </p>
                                            </div>
                                        """, unsafe_allow_html=True)
                                    
                                    # Overall direction
                                    direction = "UP" if predictions[-1] > current_price else "DOWN"
                                    color = "green" if direction == "UP" else "red"
                                    
                                    st.markdown(f"""
                                        <div style="background: #1f1f1f; padding: 1rem; border-radius: 10px; text-align: center; border: 2px solid {color}; margin: 1rem 0;">
                                            <h3 style="color: {color}; margin: 0;">
                                                Predicted Direction: {direction}
                                            </h3>
                                            <p style="color: #888; margin-top: 0.5rem;">
                                                Based on LSTM model trained on 5 years of historical data
                                            </p>
                                        </div>
                                    """, unsafe_allow_html=True)
                                    
                                    # Detailed table
                                    st.subheader("Prediction Details")
                                    
                                    pred_df = pd.DataFrame({
                                        'Day': [1, 2, 3],
                                        'Date': future_dates.strftime('%Y-%m-%d'),
                                        'Predicted Price ($)': predictions.round(2),
                                        'Change ($)': (predictions - current_price).round(2),
                                        'Change (%)': ((predictions - current_price) / current_price * 100).round(2)
                                    })
                                    
                                    st.dataframe(pred_df, use_container_width=True)
                                    
                                    # Show chart with predictions
                                    fig_pred = plot_stock_price_with_predictions(
                                        data, selected_stock, company_name, predictions
                                    )
                                    st.plotly_chart(fig_pred, use_container_width=True)
                                    
                                else:
                                    st.error("Failed to make predictions. Please check the model.")
                else:
                    st.error(f"Failed to fetch data for {selected_stock}")

# Tab 3: Top Companies
with tab3:
    st.markdown(f"## Top Companies - {country}")

    top_companies = get_top_companies(country)
    symbols = [company['symbol'] for company in top_companies]

    # Search within the 10 companies for the selected country
    company_options = [f"{company['symbol']} - {company['name']}" for company in top_companies]
    selected_company = st.selectbox(
        "Search / Select a Company",
        company_options,
        key="top_company_search"
    )
    selected_symbol = selected_company.split(" - ")[0]

    # Show detailed information for the selected company
    selected_data, selected_info = get_stock_data(selected_symbol, start_date, end_date)
    if selected_data is not None and not selected_data.empty:
        current_price = selected_data['Close'].iloc[-1]
        prev_price = selected_data['Close'].iloc[-2] if len(selected_data) > 1 else current_price
        change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0

        st.markdown(f"### {selected_symbol} - Detailed Analysis")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric("Current Price", f"${current_price:.2f}")
        with metric_cols[1]:
            st.metric("Day Change", f"{change:.2f}%")
        with metric_cols[2]:
            st.metric("Company", selected_symbol)

        fig_selected = go.Figure()
        fig_selected.add_trace(go.Scatter(
            x=selected_data.index,
            y=selected_data['Close'],
            mode='lines',
            name=selected_symbol
        ))
        fig_selected.update_layout(
            title=f"{selected_symbol} Price Chart",
            template='plotly_dark',
            xaxis_title='Date',
            yaxis_title='Price'
        )
        st.plotly_chart(fig_selected, use_container_width=True)

    st.markdown("### Top 10 Companies")
    cols = st.columns(5)

    for idx, company in enumerate(top_companies):
        with cols[idx % 5]:
            try:
                stock_data, info = get_stock_data(company['symbol'], start_date, end_date)
                if stock_data is not None and not stock_data.empty:
                    current_price = stock_data['Close'].iloc[-1]
                    prev_price = stock_data['Close'].iloc[-2] if len(stock_data) > 1 else current_price
                    change = ((current_price - prev_price) / prev_price) * 100 if prev_price != 0 else 0

                    card_color = "#00cc00" if change >= 0 else "#ff0000"
                    st.markdown(f"""
                        <div style="background: #1f1f1f; padding: 0.5rem; border-radius: 8px; margin: 0.2rem 0; border-left: 3px solid {card_color};">
                            <div style="font-weight: bold; font-size: 0.9rem;">{company['symbol']}</div>
                            <div style="font-size: 0.7rem; color: #888;">{company['name'][:15]}</div>
                            <div style="font-size: 1.1rem; font-weight: bold;">${current_price:.2f}</div>
                            <div style="color: {card_color}; font-size: 0.8rem;">
                                {'▲' if change >= 0 else '▼'} {abs(change):.2f}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass

    st.markdown("### Stock Performance Comparison")

    data_dict = get_multiple_stocks(symbols[:10], start_date, end_date)

    if not data_dict.empty:
        fig_comparison = plot_stock_comparison(data_dict)
        st.plotly_chart(fig_comparison, use_container_width=True)

    st.markdown("### Stock Correlation Matrix")

    if len(data_dict.columns) > 1:
        correlation_matrix = data_dict.pct_change().corr()

        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=correlation_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))

        fig_corr.update_layout(
            title='Stock Correlation Matrix',
            template='plotly_dark',
            height=500,
            width=700
        )

        st.plotly_chart(fig_corr, use_container_width=True)

# Tab 4: Market Overview
with tab4:
    st.markdown("## Global Market Overview")
    
    indices = {
        "S&P 500": "^GSPC",
        "Dow Jones": "^DJI",
        "NASDAQ": "^IXIC",
        "NIFTY 50": "^NSEI",
        "SENSEX": "^BSESN",
        "FTSE 100": "^FTSE",
        "DAX": "^GDAXI",
        "Nikkei 225": "^N225"
    }
    
    cols = st.columns(4)
    
    for idx, (name, symbol) in enumerate(indices.items()):
        with cols[idx % 4]:
            try:
                data, info = get_stock_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    current = data['Close'].iloc[-1]
                    prev = data['Close'].iloc[-2] if len(data) > 1 else current
                    change = ((current - prev) / prev) * 100 if prev != 0 else 0
                    
                    st.metric(
                        name,
                        f"{current:.2f}",
                        f"{change:.2f}%",
                        delta_color="normal"
                    )
            except:
                st.metric(name, "N/A", "N/A")
    
    st.markdown("---")
    st.markdown("### Index Performance Chart")
    
    index_data = {}
    for name, symbol in list(indices.items())[:6]:
        try:
            data, _ = get_stock_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                normalized = (data['Close'] / data['Close'].iloc[0] - 1) * 100
                index_data[name] = normalized
        except:
            continue
    
    if index_data:
        fig_indices = go.Figure()
        for name, data in index_data.items():
            fig_indices.add_trace(go.Scatter(
                x=data.index,
                y=data,
                name=name,
                mode='lines'
            ))
        
        fig_indices.update_layout(
            title='Index Performance Comparison (Percentage Change)',
            xaxis_title='Date',
            yaxis_title='Change (%)',
            template='plotly_dark',
            height=500,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig_indices, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Current Market Snapshot")
    
    market_indicators = {
        "US Markets": {
            "S&P 500": "^GSPC",
            "NASDAQ": "^IXIC",
            "Dow Jones": "^DJI"
        },
        "Asian Markets": {
            "Nikkei 225": "^N225",
            "Sensex": "^BSESN",
            "NIFTY 50": "^NSEI"
        },
        "European Markets": {
            "FTSE 100": "^FTSE",
            "DAX": "^GDAXI",
            "CAC 40": "^FCHI"
        }
    }
    
    for region, region_data in market_indicators.items():
        st.markdown(f"**{region}**")
        cols = st.columns(len(region_data))
        
        for idx, (name, symbol) in enumerate(region_data.items()):
            with cols[idx]:
                try:
                    data, info = get_stock_data(symbol, start_date, end_date)
                    if data is not None and not data.empty:
                        current = data['Close'].iloc[-1]
                        prev = data['Close'].iloc[-2] if len(data) > 1 else current
                        change = ((current - prev) / prev) * 100 if prev != 0 else 0
                        
                        color = "green" if change >= 0 else "red"
                        
                        st.markdown(f"""
                            <div style="background: #1f1f1f; padding: 1rem; border-radius: 8px; margin: 0.2rem 0;">
                                <div style="font-weight: bold; color: #888;">{name}</div>
                                <div style="font-size: 1.5rem; font-weight: bold;">{current:.2f}</div>
                                <div style="color: {color}; font-size: 1rem;">
                                    {change:+.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                except:
                    st.markdown(f"""
                        <div style="background: #1f1f1f; padding: 1rem; border-radius: 8px; margin: 0.2rem 0;">
                            <div style="font-weight: bold; color: #888;">{name}</div>
                            <div style="font-size: 1rem; color: #888;">Data Unavailable</div>
                        </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    st.markdown("### Volatility Indicator")
    
    try:
        vix_data, _ = get_stock_data("^VIX", start_date, end_date)
        if vix_data is not None and not vix_data.empty:
            current_vix = vix_data['Close'].iloc[-1]
            
            if current_vix < 15:
                sentiment = "Low Volatility - Market Calm"
                color = "green"
            elif current_vix < 20:
                sentiment = "Moderate Volatility - Normal"
                color = "yellow"
            elif current_vix < 30:
                sentiment = "High Volatility - Market Nervous"
                color = "orange"
            else:
                sentiment = "Extreme Volatility - Market Fear"
                color = "red"
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("VIX (Fear Index)", f"{current_vix:.2f}")
            with col2:
                st.metric("Market Sentiment", sentiment)
            with col3:
                vix_change = ((vix_data['Close'].iloc[-1] - vix_data['Close'].iloc[-2]) / vix_data['Close'].iloc[-2]) * 100 if len(vix_data) > 1 else 0
                st.metric("VIX Change", f"{vix_change:+.2f}%")
            
            fig_vix = go.Figure()
            fig_vix.add_trace(go.Scatter(
                x=vix_data.index,
                y=vix_data['Close'],
                name='VIX',
                line=dict(color='purple', width=2)
            ))
            
            fig_vix.add_hline(y=15, line_dash="dash", line_color="green", annotation_text="Low Volatility")
            fig_vix.add_hline(y=20, line_dash="dash", line_color="yellow", annotation_text="Moderate")
            fig_vix.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="High Volatility")
            
            fig_vix.update_layout(
                title='VIX - Volatility Index',
                xaxis_title='Date',
                yaxis_title='VIX Value',
                template='plotly_dark',
                height=400
            )
            
            st.plotly_chart(fig_vix, use_container_width=True)
    except:
        st.info("VIX data currently unavailable")

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.8rem;">
        <p>Data provided by Yahoo Finance | Dashboard built with Streamlit</p>
        <p>Disclaimer: This is for informational purposes only. Not financial advice.</p>
    </div>
""", unsafe_allow_html=True)
