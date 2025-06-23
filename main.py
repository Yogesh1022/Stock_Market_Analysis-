# stack/main.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os

# --- Streamlit Configuration ---
st.set_page_config(page_title="📈 Stock Market Dashboard", layout="wide", initial_sidebar_state="expanded")
sns.set(style="whitegrid")

# --- Custom Styles ---
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #4CAF50;
            font-size: 40px;
            font-weight: bold;
        }
        .description {
            text-align: center;
            font-size: 18px;
            color: #555;
        }
        .sidebar-title {
            font-weight: bold;
            color: #4CAF50;
        }
        .stTabs [role="tab"] {
            font-size: 16px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<div class='main-title'>📈 Advanced Stock Market Analysis Dashboard</div>", unsafe_allow_html=True)
st.markdown("""
<div class='description'>
    This dashboard allows you to:
    <ul>
        <li>Analyze stock prices with <strong>Moving Averages (SMA)</strong></li>
        <li>Track <strong>Relative Strength Index (RSI)</strong> and <strong>MACD</strong> indicators</li>
        <li>Compare volatility across industries</li>
        <li>Explore <strong>correlation heatmaps</strong> and <strong>return distributions</strong></li>
    </ul>
</div>
""", unsafe_allow_html=True)

# --- Helper Functions ---
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def plot_price_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['Date'], df['Close'], label='Close Price', color='blue')
    ax.plot(df['Date'], df['SMA_50'], label='SMA 50', linestyle='--', color='green')
    ax.plot(df['Date'], df['SMA_200'], label='SMA 200', linestyle='--', color='red')
    ax.set_title(f"{symbol} - Price Trend with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_rsi_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    ax.axhline(70, color='red', linestyle='--')
    ax.axhline(30, color='green', linestyle='--')
    ax.set_title(f"{symbol} - RSI (14-day)")
    ax.set_ylabel("RSI")
    ax.grid(True)
    st.pyplot(fig)

def plot_macd_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df['Date'], df['MACD'], label='MACD', color='orange')
    ax.set_title(f"{symbol} - MACD")
    ax.grid(True)
    st.pyplot(fig)

# --- Data Loading ---
uploaded_file = st.file_uploader("📤 Upload your Nifty_Stocks CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
elif os.path.exists('Nifty_Stocks.csv'):
    df = pd.read_csv('Nifty_Stocks.csv')
    st.info("✅ Loaded Nifty_Stocks.csv from local folder.")
else:
    st.warning("⚠️ Please upload a CSV file to continue.")
    st.stop()

# --- Preprocessing ---
df.fillna(method='ffill', inplace=True)
df.dropna(inplace=True)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)

# Calculate indicators
df['SMA_50'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=50).mean())
df['SMA_200'] = df.groupby('Symbol')['Close'].transform(lambda x: x.rolling(window=200).mean())
df['RSI'] = df.groupby('Symbol')['Close'].transform(compute_rsi)
ema_12 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
ema_26 = df.groupby('Symbol')['Close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
df['MACD'] = ema_12 - ema_26

# --- Sidebar Controls ---
st.sidebar.markdown("<h3 class='sidebar-title'>⚙️ Filter Options</h3>", unsafe_allow_html=True)
selected_symbols = st.sidebar.multiselect("Select Stock Symbols", sorted(df['Symbol'].unique()), default=[df['Symbol'].unique()[0]])

min_date = df['Date'].min()
max_date = df['Date'].max()

date_range = st.sidebar.date_input("Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date)

start_date = pd.to_datetime(date_range[0])
end_date = pd.to_datetime(date_range[1])

filtered_df = df[(df['Symbol'].isin(selected_symbols)) & (df['Date'] >= start_date) & (df['Date'] <= end_date)]

# --- Layout with Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Stock Charts", "📉 Volatility", "📈 Correlation Heatmap", "🔍 Daily Return Distribution"])

with tab1:
    for symbol in selected_symbols:
        stock_data = filtered_df[filtered_df['Symbol'] == symbol]
        st.subheader(f"📌 {symbol} - Stock Analysis")
        plot_price_chart(stock_data, symbol)
        col1, col2 = st.columns(2)
        with col1:
            plot_rsi_chart(stock_data, symbol)
        with col2:
            plot_macd_chart(stock_data, symbol)

with tab2:
    if 'Category' in df.columns and 'Volatility' in df.columns:
        st.subheader("📉 Volatility Comparison Across Categories")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(x='Category', y='Volatility', data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        st.pyplot(fig)
    else:
        st.info("ℹ️ Volatility or Category data not available in this dataset.")

with tab3:
    st.subheader("📈 Feature Correlation Heatmap")
    corr_cols = ['Open', 'High', 'Low', 'Close', 'SMA_50', 'SMA_200', 'RSI', 'MACD', 'Volatility']
    available_cols = [col for col in corr_cols if col in df.columns]

    if available_cols:
        fig, ax = plt.subplots(figsize=(12, 8))
        corr = df[available_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.info("ℹ️ No numerical columns found for correlation heatmap.")

with tab4:
    if 'Daily_Return' in df.columns:
        st.subheader("🔍 Distribution of Daily Returns")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(df['Daily_Return'], bins=50, kde=True, color='teal', ax=ax)
        ax.grid(True)
        st.pyplot(fig)
    else:
        st.info("ℹ️ Daily_Return column not found in this dataset.")

# --- Show Raw Data ---
with st.expander("📂 View Raw Data"):
    st.dataframe(filtered_df, use_container_width=True)

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Made with ❤️ using Streamlit | Styled by Seaborn & Matplotlib</p>", unsafe_allow_html=True)