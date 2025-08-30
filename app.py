import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ---------------- Streamlit Auto Refresh ----------------
st_autorefresh(interval=30*60*1000, key="refresh")

st.title("📊 Stock Portfolio Dashboard")

# ---------------- User Inputs ----------------
selected_stocks = st.multiselect("Select Stocks", ["TCS.NS","INFY.NS","RELIANCE.NS","HDFCBANK.NS"], default=["TCS.NS","INFY.NS"])
st.sidebar.subheader("⚙️ Alert Settings")
min_sharpe = st.sidebar.slider("Min Sharpe Ratio Alert", 0.0, 2.0, 1.0, 0.1)
max_volatility = st.sidebar.slider("Max Volatility Alert (%)", 0.0, 100.0, 30.0, 1.0)
portfolio_return_alert = st.sidebar.slider("Min Portfolio Return Alert (%)", -50.0, 50.0, 0.0, 1.0)
stock_drop = st.sidebar.slider("Stock Drop Alert (%)", -20.0, 0.0, -5.0, 0.5)
stock_jump = st.sidebar.slider("Stock Jump Alert (%)", 0.0, 20.0, 5.0, 0.5)

# ---------------- Portfolio Backtest ----------------
portfolio, trade_logs = portfolio_backtest(selected_stocks)
metrics = calculate_portfolio_metrics(portfolio)

st.subheader("📈 Portfolio Equity Curve")
fig = px.line(portfolio, y=["Portfolio"], title="Portfolio Growth")
st.plotly_chart(fig, use_container_width=True)

st.subheader("📊 Portfolio Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("CAGR %", metrics["CAGR %"])
col2.metric("Sharpe", metrics["Sharpe"])
col3.metric("Max DD %", metrics["Max DD %"])
st.success(f"✅ Total Return: {metrics['Total Return %']}%")

# ---------------- Stock Contribution Heatmap ----------------
st.subheader("📊 Stock Contribution Heatmap")
contributions = ((portfolio.iloc[-1,:-1] - portfolio.iloc[0,:-1])/portfolio.iloc[0,:-1])*100
contrib_df = pd.DataFrame(contributions, columns=["Return %"])
plt.figure(figsize=(6,4))
sns.heatmap(contrib_df.T, annot=True, cmap="RdYlGn", center=0, fmt=".2f")
plt.title("Stock Contributions")
st.pyplot(plt)

# ---------------- Live Data & Alerts ----------------
live_data = fetch_live_data(selected_stocks)
portfolio_alerts = check_portfolio_alerts(metrics["Total Return %"], metrics["Sharpe"], metrics["Max DD %"], min_sharpe, max_volatility, portfolio_return_alert)
stock_alerts = check_stock_alerts(live_data, stock_drop, stock_jump)
all_alerts = portfolio_alerts + stock_alerts

st.subheader("🔔 Alerts")
if all_alerts:
    for alert in all_alerts:
        st.error(alert)
else:
    st.success("✅ No alerts triggered.")

