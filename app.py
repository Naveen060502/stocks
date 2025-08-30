import streamlit as st
from streamlit_autorefresh import st_autorefresh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import *

# ---------------- Auto Refresh ----------------
st_autorefresh(interval=30*60*1000, key="refresh")  # refresh every 30 minutes

st.title("📊 Professional Stock Portfolio Dashboard")

# ---------------- Sidebar Inputs ----------------
st.sidebar.subheader("Stocks & Alerts Settings")

# ✅ Pre-verified NSE tickers (known to work with yfinance)
VALID_TICKERS = [
    "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "LT.NS",
    "AXISBANK.NS", "BAJFINANCE.NS"
]

selected_stocks = st.sidebar.multiselect(
    "Select Stocks", VALID_TICKERS, default=["RELIANCE.NS", "HDFCBANK.NS"]
)

# Portfolio alert sliders
min_sharpe = st.sidebar.slider("Min Sharpe Ratio Alert", 0.0, 2.0, 1.0, 0.1)
max_volatility = st.sidebar.slider("Max Volatility Alert (%)", 0.0, 100.0, 30.0, 1.0)
portfolio_return_alert = st.sidebar.slider("Min Portfolio Return Alert (%)", -50.0, 50.0, 0.0, 1.0)

# Stock alert sliders
stock_drop = st.sidebar.slider("Stock Drop Alert (%)", -20.0, 0.0, -5.0, 0.5)
stock_jump = st.sidebar.slider("Stock Jump Alert (%)", 0.0, 20.0, 5.0, 0.5)

# ---------------- Fetch & Backtest Portfolio ----------------
st.subheader("📈 Portfolio Backtest & Metrics")

# Fetch & backtest only tickers that return data
portfolio, trade_logs = portfolio_backtest(selected_stocks)

if portfolio.empty:
    st.warning("No valid stock data found for selected tickers. Please select different stocks.")
else:
    # Ensure datetime index
    if not np.issubdtype(portfolio.index.dtype, np.datetime64):
        portfolio.index = pd.to_datetime(portfolio.index)

    metrics = calculate_portfolio_metrics(portfolio)
    
    # Portfolio growth curve
    fig = px.line(portfolio, y=["Portfolio"], title="Portfolio Growth")
    st.plotly_chart(fig, use_container_width=True)

    # Portfolio metrics display
    col1, col2, col3 = st.columns(3)
    col1.metric("CAGR %", metrics["CAGR %"])
    col2.metric("Sharpe", metrics["Sharpe"])
    col3.metric("Max DD %", metrics["Max DD %"])
    st.success(f"✅ Total Return: {metrics['Total Return %']}% | Volatility: {metrics['Volatility']*100:.2f}%")

    # ---------------- Stock Contribution Heatmap ----------------
    st.subheader("📊 Stock Contribution Heatmap")
    contributions = ((portfolio.iloc[-1,:-1] - portfolio.iloc[0,:-1])/portfolio.iloc[0,:-1])*100
    contrib_df = pd.DataFrame(contributions, columns=["Return %"])
    plt.figure(figsize=(6,4))
    sns.heatmap(contrib_df.T, annot=True, cmap="RdYlGn", center=0, fmt=".2f")
    plt.title("Stock Contributions")
    st.pyplot(plt)

    # ---------------- Portfolio Optimizer ----------------
    st.subheader("⚙️ Portfolio Optimization (Max Sharpe)")
    opt_weights, opt_return, opt_vol, opt_sharpe = optimize_portfolio(selected_stocks)
    if len(opt_weights) > 0:
        opt_df = pd.DataFrame({"Stock": selected_stocks, "Weight": opt_weights})
        st.table(opt_df)
        st.markdown(f"**Expected Return:** {opt_return:.2%} | **Volatility:** {opt_vol:.2%} | **Sharpe:** {opt_sharpe:.2f}")
    else:
        st.warning("Optimizer could not run. Not enough valid stock data.")

    # ---------------- Custom Portfolio Simulator ----------------
    st.subheader("🛠️ Custom Portfolio Simulator")
    custom_weights = []
    for stock in selected_stocks:
        w = st.slider(f"Weight for {stock}", 0.0, 1.0, 0.25, 0.01)
        custom_weights.append(w)
    if sum(custom_weights) > 0:
        total_w = sum(custom_weights)
        custom_weights = [w/total_w for w in custom_weights]
        sim_return, sim_vol, sim_sharpe, sim_growth = simulate_portfolio(selected_stocks, custom_weights)
        st.markdown(f"**Simulated Portfolio:** Return {sim_return:.2%} | Volatility {sim_vol:.2%} | Sharpe {sim_sharpe:.2f}")
        if not sim_growth.empty:
            fig2 = px.line(sim_growth, title="Custom Portfolio Growth")
            st.plotly_chart(fig2)
    else:
        st.warning("Simulator skipped. All weights are zero.")

    # ---------------- Monte Carlo Efficient Frontier ----------------
    st.subheader("🎯 Monte Carlo Simulation / Efficient Frontier")
    mc_df = monte_carlo_simulation(selected_stocks, num_portfolios=2000)
    if not mc_df.empty:
        fig3, ax = plt.subplots(figsize=(8,5))
        scatter = ax.scatter(mc_df["Volatility"], mc_df["Return"], c=mc_df["Sharpe"], cmap="viridis", alpha=0.5)
        ax.set_xlabel("Volatility")
        ax.set_ylabel("Return")
        ax.set_title("Efficient Frontier")
        # Mark optimized portfolio
        if len(opt_weights) > 0:
            ax.scatter(opt_vol, opt_return, marker="*", color="red", s=200, label="Optimized")
            ax.legend()
        st.pyplot(fig3)
    else:
        st.warning("Monte Carlo simulation skipped. No valid data.")

    # ---------------- Live Data & Alerts ----------------
    st.subheader("🔔 Live Data & Alerts")
    if len(selected_stocks) > 0:
        live_data = fetch_live_data(selected_stocks)
        portfolio_alerts = check_portfolio_alerts(
            metrics["Total Return %"], metrics["Sharpe"], metrics["Volatility"],
            min_sharpe, max_volatility, portfolio_return_alert
        )
        stock_alerts = check_stock_alerts(live_data, stock_drop, stock_jump)
        all_alerts = portfolio_alerts + stock_alerts

        if all_alerts:
            alert_message = "\n".join(all_alerts)
            for alert in all_alerts:
                st.error(alert)
            # Send notifications
            send_telegram_alert(alert_message)
            send_email_alert("Stock/Portfolio Alerts 🚨", alert_message)
        else:
            st.success("✅ No alerts triggered.")
