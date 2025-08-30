import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
import requests
import smtplib
from email.mime.text import MIMEText
import streamlit as st

# ----------------- Fetch Stock Data -----------------
def fetch_stock_data(stock, start="2020-01-01", end="2025-01-01"):
    data = yf.download(stock, start=start, end=end, group_by='ticker', auto_adjust=True)
    # Handle multi-index vs single index
    if isinstance(data.columns, pd.MultiIndex):
        if stock in data.columns.levels[0]:
            data = data[stock]["Adj Close"].dropna()
        else:
            data = data["Adj Close"].dropna()
    else:
        data = data["Adj Close"].dropna()
    return data

def fetch_live_data(stocks):
    data = yf.download(stocks, period="2d", interval="15m", group_by='ticker', auto_adjust=True)
    if isinstance(data.columns, pd.MultiIndex):
        temp = pd.DataFrame()
        for stock in stocks:
            try:
                temp[stock] = data[stock]["Adj Close"]
            except:
                temp[stock] = data.iloc[:,0]  # fallback
        data = temp
    else:
        data = data[stocks]
    data = data.groupby(data.index.date).last()
    return data

# ----------------- Portfolio Backtest -----------------
def backtest_with_trades(stock, start="2020-01-01", end="2025-01-01"):
    data = fetch_stock_data(stock, start, end)
    df = pd.DataFrame(index=data.index)
    df["Close"] = data
    df["Strategy_Return"] = df["Close"].pct_change()
    df["Cumulative_Strategy"] = (1 + df["Strategy_Return"]).cumprod()
    trade_log = pd.DataFrame(columns=["Entry", "Exit", "P&L %", "Reason"])
    return df, trade_log

def portfolio_backtest(stocks, start="2020-01-01", end="2025-01-01"):
    portfolio = pd.DataFrame()
    trade_logs = {}
    for stock in stocks:
        df, log = backtest_with_trades(stock, start, end)
        portfolio[stock] = df["Cumulative_Strategy"]
        trade_logs[stock] = log
    portfolio["Portfolio"] = portfolio.mean(axis=1)
    return portfolio, trade_logs

# ----------------- Portfolio Metrics -----------------
def calculate_portfolio_metrics(portfolio, risk_free_rate=0.05):
    cum_returns = portfolio["Portfolio"]
    start_value = cum_returns.iloc[0]
    end_value = cum_returns.iloc[-1]
    n_years = (cum_returns.index[-1] - cum_returns.index[0]).days / 365
    cagr = ((end_value / start_value) ** (1/n_years) - 1) * 100
    daily_returns = cum_returns.pct_change().dropna()
    excess_returns = daily_returns - (risk_free_rate / 252)
    sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    max_dd = drawdown.min() * 100
    total_return = (end_value - 1) * 100
    volatility = daily_returns.std() * np.sqrt(252)
    return {"CAGR %": round(cagr,2), "Sharpe": round(sharpe,2), "Max DD %": round(max_dd,2), "Total Return %": round(total_return,2), "Volatility": round(volatility,2)}

# ----------------- Alerts -----------------
def check_portfolio_alerts(portfolio_return, sharpe_ratio, volatility,
                           min_sharpe, max_volatility, portfolio_return_alert):
    alerts = []
    if portfolio_return < portfolio_return_alert:
        alerts.append(f"⚠️ Portfolio return below {portfolio_return_alert}%")
    if sharpe_ratio < min_sharpe:
        alerts.append(f"⚠️ Sharpe ratio below {min_sharpe}")
    if volatility > max_volatility / 100:
        alerts.append(f"⚠️ Volatility above {max_volatility}%")
    return alerts

def check_stock_alerts(data, stock_drop, stock_jump):
    alerts = []
    latest = data.iloc[-1]
    prev = data.iloc[-2]
    pct_changes = (latest - prev) / prev
    for stock, change in pct_changes.items():
        if change <= stock_drop / 100:
            alerts.append(f"⚠️ {stock} dropped {change:.2%} today.")
        elif change >= stock_jump / 100:
            alerts.append(f"🚀 {stock} jumped {change:.2%} today.")
    return alerts

# ----------------- Portfolio Optimizer -----------------
def optimize_portfolio(stocks, start="2020-01-01", end="2025-01-01", risk_free_rate=0.05):
    data = pd.DataFrame()
    for stock in stocks:
        data[stock] = fetch_stock_data(stock, start, end)
    returns = data.pct_change().dropna()
    n = len(stocks)

    def portfolio_metrics(weights):
        weights = np.array(weights)
        port_return = np.sum(weights * returns.mean()) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
        sharpe = (port_return - risk_free_rate)/port_vol
        return port_return, port_vol, sharpe

    def neg_sharpe(weights):
        return -portfolio_metrics(weights)[2]

    constraints = ({'type':'eq', 'fun': lambda w: np.sum(w)-1})
    bounds = tuple((0,1) for _ in range(n))
    init_guess = np.array(n * [1/n])

    result = minimize(neg_sharpe, init_guess, method="SLSQP", bounds=bounds, constraints=constraints)
    weights = result.x
    ret, vol, sharpe = portfolio_metrics(weights)
    return weights, ret, vol, sharpe

# ----------------- Portfolio Simulator -----------------
def simulate_portfolio(stocks, weights, start="2020-01-01", end="2025-01-01"):
    data = pd.DataFrame()
    for stock in stocks:
        data[stock] = fetch_stock_data(stock, start, end)
    weights = np.array(weights)
    returns = data.pct_change().dropna()
    port_return = np.sum(weights * returns.mean()) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
    sharpe = (port_return - 0.05)/port_vol
    growth = (1 + (returns * weights).sum(axis=1)).cumprod()
    return port_return, port_vol, sharpe, growth

# ----------------- Monte Carlo -----------------
def monte_carlo_simulation(stocks, num_portfolios=3000, start="2020-01-01", end="2025-01-01"):
    data = pd.DataFrame()
    for stock in stocks:
        data[stock] = fetch_stock_data(stock, start, end)
    returns = data.pct_change().dropna()
    n = len(stocks)
    results = []
    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights /= np.sum(weights)
        port_return = np.sum(weights * returns.mean()) * 252
        port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov()*252, weights)))
        sharpe = (port_return - 0.05)/port_vol
        results.append([port_return, port_vol, sharpe, weights])
    df = pd.DataFrame(results, columns=["Return","Volatility","Sharpe","Weights"])
    return df

# ----------------- Notifications -----------------
def send_telegram_alert(message):
    try:
        bot_token = st.secrets["telegram"]["bot_token"]
        chat_id = st.secrets["telegram"]["chat_id"]
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message}
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram alert failed:", e)

def send_email_alert(subject, message):
    try:
        sender = st.secrets["email"]["sender"]
        password = st.secrets["email"]["password"]
        receiver = st.secrets["email"]["receiver"]

        msg = MIMEText(message)
        msg['Subject'] = subject
        msg['From'] = sender
        msg['To'] = receiver

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
    except Exception as e:
        print("Email alert failed:", e)
