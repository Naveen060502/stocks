import numpy as np
import pandas as pd
import yfinance as yf
import requests

# ----------------- Backtest / Portfolio -----------------
def backtest_with_trades(stock, start="2020-01-01", end="2025-01-01"):
    data = yf.download(stock, start=start, end=end)["Adj Close"].dropna()
    df = pd.DataFrame(index=data.index)
    df["Close"] = data
    df["Strategy_Return"] = df["Close"].pct_change()  # Simple strategy placeholder
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

# ----------------- Metrics -----------------
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
    return {"CAGR %": round(cagr,2), "Sharpe": round(sharpe,2), "Max DD %": round(max_dd,2), "Total Return %": round(total_return,2)}

# ----------------- Alerts -----------------
def check_portfolio_alerts(portfolio_return, sharpe_ratio, volatility,
                           min_sharpe, max_volatility, portfolio_return_alert):
    alerts = []
    if portfolio_return < portfolio_return_alert / 100:
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

# ----------------- Live Data -----------------
def fetch_live_data(stocks):
    data = yf.download(stocks, period="2d", interval="15m")["Adj Close"]
    data = data.groupby(data.index.date).last()
    return data
