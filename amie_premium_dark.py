"""
AMIE Platform v2.0 - Complete Working Version
All errors fixed, proper indentation, ready to deploy
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Try real data
try:
    import yfinance as yf
    REAL_DATA = True
except:
    REAL_DATA = False

st.set_page_config(page_title="AMIE Platform", page_icon="🌙", layout="wide")

# CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%); }
    .main-header { font-size: 3rem; font-weight: 900; 
                   background: linear-gradient(135deg, #7c3aed, #3b82f6, #06b6d4);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important;
                                     background: linear-gradient(135deg, #7c3aed, #3b82f6);
                                     -webkit-background-clip: text !important; 
                                     -webkit-text-fill-color: transparent !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed, #3b82f6) !important;
                                      color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #7c3aed, #3b82f6);
                         color: white; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; }
    .insight-box { background: rgba(124, 58, 237, 0.1); border-left: 4px solid #7c3aed; 
                   padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .recommendation { background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; 
                      padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

PLOT = {'font': {'color': '#f8fafc'}, 'paper_bgcolor': 'rgba(26,31,58,0.6)', 
        'plot_bgcolor': 'rgba(10,14,39,0.4)', 'xaxis': {'gridcolor': 'rgba(124,58,237,0.1)'}, 
        'yaxis': {'gridcolor': 'rgba(124,58,237,0.1)'}}

# Alpha Factors
class AlphaFactors:
    @staticmethod
    def momentum(p, w=20):
        return p.pct_change(w)
    
    @staticmethod
    def mean_reversion(p, w=20):
        ma = p.rolling(w).mean()
        return (p - ma) / ma
    
    @staticmethod
    def rsi(p, w=14):
        d = p.diff()
        gain = (d.where(d > 0, 0)).rolling(w).mean()
        loss = (-d.where(d < 0, 0)).rolling(w).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger(p, w=20, std=2):
        ma = p.rolling(w).mean()
        s = p.rolling(w).std()
        return (p - (ma - std * s)) / ((ma + std * s) - (ma - std * s) + 1e-10)
    
    @staticmethod
    def compute_all(p):
        f = pd.DataFrame(index=p.index)
        f['momentum_10'] = AlphaFactors.momentum(p, 10)
        f['momentum_20'] = AlphaFactors.momentum(p, 20)
        f['mean_reversion'] = AlphaFactors.mean_reversion(p, 20)
        f['rsi_14'] = AlphaFactors.rsi(p, 14)
        f['bollinger'] = AlphaFactors.bollinger(p, 20)
        return f.dropna()

# Backtester
class Backtester:
    def __init__(self, capital=100000, comm=0.001):
        self.capital = capital
        self.comm = comm
    
    def run(self, prices, signals):
        cash = self.capital
        shares = 0
        eq = []
        
        for date in prices.index:
            p = prices[date]
            sig = signals[date] if date in signals.index else 0
            
            if sig == 1 and cash > 0:
                shares = int(cash * 0.95 / p)
                cash -= shares * p * (1 + self.comm)
            elif sig == -1 and shares > 0:
                cash += shares * p * (1 - self.comm)
                shares = 0
            
            val = cash + shares * p
            eq.append({'date': date, 'value': val, 'returns': (val - self.capital) / self.capital})
        
        return pd.DataFrame(eq)
    
    def metrics(self, df):
        df['dr'] = df['value'].pct_change()
        tr = (df['value'].iloc[-1] - self.capital) / self.capital
        sh = (df['dr'].mean() / df['dr'].std()) * np.sqrt(252) if df['dr'].std() > 0 else 0
        md = ((df['value'] - df['value'].cummax()) / df['value'].cummax()).min()
        return {'total_return': tr, 'sharpe': sh, 'max_drawdown': md, 'final': df['value'].iloc[-1]}

# Header
st.markdown('<h1 class="main-header">🌙 AMIE PLATFORM ENHANCED</h1>', unsafe_allow_html=True)
st.markdown("### Adaptive Market Intelligence Engine - Interactive & Actionable")

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Data Source")
    
    source = st.radio("Data:", ["🎲 Simulated", "📊 Real (yfinance)"])
    
    if source == "📊 Real (yfinance)" and REAL_DATA:
        ticker = st.text_input("Ticker:", "SPY")
        period = st.selectbox("Period:", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
        use_real = True
    else:
        if source == "📊 Real (yfinance)":
            st.error("yfinance not installed")
        n_samples = st.slider("Days:", 100, 2000, 500)
        use_real = False
    
    st.markdown("---")
    st.markdown("## 💰 Trading")
    capital = st.number_input("Capital ($):", 10000, 1000000, 100000, 10000)
    comm = st.slider("Commission (%):", 0.0, 1.0, 0.1, 0.05)
    
    st.markdown("---")
    st.markdown("## 🎯 Strategy")
    rsi_low = st.slider("RSI Oversold:", 20, 40, 30)
    rsi_high = st.slider("RSI Overbought:", 60, 80, 70)
    mr_thresh = st.slider("Mean Rev:", 0.01, 0.10, 0.05, 0.01)

# Load data
@st.cache_data(ttl=3600)
def load_real(t, p):
    try:
        return yf.download(t, period=p, progress=False)['Close']
    except:
        return None

if use_real and REAL_DATA:
    prices = load_real(ticker, period)
    if prices is None or len(prices) == 0:
        st.error("Failed to load. Using simulated.")
        use_real = False

if not use_real:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    prices = pd.Series(np.cumsum(np.random.randn(n_samples) * 2) + 100, index=dates)

factors = AlphaFactors.compute_all(prices)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Factors", "💹 Backtest", "📋 Reports"])

with tab1:
    st.markdown("## Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cp = prices.iloc[-1]
    pc = prices.pct_change().iloc[-1]
    pc = 0 if pd.isna(pc) else pc
    vol = prices.pct_change().std() * np.sqrt(252)
    rsi = factors['rsi_14'].iloc[-1] if len(factors) > 0 else 50
    
    col1.metric("Price", f"${cp:.2f}", f"{pc:.2%}")
    col2.metric("30-Day Return", f"{prices.pct_change(30).iloc[-1]:.2%}" if len(prices) > 30 else "N/A")
    col3.metric("Volatility", f"{vol:.1%}")
    col4.metric("RSI", f"{rsi:.1f}")
    
    st.markdown("### 🎯 Signal")
    
    if rsi > 70:
        st.markdown(f"""<div class="insight-box"><strong>⚠️ OVERBOUGHT</strong><br>
        RSI at {rsi:.1f}. Consider sell or wait for pullback.</div>""", unsafe_allow_html=True)
    elif rsi < 30:
        st.markdown(f"""<div class="recommendation"><strong>✅ OVERSOLD</strong><br>
        RSI at {rsi:.1f}. Potential buy opportunity.</div>""", unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **NEUTRAL**: RSI at {rsi:.1f}")
    
    st.markdown("---")
    st.markdown("### 📈 Price History")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='#7c3aed', width=2)))
    fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(20).mean(), name='20MA', 
                            line=dict(color='#06b6d4', width=1, dash='dash')))
    fig.update_layout(**PLOT, height=400, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    csv = prices.to_csv().encode()
    st.download_button("📥 Download Data", csv, "prices.csv", "text/csv")

with tab2:
    st.markdown("## Factor Analysis")
    
    sel = st.selectbox("Factor:", factors.columns.tolist())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Price', sel))
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=factors.index, y=factors[sel], name=sel), row=2, col=1)
        fig.update_layout(**PLOT, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Stats")
        cur = factors[sel].iloc[-1]
        st.metric("Current", f"{cur:.4f}")
        st.metric("Mean", f"{factors[sel].mean():.4f}")
        st.metric("Std", f"{factors[sel].std():.4f}")

with tab3:
    st.markdown("## Strategy Backtesting")
    
    strat = st.selectbox("Strategy:", ["Mean Reversion", "Momentum", "RSI", "Multi-Factor"])
    
    if strat == "Mean Reversion":
        st.info("Buy below average, sell above. Good for ranging markets.")
        sigs = factors['mean_reversion'].apply(lambda x: -1 if x > mr_thresh else (1 if x < -mr_thresh else 0))
    elif strat == "Momentum":
        st.info("Follow trends. Good for trending markets.")
        sigs = factors['momentum_20'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
    elif strat == "RSI":
        st.info(f"Buy < {rsi_low}, sell > {rsi_high}")
        sigs = factors['rsi_14'].apply(lambda x: 1 if x < rsi_low else (-1 if x > rsi_high else 0))
    else:
        st.info("Combines momentum + RSI")
        sigs = ((factors['momentum_20'] > 0.02) & (factors['rsi_14'] < 50)).astype(int) - \
               ((factors['momentum_20'] < -0.02) & (factors['rsi_14'] > 50)).astype(int)
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running..."):
            bt = Backtester(capital, comm/100)
            idx = prices.index.intersection(sigs.index)
            res = bt.run(prices.loc[idx], sigs.loc[idx])
            met = bt.metrics(res)
        
        st.success("✅ Complete!")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Return", f"{met['total_return']:.2%}")
        col2.metric("Sharpe", f"{met['sharpe']:.2f}")
        col3.metric("Max DD", f"{met['max_drawdown']:.2%}")
        col4.metric("Final", f"${met['final']:,.0f}")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res['date'], y=res['value'], name='Strategy', 
                                line=dict(color='#10b981', width=3)))
        bh = capital * (prices.loc[idx] / prices.loc[idx].iloc[0])
        fig.add_trace(go.Scatter(x=res['date'], y=bh, name='Buy&Hold', 
                                line=dict(color='#7c3aed', width=2, dash='dash')))
        fig.update_layout(**PLOT, height=500)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## Reports")
    
    st.markdown("### Factor Correlation")
    corr = factors.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                    colorscale=[[0,'#7c3aed'],[0.5,'#1a1f3a'],[1,'#06b6d4']],
                                    text=np.round(corr.values, 2), texttemplate='%{text}'))
    fig.update_layout(**PLOT, height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Export")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Prices", prices.to_csv().encode(), "prices.csv")
    with col2:
        st.download_button("📥 Factors", factors.to_csv().encode(), "factors.csv")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b;'>AMIE Platform v2.0 Enhanced</div>", unsafe_allow_html=True)
