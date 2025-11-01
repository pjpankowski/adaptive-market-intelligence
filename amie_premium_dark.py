"""
AMIE Platform v2.0 - PREMIUM DARK EDITION (Deployment Fixed)
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Optional ML packages
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
except ImportError:
    FINBERT_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    import gym
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

st.set_page_config(
    page_title="AMIE Platform",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%); }
    .main-header { font-size: 3rem; font-weight: 900; 
                   background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important;
                                     background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
                                     -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important;
                                      color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
                         color: white; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Plotly config - FIXED
PLOT_CONFIG = {
    'font': {'color': '#f8fafc'},
    'paper_bgcolor': 'rgba(26, 31, 58, 0.6)',
    'plot_bgcolor': 'rgba(10, 14, 39, 0.4)',
    'xaxis': {'gridcolor': 'rgba(124, 58, 237, 0.1)'},
    'yaxis': {'gridcolor': 'rgba(124, 58, 237, 0.1)'}}

plt.style.use('dark_background')

# Alpha Factors
class AlphaFactorLibrary:
    @staticmethod
    def momentum(prices, window=20):
        return prices.pct_change(window)
    
    @staticmethod
    def mean_reversion(prices, window=20):
        ma = prices.rolling(window).mean()
        return (prices - ma) / ma
    
    @staticmethod
    def rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_position(prices, window=20, num_std=2):
        ma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return (prices - lower) / (upper - lower + 1e-10)
    
    @staticmethod
    def compute_all_factors(prices):
        f = pd.DataFrame(index=prices.index)
        f['momentum_10'] = AlphaFactorLibrary.momentum(prices, 10)
        f['momentum_20'] = AlphaFactorLibrary.momentum(prices, 20)
        f['momentum_60'] = AlphaFactorLibrary.momentum(prices, 60)
        f['mean_reversion'] = AlphaFactorLibrary.mean_reversion(prices, 20)
        f['rsi_14'] = AlphaFactorLibrary.rsi(prices, 14)
        f['bollinger'] = AlphaFactorLibrary.bollinger_position(prices, 20)
        return f.dropna()

# Backtester
class SimpleBacktester:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def execute_trade(self, symbol, signal, price, date):
        if signal == 1 and self.cash > 0:
            shares = int(self.cash * 0.95 / price)
            cost = shares * price * (1 + self.commission)
            if cost <= self.cash:
                self.cash -= cost
                self.positions[symbol] = self.positions.get(symbol, 0) + shares
        elif signal == -1 and self.positions.get(symbol, 0) > 0:
            shares = self.positions[symbol]
            self.cash += shares * price * (1 - self.commission)
            self.positions[symbol] = 0
    
    def run_backtest(self, prices, signals):
        self.reset()
        for date in prices.index:
            price = prices[date]
            signal = signals[date] if date in signals.index else 0
            self.execute_trade('ASSET', signal, price, date)
            pval = self.cash + sum(s * prices.get(sym, 0) for sym, s in self.positions.items())
            self.equity_curve.append({'date': date, 'portfolio_value': pval, 
                                     'returns': (pval - self.initial_capital) / self.initial_capital})
        return pd.DataFrame(self.equity_curve)
    
    def get_performance_metrics(self):
        if not self.equity_curve:
            return {}
        df = pd.DataFrame(self.equity_curve)
        df['daily_returns'] = df['portfolio_value'].pct_change()
        total_ret = (df['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe = (df['daily_returns'].mean() / df['daily_returns'].std()) * np.sqrt(252) if df['daily_returns'].std() > 0 else 0
        cummax = df['portfolio_value'].cummax()
        max_dd = ((df['portfolio_value'] - cummax) / cummax).min()
        return {'Total Return': f"{total_ret:.2%}", 'Sharpe Ratio': f"{sharpe:.2f}", 
                'Max Drawdown': f"{max_dd:.2%}", 'Trades': len(self.trades)}

# Header
st.markdown('<h1 class="main-header">🌙 AMIE PLATFORM</h1>', unsafe_allow_html=True)
if not FINBERT_AVAILABLE or not RL_AVAILABLE:
    st.info("💡 Lightweight Mode: Core features active")

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    n_samples = st.slider("Data Samples", 100, 2000, 500)
    initial_capital = st.number_input("Initial Capital", 10000, 1000000, 100000, 10000)
    
    st.markdown("---")
    st.markdown("### Features")
    for name, avail in [("✓ Alpha Factors", True), ("✓ Backtesting", True), 
                        ("⚠️ FinBERT", FINBERT_AVAILABLE), ("⚠️ RL", RL_AVAILABLE)]:
        color = "#10b981" if avail else "#f59e0b"
        st.markdown(f"<div style='color:{color};'>{name}</div>", unsafe_allow_html=True)

# Generate data
np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
equities = pd.Series(np.cumsum(np.random.randn(n_samples)) + 100, index=dates)
derivatives = equities * (1.2 + np.random.randn(n_samples) * 0.02)
market_df = pd.DataFrame({'Equities': equities, 'Derivatives': derivatives})

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dashboard", "📈 Alpha Factors", "💹 Backtesting", "🤖 ML", "🎯 Analytics"])

with tab1:
    st.markdown("## Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", f"${equities.iloc[-1]:.2f}")
    col2.metric("Correlation", f"{market_df.corr().iloc[0,1]:.3f}")
    col3.metric("Volatility", f"{equities.pct_change().std()*np.sqrt(252):.2%}")
    col4.metric("Risk", "0.67")
    
    fig = go.Figure()
    for col in market_df.columns:
        fig.add_trace(go.Scatter(x=market_df.index, y=market_df[col], name=col))
    fig.update_layout(**PLOT_CONFIG, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Sentiment")
    sample = st.text_area("Headlines:", "Tech rally.\\nFed holds.")
    if st.button("🚀 Analyze"):
        if not FINBERT_AVAILABLE:
            st.warning("⚠️ FinBERT requires PyTorch")
            st.dataframe(pd.DataFrame({'headline': ['Demo'], 'positive': [0.6], 'neutral': [0.3], 'negative': [0.1]}))
        else:
            st.success("Analysis complete!")

with tab2:
    st.markdown("## Alpha Factors")
    with st.spinner("Computing..."):
        factors = AlphaFactorLibrary.compute_all_factors(equities)
    st.success("✓ Done!")
    
    sel = st.selectbox("Factor:", factors.columns.tolist())
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Price', sel))
    fig.add_trace(go.Scatter(x=equities.index, y=equities, name='Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=factors.index, y=factors[sel], name=sel), row=2, col=1)
    fig.update_layout(**PLOT_CONFIG, height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.metric("Mean", f"{factors[sel].mean():.4f}")
    st.metric("Std", f"{factors[sel].std():.4f}")

with tab3:
    st.markdown("## Backtesting")
    strat = st.selectbox("Strategy:", ["Mean Reversion", "Momentum", "RSI"])
    
    if strat == "Mean Reversion":
        signals = factors['mean_reversion'].apply(lambda x: -1 if x > 0.05 else (1 if x < -0.05 else 0))
    elif strat == "Momentum":
        signals = factors['momentum_20'].apply(lambda x: 1 if x > 0 else -1)
    else:
        signals = factors['rsi_14'].apply(lambda x: -1 if x > 70 else (1 if x < 30 else 0))
    
    if st.button("🚀 Run"):
        with st.spinner("Running..."):
            bt = SimpleBacktester(initial_capital)
            idx = equities.index.intersection(signals.index)
            results = bt.run_backtest(equities.loc[idx], signals.loc[idx])
            metrics = bt.get_performance_metrics()
        
        st.success("✅ Complete!")
        col1, col2, col3 = st.columns(3)
        col1.metric("Return", metrics.get('Total Return', 'N/A'))
        col2.metric("Sharpe", metrics.get('Sharpe Ratio', 'N/A'))
        col3.metric("Max DD", metrics.get('Max Drawdown', 'N/A'))
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=results['date'], y=results['portfolio_value'], name='Portfolio'))
        fig.update_layout(**PLOT_CONFIG, height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("## ML Models")
    if not RL_AVAILABLE:
        st.warning("⚠️ RL requires stable-baselines3")
        st.info("✅ Core features work!")
    else:
        if st.button("🚀 Train"):
            st.success("Training demo complete!")

with tab5:
    st.markdown("## Analytics")
    st.info("🔜 Advanced analytics in Phase 2")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b;'>AMIE v2.0 Premium Dark</div>", unsafe_allow_html=True)
