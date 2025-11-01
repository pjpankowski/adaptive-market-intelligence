"""
AMIE Platform v2.0 - Chart Fixed Version
Beautiful charts for real and simulated data
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    REAL_DATA = True
except:
    REAL_DATA = False

st.set_page_config(page_title="AMIE Platform", page_icon="🌙", layout="wide")

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

def safe_float(value, default=0.0):
    """Safely convert any value to float"""
    try:
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except:
        return default

def clean_series_for_plot(series):
    """Clean pandas series for Plotly - remove timezone, NaN, and ensure proper format"""
    clean = series.copy().dropna()
    # Remove timezone if present
    if hasattr(clean.index, 'tz') and clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)
    return clean

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

class Backtester:
    def __init__(self, capital=100000, comm=0.001):
        self.capital = capital
        self.comm = comm
    
    def run(self, prices, signals):
        cash = self.capital
        shares = 0
        eq = []
        
        for date in prices.index:
            p = safe_float(prices[date], 100)
            sig = int(signals[date]) if date in signals.index else 0
            
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

st.markdown('<h1 class="main-header">🌙 AMIE PLATFORM</h1>', unsafe_allow_html=True)
st.markdown("### Adaptive Market Intelligence Engine")

with st.sidebar:
    st.markdown("## ⚙️ Data Source")
    source = st.radio("Data:", ["🎲 Simulated", "📊 Real Stock Data"])
    
    if source == "📊 Real Stock Data" and REAL_DATA:
        ticker = st.text_input("Ticker:", "SPY")
        period = st.selectbox("Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        use_real = True
    else:
        if source == "📊 Real Stock Data":
            st.error("⚠️ yfinance not installed")
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

@st.cache_data(ttl=3600)
def load_real(t, p):
    try:
        data = yf.download(t, period=p, progress=False)
        if len(data) > 0:
            prices = data['Close']
            # Clean timezone info
            if hasattr(prices.index, 'tz') and prices.index.tz is not None:
                prices.index = prices.index.tz_localize(None)
            return prices
        return None
    except:
        return None

if use_real and REAL_DATA:
    with st.spinner(f"Loading {ticker}..."):
        prices = load_real(ticker, period)
        if prices is None or len(prices) == 0:
            st.error("Failed to load. Using simulated data.")
            use_real = False
        else:
            st.success(f"✅ Loaded {len(prices)} days of {ticker}")

if not use_real:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    prices = pd.Series(np.cumsum(np.random.randn(n_samples) * 2) + 100, index=dates)

factors = AlphaFactors.compute_all(prices)

tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Factors", "💹 Backtest", "📋 Reports"])

with tab1:
    st.markdown("## Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    cp = safe_float(prices.iloc[-1], 100)
    pc = safe_float(prices.pct_change().iloc[-1], 0)
    vol = safe_float(prices.pct_change().std() * np.sqrt(252), 0)
    rsi = safe_float(factors['rsi_14'].iloc[-1] if len(factors) > 0 else 50, 50)
    
    col1.metric("Price", f"${cp:.2f}", f"{pc:.2%}")
    
    ret_30 = safe_float(prices.pct_change(30).iloc[-1] if len(prices) > 30 else 0, 0)
    col2.metric("30-Day Return", f"{ret_30:.2%}" if ret_30 != 0 else "N/A")
    
    col3.metric("Volatility", f"{vol:.1%}")
    col4.metric("RSI", f"{rsi:.1f}")
    
    st.markdown("### 🎯 Current Signal")
    
    if rsi > 70:
        st.markdown(f"""<div class="insight-box"><strong>⚠️ OVERBOUGHT</strong><br>
        RSI at {rsi:.1f}. Consider profit-taking or waiting for pullback.</div>""", unsafe_allow_html=True)
    elif rsi < 30:
        st.markdown(f"""<div class="recommendation"><strong>✅ OVERSOLD</strong><br>
        RSI at {rsi:.1f}. Potential buying opportunity.</div>""", unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **NEUTRAL**: RSI at {rsi:.1f} is in normal range")
    
    st.markdown("---")
    st.markdown("### 📈 Price Chart with Moving Averages")
    
    # Clean data for plotting
    plot_prices = clean_series_for_plot(prices)
    ma20 = plot_prices.rolling(20).mean().dropna()
    ma50 = plot_prices.rolling(50).mean().dropna()
    
    fig = go.Figure()
    
    # Price line
    fig.add_trace(go.Scatter(
        x=plot_prices.index, 
        y=plot_prices.values, 
        name='Price', 
        line=dict(color='#7c3aed', width=2.5),
        mode='lines'
    ))
    
    # 20-day MA
    if len(ma20) > 0:
        fig.add_trace(go.Scatter(
            x=ma20.index, 
            y=ma20.values, 
            name='20-Day MA', 
            line=dict(color='#06b6d4', width=1.5, dash='dash'),
            mode='lines'
        ))
    
    # 50-day MA
    if len(ma50) > 0:
        fig.add_trace(go.Scatter(
            x=ma50.index, 
            y=ma50.values, 
            name='50-Day MA', 
            line=dict(color='#ec4899', width=1.5, dash='dot'),
            mode='lines'
        ))
    
    fig.update_layout(
        **PLOT, 
        height=500, 
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Price ($)",
        showlegend=True,
        legend=dict(
            x=0.01, 
            y=0.99, 
            bgcolor='rgba(26,31,58,0.8)',
            bordercolor='#7c3aed',
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Format axis
    fig.update_xaxes(showgrid=True, gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridwidth=0.5, tickformat='$,.2f')
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        csv = prices.to_csv().encode()
        st.download_button("📥 Download Price Data", csv, "prices.csv", use_container_width=True)
    with col2:
        st.markdown(f"**Data Points:** {len(prices)} | **Date Range:** {plot_prices.index[0].date()} to {plot_prices.index[-1].date()}")

with tab2:
    st.markdown("## Factor Analysis")
    
    if len(factors) == 0:
        st.warning("Need at least 60 days of data to compute factors")
    else:
        sel = st.selectbox("Select Factor:", factors.columns.tolist())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Clean data for plotting
            plot_prices_clean = clean_series_for_plot(prices)
            plot_factors_clean = clean_series_for_plot(factors[sel])
            
            fig = make_subplots(
                rows=2, cols=1, 
                subplot_titles=('Price Movement', f'Factor: {sel.replace("_", " ").title()}'),
                vertical_spacing=0.15,
                row_heights=[0.5, 0.5]
            )
            
            fig.add_trace(go.Scatter(
                x=plot_prices_clean.index, 
                y=plot_prices_clean.values, 
                name='Price',
                line=dict(color='#7c3aed', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=plot_factors_clean.index, 
                y=plot_factors_clean.values, 
                name=sel,
                line=dict(color='#06b6d4', width=2),
                fill='tozeroy',
                fillcolor='rgba(6, 182, 212, 0.1)'
            ), row=2, col=1)
            
            fig.update_layout(**PLOT, height=700, showlegend=False)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickformat='$,.2f')
            fig.update_yaxes(title_text="Factor Value", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Statistics")
            cur = safe_float(factors[sel].iloc[-1])
            mean = safe_float(factors[sel].mean())
            std = safe_float(factors[sel].std())
            
            st.metric("Current Value", f"{cur:.4f}")
            st.metric("Mean", f"{mean:.4f}")
            st.metric("Std Dev", f"{std:.4f}")
            
            if std > 0:
                z = (cur - mean) / std
                st.metric("Z-Score", f"{z:.2f}")
                
                if abs(z) > 2:
                    st.warning("⚠️ **Extreme Value** (|Z| > 2)")
                elif abs(z) > 1:
                    st.info("ℹ️ **Notable Deviation**")
                else:
                    st.success("✅ **Normal Range**")

with tab3:
    st.markdown("## Strategy Backtesting")
    
    strat = st.selectbox("Strategy:", ["Mean Reversion", "Momentum", "RSI", "Multi-Factor"])
    
    if len(factors) == 0:
        st.warning("Not enough data for backtesting")
    else:
        if strat == "Mean Reversion":
            st.info("**Buy** when price is far below average | **Sell** when far above")
            sigs = factors['mean_reversion'].apply(lambda x: -1 if x > mr_thresh else (1 if x < -mr_thresh else 0))
        elif strat == "Momentum":
            st.info("**Buy** uptrends | **Sell** downtrends")
            sigs = factors['momentum_20'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
        elif strat == "RSI":
            st.info(f"**Buy** when RSI < {rsi_low} | **Sell** when RSI > {rsi_high}")
            sigs = factors['rsi_14'].apply(lambda x: 1 if x < rsi_low else (-1 if x > rsi_high else 0))
        else:
            st.info("**Combined** momentum and RSI signals")
            sigs = ((factors['momentum_20'] > 0.02) & (factors['rsi_14'] < 50)).astype(int) - \
                   ((factors['momentum_20'] < -0.02) & (factors['rsi_14'] > 50)).astype(int)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running backtest..."):
                bt = Backtester(capital, comm/100)
                idx = prices.index.intersection(sigs.index)
                if len(idx) > 0:
                    res = bt.run(prices.loc[idx], sigs.loc[idx])
                    met = bt.metrics(res)
                    
                    st.success("✅ Backtest Complete!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Total Return", f"{met['total_return']:.2%}")
                    col2.metric("Sharpe Ratio", f"{met['sharpe']:.2f}")
                    col3.metric("Max Drawdown", f"{met['max_drawdown']:.2%}")
                    col4.metric("Final Value", f"${met['final']:,.0f}")
                    
                    st.markdown("### 📈 Performance vs Buy & Hold")
                    
                    # Clean data for plotting
                    res_clean = res.copy()
                    if hasattr(res_clean['date'].iloc[0], 'tz'):
                        res_clean['date'] = pd.to_datetime(res_clean['date']).dt.tz_localize(None)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=res_clean['date'], 
                        y=res_clean['value'], 
                        name='Strategy', 
                        line=dict(color='#10b981', width=3),
                        fill='tozeroy',
                        fillcolor='rgba(16, 185, 129, 0.1)'
                    ))
                    
                    bh = capital * (prices.loc[idx] / prices.loc[idx].iloc[0])
                    fig.add_trace(go.Scatter(
                        x=res_clean['date'], 
                        y=bh.values, 
                        name='Buy & Hold', 
                        line=dict(color='#7c3aed', width=2, dash='dash')
                    ))
                    
                    fig.update_layout(
                        **PLOT, 
                        height=500, 
                        hovermode='x unified',
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        showlegend=True,
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(26,31,58,0.8)')
                    )
                    fig.update_yaxes(tickformat='$,.0f')
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    csv_res = res.to_csv(index=False).encode()
                    st.download_button("📥 Download Results", csv_res, "backtest.csv")

with tab4:
    st.markdown("## Analysis Reports")
    
    if len(factors) > 0:
        st.markdown("### Factor Correlation Matrix")
        corr = factors.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, 
            x=corr.columns, 
            y=corr.columns,
            colorscale=[[0,'#7c3aed'],[0.5,'#1a1f3a'],[1,'#06b6d4']],
            text=np.round(corr.values, 2), 
            texttemplate='%{text}',
            textfont={"size": 12, "color": '#f8fafc'},
            hoverongaps=False
        ))
        
        fig.update_layout(**PLOT, height=500, xaxis_showgrid=False, yaxis_showgrid=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button("📥 Price Data (CSV)", prices.to_csv().encode(), "prices.csv", use_container_width=True)
    
    with col2:
        if len(factors) > 0:
            st.download_button("📥 Factor Data (CSV)", factors.to_csv().encode(), "factors.csv", use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:#64748b;'>AMIE Platform v2.0 Enhanced | Professional-Grade Analytics</div>", unsafe_allow_html=True)
