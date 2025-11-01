"""
AMIE Platform v2.0 - ENHANCED & ROBUST VERSION
Fully interactive with real data and actionable insights
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

# Try to import yfinance for real data
try:
    import yfinance as yf
    REAL_DATA_AVAILABLE = True
except ImportError:
    REAL_DATA_AVAILABLE = False

st.set_page_config(page_title="AMIE Platform", page_icon="🌙", layout="wide")

# Premium CSS
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%); }
    .main-header { font-size: 3rem; font-weight: 900; 
                   background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    [data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 700 !important;
                                     background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
                                     -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%) !important; color: white !important; }
    .stButton > button { background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%); color: white; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600; }
    .insight-box { background: rgba(124, 58, 237, 0.1); border-left: 4px solid #7c3aed; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
    .recommendation { background: rgba(16, 185, 129, 0.1); border-left: 4px solid #10b981; padding: 1rem; border-radius: 8px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

PLOT_CONFIG = {'font': {'color': '#f8fafc'}, 'paper_bgcolor': 'rgba(26, 31, 58, 0.6)', 
               'plot_bgcolor': 'rgba(10, 14, 39, 0.4)', 'xaxis': {'gridcolor': 'rgba(124, 58, 237, 0.1)'}, 
               'yaxis': {'gridcolor': 'rgba(124, 58, 237, 0.1)'}}

# Alpha Factors
class AlphaFactors:
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
    def compute_all(prices):
        f = pd.DataFrame(index=prices.index)
        f['momentum_10'] = AlphaFactors.momentum(prices, 10)
        f['momentum_20'] = AlphaFactors.momentum(prices, 20)
        f['mean_reversion'] = AlphaFactors.mean_reversion(prices, 20)
        f['rsi_14'] = AlphaFactors.rsi(prices, 14)
        f['bollinger'] = AlphaFactors.bollinger_position(prices, 20)
        return f.dropna()

# Backtester
class Backtester:
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
    
    def run(self, prices, signals):
        cash = self.initial_capital
        shares = 0
        equity = []
        
        for date in prices.index:
            price = prices[date]
            signal = signals[date] if date in signals.index else 0
            
            if signal == 1 and cash > 0:
                shares = int(cash * 0.95 / price)
                cash -= shares * price * (1 + self.commission)
            elif signal == -1 and shares > 0:
                cash += shares * price * (1 - self.commission)
                shares = 0
            
            portfolio_val = cash + shares * price
            equity.append({'date': date, 'value': portfolio_val, 'returns': (portfolio_val - self.initial_capital) / self.initial_capital})
        
        return pd.DataFrame(equity)
    
    def metrics(self, equity_df):
        equity_df['daily_ret'] = equity_df['value'].pct_change()
        total_ret = (equity_df['value'].iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe = (equity_df['daily_ret'].mean() / equity_df['daily_ret'].std()) * np.sqrt(252) if equity_df['daily_ret'].std() > 0 else 0
        max_dd = ((equity_df['value'] - equity_df['value'].cummax()) / equity_df['value'].cummax()).min()
        return {'total_return': total_ret, 'sharpe': sharpe, 'max_drawdown': max_dd, 
                'final_value': equity_df['value'].iloc[-1], 'num_days': len(equity_df)}

# Header
st.markdown('<h1 class="main-header">🌙 AMIE PLATFORM - ENHANCED</h1>', unsafe_allow_html=True)
st.markdown("### Adaptive Market Intelligence Engine - **Now with Real Data & Actionable Insights**")

# Sidebar - ENHANCED with real controls
with st.sidebar:
    st.markdown("## ⚙️ Data Source")
    
    data_source = st.radio("Choose data:", ["📊 Real Market Data (yfinance)", "🎲 Simulated Data"])
    
    if data_source == "📊 Real Market Data (yfinance)":
        if REAL_DATA_AVAILABLE:
            ticker = st.text_input("Enter Ticker Symbol:", value="SPY", help="e.g., SPY, AAPL, TSLA, BTC-USD")
            period = st.selectbox("Time Period:", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
            use_real_data = True
        else:
            st.error("⚠️ yfinance not installed. Add to requirements.txt")
            use_real_data = False
            n_samples = 500
    else:
        use_real_data = False
        n_samples = st.slider("Simulated Days:", 100, 2000, 500)
    
    st.markdown("---")
    st.markdown("## 💰 Trading Parameters")
    initial_capital = st.number_input("Initial Capital ($):", 10000, 1000000, 100000, 10000)
    commission_pct = st.slider("Commission (%):", 0.0, 1.0, 0.1, 0.05)
    
    st.markdown("---")
    st.markdown("## 🎯 Strategy Settings")
    rsi_oversold = st.slider("RSI Oversold Threshold:", 20, 40, 30)
    rsi_overbought = st.slider("RSI Overbought Threshold:", 60, 80, 70)
    mean_rev_threshold = st.slider("Mean Reversion Threshold:", 0.01, 0.10, 0.05, 0.01)

# Load data
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    try:
        data = yf.download(ticker, period=period, progress=False)
        return data['Close']
    except:
        return None

if use_real_data and REAL_DATA_AVAILABLE:
    with st.spinner(f"Loading {ticker} data..."):
        prices = load_data(ticker, period)
        if prices is not None and len(prices) > 0:
            st.success(f"✅ Loaded {len(prices)} days of {ticker} data")
            data_type = f"Real: {ticker}"
        else:
            st.error("Failed to load data. Using simulated data.")
            use_real_data = False
else:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    prices = pd.Series(np.cumsum(np.random.randn(n_samples) * 2) + 100, index=dates)
    data_type = "Simulated"

# Compute factors
factors = AlphaFactors.compute_all(prices)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Market Overview", "📈 Factor Analysis", "💹 Strategy Backtesting", "📋 Reports"])

# TAB 1: Market Overview with REAL insights
with tab1:
    st.markdown("## 📊 Market Overview & Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    current_price = prices.iloc[-1]
    price_change = prices.pct_change().iloc[-1]
    volatility = prices.pct_change().std() * np.sqrt(252)
    current_rsi = factors['rsi_14'].iloc[-1]
    
    col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:.2%}")
    col2.metric("30-Day Return", f"{prices.pct_change(30).iloc[-1]:.2%}")
    col3.metric("Volatility (Annual)", f"{volatility:.1%}")
    col4.metric("RSI (14)", f"{current_rsi:.1f}")
    
    # ACTIONABLE INSIGHTS
    st.markdown("### 🎯 Current Market Signal")
    
    if current_rsi > 70:
        st.markdown(f"""
        <div class="insight-box">
        <strong>⚠️ OVERBOUGHT SIGNAL</strong><br>
        RSI at {current_rsi:.1f} (above 70) suggests the asset may be overvalued.<br>
        <strong>Consideration:</strong> Potential sell opportunity or wait for pullback.
        </div>
        """, unsafe_allow_html=True)
    elif current_rsi < 30:
        st.markdown(f"""
        <div class="recommendation">
        <strong>✅ OVERSOLD SIGNAL</strong><br>
        RSI at {current_rsi:.1f} (below 30) suggests the asset may be undervalued.<br>
        <strong>Consideration:</strong> Potential buy opportunity.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **NEUTRAL**: RSI at {current_rsi:.1f} is in normal range (30-70). No strong signal.")
    
    st.markdown("---")
    
    # Interactive price chart
    st.markdown("### 📈 Price History")
    
    chart_type = st.radio("Chart Type:", ["Line", "Candlestick (with Moving Averages)"], horizontal=True)
    
    if chart_type == "Line":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='#7c3aed', width=2)))
        fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(20).mean(), name='20-day MA', 
                                line=dict(color='#06b6d4', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(50).mean(), name='50-day MA', 
                                line=dict(color='#ec4899', width=1, dash='dash')))
    else:
        # Candlestick chart (simplified with OHLC simulation)
        df = pd.DataFrame(index=prices.index)
        df['close'] = prices
        df['open'] = prices.shift(1)
        df['high'] = prices.rolling(2).max()
        df['low'] = prices.rolling(2).min()
        
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['open'], high=df['high'], 
                                             low=df['low'], close=df['close'], name='OHLC')])
        fig.add_trace(go.Scatter(x=prices.index, y=prices.rolling(20).mean(), name='20-day MA', 
                                line=dict(color='#06b6d4', width=2)))
    
    fig.update_layout(**PLOT_CONFIG, height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Download data
    col1, col2 = st.columns(2)
    with col1:
        csv = prices.to_csv().encode('utf-8')
        st.download_button("📥 Download Price Data (CSV)", csv, "prices.csv", "text/csv")
    with col2:
        factor_csv = factors.to_csv().encode('utf-8')
        st.download_button("📥 Download Factors (CSV)", factor_csv, "factors.csv", "text/csv")

# TAB 2: Factor Analysis with EXPLANATIONS
with tab2:
    st.markdown("## 📈 Alpha Factor Analysis")
    
    st.markdown("""
    **What are Alpha Factors?** They're quantitative signals that may predict future price movements.
    High-quality factors have strong predictive power (high IC).
    """)
    
    selected = st.selectbox("Select Factor to Analyze:", factors.columns.tolist())
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{data_type} Price', f'Factor: {selected}'), 
                           vertical_spacing=0.15, row_heights=[0.5, 0.5])
        
        fig.add_trace(go.Scatter(x=prices.index, y=prices, name='Price', line=dict(color='#7c3aed', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=factors.index, y=factors[selected], name=selected, 
                                line=dict(color='#06b6d4', width=2), fill='tozeroy'), row=2, col=1)
        
        fig.update_layout(**PLOT_CONFIG, height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Factor Statistics")
        
        current_val = factors[selected].iloc[-1]
        mean_val = factors[selected].mean()
        std_val = factors[selected].std()
        
        st.metric("Current Value", f"{current_val:.4f}")
        st.metric("Mean", f"{mean_val:.4f}")
        st.metric("Std Dev", f"{std_val:.4f}")
        st.metric("Z-Score", f"{(current_val - mean_val) / std_val:.2f}")
        
        # Predictive power
        forward_ret = prices.pct_change(5).shift(-5)
        valid_idx = factors.index.intersection(forward_ret.index)
        
        if len(valid_idx) > 20:
            fv = factors.loc[valid_idx, selected]
            fr = forward_ret.loc[valid_idx]
            mask = ~(fv.isna() | fr.isna())
            
            if mask.sum() > 20:
                ic, pval = pearsonr(fv[mask], fr[mask])
                st.metric("IC (5-day)", f"{ic:.4f}")
                
                if abs(ic) > 0.1:
                    st.success("✅ Strong Predictor!")
                elif abs(ic) > 0.05:
                    st.info("ℹ️ Moderate Predictor")
                else:
                    st.warning("⚠️ Weak Signal")
        
        # Interpretation
        st.markdown("### 💡 What This Means")
        
        if selected.startswith('momentum'):
            if current_val > 0.05:
                st.success("📈 **Strong Upward Momentum** - Trend is bullish")
            elif current_val < -0.05:
                st.error("📉 **Strong Downward Momentum** - Trend is bearish")
            else:
                st.info("➡️ **Neutral Momentum** - No clear trend")
        
        elif selected == 'rsi_14':
            if current_val > 70:
                st.warning("⚠️ **Overbought** - Possible reversal coming")
            elif current_val < 30:
                st.success("✅ **Oversold** - Possible bounce coming")
            else:
                st.info("ℹ️ **Normal Range** - No extreme condition")
        
        elif selected == 'mean_reversion':
            if abs(current_val) > 0.05:
                st.warning("⚠️ **Far from Mean** - Likely to revert")
            else:
                st.info("ℹ️ **Near Mean** - Stable price")

# Continued in next message due to length...
# TAB 3: Interactive Strategy Backtesting
with tab3:
    st.markdown("## 💹 Interactive Strategy Backtesting")
    
    st.markdown("""
    **Test trading strategies** on historical data to see if they would have been profitable.
    Adjust parameters in the sidebar to see how they affect performance.
    """)
    
    # Strategy selection with descriptions
    strategy = st.selectbox("Select Trading Strategy:", 
                           ["Mean Reversion", "Momentum Trend Following", "RSI Oscillator", "Custom Multi-Factor"])
    
    if strategy == "Mean Reversion":
        st.info("**Strategy:** Buy when price is far below average, sell when far above. Works best in ranging markets.")
        signals = factors['mean_reversion'].apply(lambda x: -1 if x > mean_rev_threshold else (1 if x < -mean_rev_threshold else 0))
    
    elif strategy == "Momentum Trend Following":
        st.info("**Strategy:** Buy uptrends, sell downtrends. Works best in trending markets.")
        signals = factors['momentum_20'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
    
    elif strategy == "RSI Oscillator":
        st.info(f"**Strategy:** Buy when RSI < {rsi_oversold} (oversold), sell when RSI > {rsi_overbought} (overbought).")
        signals = factors['rsi_14'].apply(lambda x: 1 if x < rsi_oversold else (-1 if x > rsi_overbought else 0))
    
    else:
        st.info("**Strategy:** Combines momentum + RSI for stronger signals.")
        mom_signal = factors['momentum_20'] > 0.02
        rsi_buy = factors['rsi_14'] < 50
        rsi_sell = factors['rsi_14'] > 50
        signals = ((mom_signal & rsi_buy).astype(int) - ((~mom_signal) & rsi_sell).astype(int))
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📊 Signal Distribution")
        signal_counts = signals.value_counts()
        fig_signals = go.Figure(data=[go.Bar(
            x=['Sell', 'Hold', 'Buy'],
            y=[signal_counts.get(-1, 0), signal_counts.get(0, 0), signal_counts.get(1, 0)],
            marker_color=['#ef4444', '#64748b', '#10b981']
        )])
        fig_signals.update_layout(**PLOT_CONFIG, height=300, showlegend=False)
        st.plotly_chart(fig_signals, use_container_width=True)
    
    with col2:
        st.markdown("### ⚙️ Backtest Settings")
        st.write(f"**Initial Capital:** ${initial_capital:,}")
        st.write(f"**Commission:** {commission_pct}%")
        st.write(f"**Test Period:** {len(prices)} days")
        st.write(f"**Strategy:** {strategy}")
    
    if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
        with st.spinner("Running backtest..."):
            bt = Backtester(initial_capital, commission_pct/100)
            common_idx = prices.index.intersection(signals.index)
            results = bt.run(prices.loc[common_idx], signals.loc[common_idx])
            metrics = bt.metrics(results)
        
        st.success("✅ Backtest Complete!")
        
        # Performance metrics in a nice grid
        st.markdown("### 📊 Performance Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_ret = metrics['total_return']
        col1.metric("Total Return", f"{total_ret:.2%}", 
                   f"{(total_ret - (prices.iloc[-1]/prices.iloc[0] - 1)):.2%} vs Buy&Hold")
        
        sharpe = metrics['sharpe']
        col2.metric("Sharpe Ratio", f"{sharpe:.2f}", 
                   "Excellent" if sharpe > 2 else ("Good" if sharpe > 1 else "Poor"))
        
        max_dd = metrics['max_drawdown']
        col3.metric("Max Drawdown", f"{max_dd:.2%}", 
                   "Low Risk" if max_dd > -0.1 else ("Moderate" if max_dd > -0.2 else "High Risk"))
        
        col4.metric("Final Value", f"${metrics['final_value']:,.0f}", 
                   f"${metrics['final_value'] - initial_capital:,.0f}")
        
        # Performance interpretation
        st.markdown("### 💡 Strategy Performance Analysis")
        
        if total_ret > 0.15 and sharpe > 1.5:
            st.markdown("""
            <div class="recommendation">
            <strong>✅ STRONG PERFORMANCE</strong><br>
            This strategy showed excellent risk-adjusted returns. High Sharpe ratio indicates consistent profits relative to volatility.
            <strong>Consider:</strong> This strategy may work well in similar market conditions.
            </div>
            """, unsafe_allow_html=True)
        elif total_ret > 0 and sharpe > 0.5:
            st.markdown("""
            <div class="insight-box">
            <strong>ℹ️ MODERATE PERFORMANCE</strong><br>
            Strategy was profitable but returns were modest relative to risk taken.
            <strong>Consider:</strong> Test with different parameters or combine with other signals.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box">
            <strong>⚠️ POOR PERFORMANCE</strong><br>
            Strategy underperformed or had negative returns. High volatility relative to returns.
            <strong>Consider:</strong> This strategy may not work in current market conditions. Try different approach.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Equity curve visualization
        st.markdown("### 📈 Portfolio Value Over Time")
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Portfolio Equity Curve', 'Drawdown'), 
                           vertical_spacing=0.12, row_heights=[0.65, 0.35])
        
        # Equity curve
        fig.add_trace(go.Scatter(x=results['date'], y=results['value'], name='Portfolio Value', 
                                line=dict(color='#10b981', width=3), fill='tozeroy', 
                                fillcolor='rgba(16, 185, 129, 0.1)'), row=1, col=1)
        
        # Buy & Hold comparison
        buy_hold = initial_capital * (prices.loc[common_idx] / prices.loc[common_idx].iloc[0])
        fig.add_trace(go.Scatter(x=results['date'], y=buy_hold, name='Buy & Hold', 
                                line=dict(color='#7c3aed', width=2, dash='dash')), row=1, col=1)
        
        # Drawdown
        cummax = results['value'].cummax()
        drawdown = ((results['value'] - cummax) / cummax) * 100
        fig.add_trace(go.Scatter(x=results['date'], y=drawdown, name='Drawdown %', 
                                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.3)', 
                                line=dict(color='#ef4444', width=2)), row=2, col=1)
        
        fig.update_layout(**PLOT_CONFIG, height=700, hovermode='x unified')
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            results_csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Backtest Results (CSV)", results_csv, 
                             f"backtest_{strategy.replace(' ', '_')}.csv", "text/csv")
        
        with col2:
            # Create summary report
            report = f"""AMIE Platform - Backtest Report
================================
Data: {data_type}
Strategy: {strategy}
Period: {results['date'].iloc[0].date()} to {results['date'].iloc[-1].date()}

Performance Metrics:
--------------------
Initial Capital: ${initial_capital:,}
Final Value: ${metrics['final_value']:,.2f}
Total Return: {metrics['total_return']:.2%}
Sharpe Ratio: {metrics['sharpe']:.2f}
Max Drawdown: {metrics['max_drawdown']:.2%}

Buy & Hold Return: {(prices.iloc[-1]/prices.iloc[0] - 1):.2%}
Outperformance: {(total_ret - (prices.iloc[-1]/prices.iloc[0] - 1)):.2%}

Strategy Parameters:
--------------------
Commission: {commission_pct}%
RSI Oversold: {rsi_oversold}
RSI Overbought: {rsi_overbought}
Mean Reversion Threshold: {mean_rev_threshold:.3f}
"""
            st.download_button("📥 Download Summary Report (TXT)", report.encode('utf-8'), 
                             f"report_{strategy.replace(' ', '_')}.txt", "text/plain")

# TAB 4: Comprehensive Reports
with tab4:
    st.markdown("## 📋 Comprehensive Analysis Reports")
    
    st.markdown("### 📊 Factor Correlation Matrix")
    
    corr_matrix = factors.corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale=[[0, '#7c3aed'], [0.5, '#1a1f3a'], [1, '#06b6d4']],
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10, "color": '#f8fafc'}
    ))
    
    fig_corr.update_layout(**PLOT_CONFIG, height=500)
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("### 📈 Factor Performance Summary")
    
    # Calculate IC for all factors
    forward_ret = prices.pct_change(5).shift(-5)
    valid_idx = factors.index.intersection(forward_ret.index)
    
    ic_results = []
    for col in factors.columns:
        fv = factors.loc[valid_idx, col]
        fr = forward_ret.loc[valid_idx]
        mask = ~(fv.isna() | fr.isna())
        
        if mask.sum() > 20:
            ic, pval = pearsonr(fv[mask], fr[mask])
            ic_results.append({
                'Factor': col,
                'IC (5-day)': ic,
                'Abs IC': abs(ic),
                'Signal': '✅ Strong' if abs(ic) > 0.1 else ('ℹ️ Moderate' if abs(ic) > 0.05 else '⚠️ Weak')
            })
    
    if ic_results:
        ic_df = pd.DataFrame(ic_results).sort_values('Abs IC', ascending=False)
        st.dataframe(ic_df[['Factor', 'IC (5-day)', 'Signal']], use_container_width=True, hide_index=True)
    
    st.markdown("### 🎯 Key Takeaways")
    
    st.markdown("""
    **How to Use This Platform:**
    
    1. **Market Overview Tab:** Check current price, RSI, and momentum to gauge market conditions
    2. **Factor Analysis Tab:** Identify which factors are currently signaling buy/sell opportunities
    3. **Backtesting Tab:** Test different strategies and parameters to find what works
    4. **This Tab:** Review overall factor performance and correlations
    
    **Next Steps:**
    - Adjust strategy parameters in sidebar and re-run backtests
    - Try different tickers if using real data
    - Export results for further analysis in Excel/Python
    - Combine insights from multiple factors for stronger signals
    """)
    
    # Data export section
    st.markdown("### 📥 Export All Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_csv = prices.to_csv().encode('utf-8')
        st.download_button("📥 Prices", price_csv, "prices.csv", "text/csv", use_container_width=True)
    
    with col2:
        factor_csv = factors.to_csv().encode('utf-8')
        st.download_button("📥 Factors", factor_csv, "factors.csv", "text/csv", use_container_width=True)
    
    with col3:
        if ic_results:
            ic_csv = ic_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 IC Analysis", ic_csv, "ic_analysis.csv", "text/csv", use_container_width=True)

# Footer with helpful tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem; color: #64748b;'>
    <strong>AMIE Platform v2.0 Enhanced</strong> | Institutional-Grade Quantitative Research<br>
    💡 <strong>Tip:</strong> Try different combinations of parameters to find optimal strategies for your trading style
</div>
""", unsafe_allow_html=True)
