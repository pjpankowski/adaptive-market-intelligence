"""
AMIE Platform v2.5 - COMPLETE WORKING VERSION
Vise-inspired design + All bugs fixed + Full functionality
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, skew, kurtosis
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import yfinance as yf
    REAL_DATA = True
except:
    REAL_DATA = False

st.set_page_config(page_title="AMIE Platform", page_icon="💎", layout="wide")

# VISE-INSPIRED CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp { background: linear-gradient(135deg, #0A0E1A 0%, #141824 50%, #1A1F2E 100%); }
    .main-header { font-size: 3.5rem; font-weight: 900; letter-spacing: -0.03em;
                   background: linear-gradient(135deg, #00D9FF 0%, #00BFD8 50%, #6B4FFF 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem; }
    .subtitle { font-size: 1.1rem; color: #A0AEC0; font-weight: 400; margin-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 700 !important; color: #00D9FF !important; }
    [data-testid="stMetricLabel"] { font-size: 0.875rem !important; color: #718096 !important; 
                                    font-weight: 500 !important; text-transform: uppercase; letter-spacing: 0.05em; }
    .stTabs [data-baseweb="tab-list"] { gap: 2rem; background-color: transparent; border-bottom: 1px solid rgba(255, 255, 255, 0.1); }
    .stTabs [data-baseweb="tab"] { background-color: transparent; border: none; color: #718096; 
                                   font-weight: 600; font-size: 0.95rem; padding: 1rem 0; transition: all 0.3s ease; }
    .stTabs [aria-selected="true"] { background-color: transparent !important; color: #00D9FF !important; 
                                     border-bottom: 2px solid #00D9FF; }
    .stButton > button { background: linear-gradient(135deg, #00D9FF 0%, #6B4FFF 100%); color: white;
                         border: none; border-radius: 12px; padding: 0.75rem 2rem; font-weight: 600;
                         font-size: 1rem; transition: all 0.3s ease; box-shadow: 0 4px 15px rgba(0, 217, 255, 0.3); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 217, 255, 0.4); }
    .insight-box { background: rgba(0, 217, 255, 0.08); backdrop-filter: blur(10px);
                   border: 1px solid rgba(0, 217, 255, 0.2); border-left: 4px solid #00D9FF;
                   padding: 1.5rem; border-radius: 16px; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }
    .recommendation { background: rgba(107, 79, 255, 0.08); backdrop-filter: blur(10px);
                      border: 1px solid rgba(107, 79, 255, 0.2); border-left: 4px solid #6B4FFF;
                      padding: 1.5rem; border-radius: 16px; margin: 1rem 0; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2); }
    .metric-card { background: rgba(26, 31, 46, 0.6); backdrop-filter: blur(10px);
                   border: 1px solid rgba(255, 255, 255, 0.1); padding: 1.5rem; border-radius: 16px;
                   margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #141824 0%, #1A1F2E 100%);
                                 border-right: 1px solid rgba(255, 255, 255, 0.1); }
    .stTextInput > div > div > input, .stSelectbox > div > div > select, .stNumberInput > div > div > input {
        background-color: rgba(26, 31, 46, 0.6); border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px; color: #E2E8F0; }
    .stSlider > div > div > div > div { background-color: #00D9FF; }
    .stInfo { background-color: rgba(0, 217, 255, 0.1); border-left: 4px solid #00D9FF; border-radius: 8px; }
    .stDownloadButton > button { background: rgba(0, 217, 255, 0.1); border: 1px solid rgba(0, 217, 255, 0.3);
                                 color: #00D9FF; border-radius: 8px; padding: 0.5rem 1rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

PLOT_VISE = {
    'font': {'family': 'Inter, sans-serif', 'color': '#E2E8F0', 'size': 12},
    'paper_bgcolor': 'rgba(20, 24, 36, 0.4)', 'plot_bgcolor': 'rgba(10, 14, 26, 0.6)',
    'xaxis': {'gridcolor': 'rgba(0, 217, 255, 0.1)', 'linecolor': 'rgba(255, 255, 255, 0.1)', 
              'zerolinecolor': 'rgba(0, 217, 255, 0.2)'},
    'yaxis': {'gridcolor': 'rgba(0, 217, 255, 0.1)', 'linecolor': 'rgba(255, 255, 255, 0.1)', 
              'zerolinecolor': 'rgba(0, 217, 255, 0.2)'},
    'colorway': ['#00D9FF', '#6B4FFF', '#00BFD8', '#9D7FFF', '#00FFF0']
}

def safe_float(value, default=0.0):
    try:
        val = float(value)
        return default if np.isnan(val) or np.isinf(val) else val
    except:
        return default

def clean_series_for_plot(series):
    clean = series.copy().dropna()
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

class AdvancedBacktester:
    def __init__(self, capital=100000, commission=0.001, slippage=0.0005):
        self.capital = capital
        self.commission = commission
        self.slippage = slippage
        self.trades = []
    
    def run(self, prices, signals):
        cash = self.capital
        shares = 0
        eq = []
        self.trades = []
        
        for i in range(len(prices)):
            try:
                date = prices.index[i]
                p = safe_float(prices.iloc[i], 100)
            except:
                continue
            
            try:
                sig = int(signals.loc[date]) if date in signals.index else 0
            except:
                sig = 0
            
            if sig == 1 and cash > 0:
                shares = int(cash * 0.95 / p)
                entry_price = p * (1 + self.slippage)
                total_cost = shares * entry_price * (1 + self.commission)
                
                if total_cost <= cash:
                    cash -= total_cost
                    self.trades.append({
                        'date': date, 'action': 'BUY', 'shares': shares, 'price': entry_price,
                        'commission': shares * entry_price * self.commission,
                        'slippage': shares * p * self.slippage
                    })
            
            elif sig == -1 and shares > 0:
                exit_price = p * (1 - self.slippage)
                proceeds = shares * exit_price * (1 - self.commission)
                
                pnl = 0
                pnl_pct = 0
                if len(self.trades) > 0 and self.trades[-1]['action'] == 'BUY':
                    entry = self.trades[-1]['price']
                    pnl = (exit_price - entry) * shares
                    pnl_pct = (exit_price - entry) / entry if entry != 0 else 0
                
                cash += proceeds
                self.trades.append({
                    'date': date, 'action': 'SELL', 'shares': shares, 'price': exit_price,
                    'commission': shares * exit_price * self.commission,
                    'slippage': shares * p * self.slippage,
                    'pnl': pnl, 'pnl_pct': pnl_pct
                })
                shares = 0
            
            val = cash + shares * p
            eq.append({'date': date, 'value': val, 'returns': (val - self.capital) / self.capital})
        
        return pd.DataFrame(eq)
    
    def compute_metrics(self, equity_df):
        if len(equity_df) == 0:
            return {'total_return': 0, 'sharpe': 0, 'sortino': 0, 'max_drawdown': 0, 'calmar': 0,
                   'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'profit_factor': 0,
                   'total_commission': 0, 'total_slippage': 0, 'num_trades': 0,
                   'skewness': 0, 'kurtosis': 0, 'final': self.capital}
        
        equity_df['daily_ret'] = equity_df['value'].pct_change()
        total_ret = (equity_df['value'].iloc[-1] - self.capital) / self.capital
        daily_ret_mean = equity_df['daily_ret'].mean()
        daily_ret_std = equity_df['daily_ret'].std()
        
        sharpe = (daily_ret_mean / daily_ret_std) * np.sqrt(252) if daily_ret_std > 0 else 0
        downside_ret = equity_df['daily_ret'][equity_df['daily_ret'] < 0]
        downside_std = downside_ret.std() if len(downside_ret) > 0 else daily_ret_std
        sortino = (daily_ret_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0
        
        cummax = equity_df['value'].cummax()
        drawdown = (equity_df['value'] - cummax) / cummax
        max_dd = drawdown.min()
        calmar = (total_ret / abs(max_dd)) if max_dd != 0 else 0
        
        trades_df = pd.DataFrame(self.trades)
        if len(trades_df) > 0 and 'pnl' in trades_df.columns:
            winning_trades = trades_df[trades_df['pnl'] > 0]
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            win_rate = len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            losing_trades = trades_df[trades_df['pnl'] < 0]
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        total_commission = trades_df['commission'].sum() if len(trades_df) > 0 else 0
        total_slippage = trades_df['slippage'].sum() if len(trades_df) > 0 else 0
        
        ret_vals = equity_df['daily_ret'].dropna()
        ret_skew = skew(ret_vals) if len(ret_vals) > 3 else 0
        ret_kurt = kurtosis(ret_vals) if len(ret_vals) > 3 else 0
        
        return {
            'total_return': total_ret, 'sharpe': sharpe, 'sortino': sortino,
            'max_drawdown': max_dd, 'calmar': calmar, 'win_rate': win_rate,
            'avg_win': avg_win, 'avg_loss': avg_loss, 'profit_factor': profit_factor,
            'total_commission': total_commission, 'total_slippage': total_slippage,
            'num_trades': len(trades_df[trades_df['action'] == 'SELL']) if len(trades_df) > 0 else 0,
            'skewness': ret_skew, 'kurtosis': ret_kurt, 'final': equity_df['value'].iloc[-1]
        }

st.markdown('<div class="main-header">💎 AMIE PLATFORM</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Institutional-grade quantitative analytics powered by AI</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    source = st.radio("Data Source", ["🎲 Simulated", "📊 Real Market Data"])
    
    if source == "📊 Real Market Data" and REAL_DATA:
        ticker = st.text_input("Ticker Symbol", "SPY")
        period = st.selectbox("Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
        use_real = True
    else:
        if source == "📊 Real Market Data":
            st.error("⚠️ yfinance not installed")
        n_samples = st.slider("Simulation Days", 100, 2000, 500)
        use_real = False
    
    st.markdown("---")
    st.markdown("## 💰 Trading Setup")
    capital = st.number_input("Initial Capital ($)", 10000, 1000000, 100000, 10000)
    comm = st.slider("Commission (%)", 0.0, 1.0, 0.1, 0.01)
    slippage = st.slider("Slippage (%)", 0.0, 0.5, 0.05, 0.01)
    
    st.markdown("---")
    st.markdown("## 🎯 Strategy Parameters")
    rsi_low = st.slider("RSI Oversold Level", 20, 40, 30)
    rsi_high = st.slider("RSI Overbought Level", 60, 80, 70)
    mr_thresh = st.slider("Mean Reversion Threshold", 0.01, 0.10, 0.05, 0.01)

@st.cache_data(ttl=3600)
def load_real(t, p):
    try:
        data = yf.download(t, period=p, progress=False)
        if len(data) > 0:
            prices = data['Close']
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
            st.error("Failed. Using simulated.")
            use_real = False
        else:
            st.success(f"✅ {len(prices)} days loaded")

if not use_real:
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    prices = pd.Series(np.cumsum(np.random.randn(n_samples) * 2) + 100, index=dates)

factors = AlphaFactors.compute_all(prices)

if 'backtest_results' not in st.session_state:
    st.session_state.backtest_results = None

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Factors", "💹 Backtest", "📋 Tearsheet", "🔬 Analytics"])

# TAB 1: Overview
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
    col4.metric("RSI (14)", f"{rsi:.1f}")
    
    st.markdown("### 🎯 Current Market Signal")
    if rsi > 70:
        st.markdown(f"""<div class="insight-box"><strong>⚠️ OVERBOUGHT SIGNAL</strong><br>
        RSI at {rsi:.1f}. Consider profit-taking or wait for pullback.</div>""", unsafe_allow_html=True)
    elif rsi < 30:
        st.markdown(f"""<div class="recommendation"><strong>✅ OVERSOLD SIGNAL</strong><br>
        RSI at {rsi:.1f}. Potential buying opportunity.</div>""", unsafe_allow_html=True)
    else:
        st.info(f"ℹ️ **NEUTRAL**: RSI at {rsi:.1f}")
    
    st.markdown("---")
    st.markdown("### 📈 Price Analysis")
    plot_prices = clean_series_for_plot(prices)
    ma20 = plot_prices.rolling(20).mean().dropna()
    ma50 = plot_prices.rolling(50).mean().dropna()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_prices.index, y=plot_prices.values, name='Price', 
                            line=dict(color='#00D9FF', width=3), fill='tozeroy', 
                            fillcolor='rgba(0, 217, 255, 0.1)', mode='lines'))
    if len(ma20) > 0:
        fig.add_trace(go.Scatter(x=ma20.index, y=ma20.values, name='20-Day MA', 
                                line=dict(color='#6B4FFF', width=2, dash='dash'), mode='lines'))
    if len(ma50) > 0:
        fig.add_trace(go.Scatter(x=ma50.index, y=ma50.values, name='50-Day MA', 
                                line=dict(color='#9D7FFF', width=2, dash='dot'), mode='lines'))
    
    fig.update_layout(**PLOT_VISE, height=550, hovermode='x unified', xaxis_title="Date", 
                     yaxis_title="Price ($)", showlegend=True,
                     legend=dict(x=0.01, y=0.99, bgcolor='rgba(20, 24, 36, 0.8)'))
    fig.update_yaxes(tickformat='$,.2f')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("📥 Download Price Data", prices.to_csv().encode(), "prices.csv", use_container_width=True)
    with col2:
        st.markdown(f"**Data:** {len(prices)} days | {plot_prices.index[0].date()} to {plot_prices.index[-1].date()}")

# TAB 2: Factors
with tab2:
    st.markdown("## Factor Analysis")
    if len(factors) == 0:
        st.warning("Need at least 60 days")
    else:
        sel = st.selectbox("Select Factor", factors.columns.tolist())
        col1, col2 = st.columns([2, 1])
        
        with col1:
            plot_prices_clean = clean_series_for_plot(prices)
            plot_factors_clean = clean_series_for_plot(factors[sel])
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=('Price', f'{sel.replace("_", " ").title()}'),
                               vertical_spacing=0.15, row_heights=[0.5, 0.5])
            fig.add_trace(go.Scatter(x=plot_prices_clean.index, y=plot_prices_clean.values, name='Price', 
                                    line=dict(color='#00D9FF', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=plot_factors_clean.index, y=plot_factors_clean.values, name=sel, 
                                    line=dict(color='#6B4FFF', width=2), fill='tozeroy', 
                                    fillcolor='rgba(107, 79, 255, 0.15)'), row=2, col=1)
            fig.update_layout(**PLOT_VISE, height=700, showlegend=False)
            fig.update_yaxes(title_text="Price ($)", row=1, col=1, tickformat='$,.2f')
            fig.update_yaxes(title_text="Factor Value", row=2, col=1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Statistics")
            cur = safe_float(factors[sel].iloc[-1])
            mean = safe_float(factors[sel].mean())
            std = safe_float(factors[sel].std())
            st.metric("Current", f"{cur:.4f}")
            st.metric("Mean", f"{mean:.4f}")
            st.metric("Std Dev", f"{std:.4f}")
            if std > 0:
                z = (cur - mean) / std
                st.metric("Z-Score", f"{z:.2f}")

# TAB 3: Backtest
with tab3:
    st.markdown("## Strategy Backtesting")
    strat = st.selectbox("Strategy", ["Mean Reversion", "Momentum", "RSI Oscillator", "Multi-Factor"])
    
    if len(factors) == 0:
        st.warning("Not enough data")
    else:
        if strat == "Mean Reversion":
            st.info("Buy below average, sell above")
            sigs = factors['mean_reversion'].apply(lambda x: -1 if x > mr_thresh else (1 if x < -mr_thresh else 0))
        elif strat == "Momentum":
            st.info("Follow trends")
            sigs = factors['momentum_20'].apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
        elif strat == "RSI Oscillator":
            st.info(f"Buy < {rsi_low}, sell > {rsi_high}")
            sigs = factors['rsi_14'].apply(lambda x: 1 if x < rsi_low else (-1 if x > rsi_high else 0))
        else:
            st.info("Combined signals")
            sigs = ((factors['momentum_20'] > 0.02) & (factors['rsi_14'] < 50)).astype(int) - \
                   ((factors['momentum_20'] < -0.02) & (factors['rsi_14'] > 50)).astype(int)
        
        if st.button("🚀 Run Backtest", type="primary"):
            with st.spinner("Running..."):
                bt = AdvancedBacktester(capital, comm/100, slippage/100)
                idx = prices.index.intersection(sigs.index)
                if len(idx) > 0:
                    res = bt.run(prices.loc[idx], sigs.loc[idx])
                    met = bt.compute_metrics(res)
                    st.session_state.backtest_results = {'results': res, 'metrics': met, 'prices': prices.loc[idx],
                                                         'signals': sigs.loc[idx], 'strategy': strat, 'backtester': bt}
                    st.success("✅ Complete!")
                    
                    st.markdown("### 📊 Performance")
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Total Return", f"{met['total_return']:.2%}")
                    col2.metric("Sharpe", f"{met['sharpe']:.2f}")
                    col3.metric("Sortino", f"{met['sortino']:.2f}")
                    col4.metric("Max DD", f"{met['max_drawdown']:.2%}")
                    col5.metric("Calmar", f"{met['calmar']:.2f}")
                    
                    st.markdown("### 📈 Equity Curve")
                    res_clean = res.copy()
                    if hasattr(res_clean['date'].iloc[0], 'tz'):
                        res_clean['date'] = pd.to_datetime(res_clean['date']).dt.tz_localize(None)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res_clean['date'], y=res_clean['value'], name='Strategy', 
                                            line=dict(color='#00D9FF', width=3), fill='tozeroy', 
                                            fillcolor='rgba(0, 217, 255, 0.15)'))
                    bh = capital * (prices.loc[idx] / prices.loc[idx].iloc[0])
                    fig.add_trace(go.Scatter(x=res_clean['date'], y=bh.values, name='Buy&Hold', 
                                            line=dict(color='#6B4FFF', width=2, dash='dash')))
                    fig.update_layout(**PLOT_VISE, height=500, hovermode='x unified')
                    fig.update_yaxes(tickformat='$,.0f')
                    st.plotly_chart(fig, use_container_width=True)

# TAB 4: Tearsheet
with tab4:
    st.markdown("## 📋 Performance Tearsheet")
    if st.session_state.backtest_results is None:
        st.info("👈 Run a backtest first")
    else:
        res = st.session_state.backtest_results['results']
        met = st.session_state.backtest_results['metrics']
        bt = st.session_state.backtest_results['backtester']
        
        st.markdown(f"### {st.session_state.backtest_results['strategy']}")
        st.markdown(f"**Period:** {res['date'].iloc[0].date()} to {res['date'].iloc[-1].date()}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Returns**")
            st.write(f"• Total: **{met['total_return']:.2%}**")
            st.write(f"• Sharpe: **{met['sharpe']:.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Risk**")
            st.write(f"• Max DD: **{met['max_drawdown']:.2%}**")
            st.write(f"• Sortino: **{met['sortino']:.2f}**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 📈 Returns & Drawdown")
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Cumulative Returns', 'Drawdown'), 
                           vertical_spacing=0.12, row_heights=[0.6, 0.4])
        cum_ret = (res['value'] / capital - 1) * 100
        fig.add_trace(go.Scatter(x=res['date'], y=cum_ret, name='Returns', 
                                line=dict(color='#00D9FF', width=2), fill='tozeroy'), row=1, col=1)
        cummax = res['value'].cummax()
        dd = ((res['value'] - cummax) / cummax) * 100
        fig.add_trace(go.Scatter(x=res['date'], y=dd, name='DD', line=dict(color='#ef4444', width=2), 
                                fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.2)'), row=2, col=1)
        fig.update_layout(**PLOT_VISE, height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# TAB 5: Analytics
with tab5:
    st.markdown("## 🔬 Advanced Analytics")
    if st.session_state.backtest_results is None:
        st.info("👈 Run a backtest first")
    else:
        res = st.session_state.backtest_results['results']
        prices_used = st.session_state.backtest_results['prices']
        
        st.markdown("### 📈 Rolling Metrics")
        window = st.slider("Window (days)", 20, 120, 60, 10)
        
        res_copy = res.copy()
        res_copy['rolling_ret'] = res_copy['daily_ret'].rolling(window).apply(lambda x: (1+x).prod()-1) * 100
        res_copy['rolling_sharpe'] = res_copy['daily_ret'].rolling(window).apply(
            lambda x: (x.mean()/x.std())*np.sqrt(252) if x.std()>0 else 0)
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=(f'{window}d Return', f'{window}d Sharpe'), 
                           vertical_spacing=0.1)
        fig.add_trace(go.Scatter(x=res_copy['date'], y=res_copy['rolling_ret'], name='Return', 
                                line=dict(color='#00D9FF', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=res_copy['date'], y=res_copy['rolling_sharpe'], name='Sharpe', 
                                line=dict(color='#6B4FFF', width=2)), row=2, col=1)
        fig.add_hline(y=1, line_dash="dash", line_color="#00BFD8", row=2, col=1)
        fig.update_layout(**PLOT_VISE, height=700, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Factor Attribution")
        fwd_ret = prices_used.pct_change(5).shift(-5)
        all_factors = AlphaFactors.compute_all(prices_used)
        
        attribution = []
        for col in all_factors.columns:
            common_idx = all_factors.index.intersection(fwd_ret.index)
            if len(common_idx) > 20:
                f_vals = all_factors.loc[common_idx, col]
                ret_vals = fwd_ret.loc[common_idx]
                mask = ~(f_vals.isna() | ret_vals.isna())
                
                if int(mask.sum()) > 20:
                    try:
                        ic, _ = pearsonr(f_vals[mask], ret_vals[mask])
                        attribution.append({'Factor': col.replace('_',' ').title(), 'IC': ic, 'Abs_IC': abs(ic)})
                    except:
                        continue
        
        if len(attribution) > 0:
            attr_df = pd.DataFrame(attribution).sort_values('Abs_IC', ascending=False)
            fig = go.Figure(data=[go.Bar(x=attr_df['Factor'], y=attr_df['IC'],
                                        marker_color=['#00D9FF' if x>0 else '#ef4444' for x in attr_df['IC']],
                                        text=np.round(attr_df['IC'], 3), textposition='outside')])
            fig.add_hline(y=0, line_color='#718096', line_width=1)
            fig.update_layout(**PLOT_VISE, height=400, xaxis_title="Factor", yaxis_title="IC")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(attr_df[['Factor', 'IC']], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style='text-align:center; padding: 2rem;'>
    <div style='color: #00D9FF; font-size: 0.9rem; font-weight: 600;'>💎 AMIE PLATFORM v2.5 PRO</div>
    <div style='color: #718096; font-size: 0.8rem;'>Institutional-Grade Analytics | Powered by AI</div>
</div>
""", unsafe_allow_html=True)

