import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Car Sales Hierarchical Forecasting System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for light-mode glassmorphism & premium UI with pastel accents
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Background - Clean Light Off-White */
    .stApp {
        background-color: #f8fafc !important;
        color: #334155;
    }
    
    /* Sidebar styling - Clean Gradient Slate White */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%) !important;
        border-right: 1px solid #e2e8f0;
    }
    
    /* Remove sidebar top padding gap */
    [data-testid="stSidebarUserContent"] {
        padding-top: 0.2rem !important;
    }
    
    [data-testid="stSidebarHeader"] {
        display: none !important;
    }
    
    /* Pull the logo widget container up */
    [data-testid="stSidebarUserContent"] > div:first-child {
        margin-top: -6px !important;
    }
    
    /* Custom tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #f1f5f9;
        padding: 5px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 600;
        font-size: 14px;
        border: none;
        padding: 0 16px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #6366f1;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #6366f1 !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.15) !important;
    }
    
    /* Premium Light-mode Metric Cards with Soft Shadow and Hover Effects */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(99, 102, 241, 0.25);
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.08);
    }
    
    /* Pastel Icon Boxes */
    .metric-icon-box {
        border-radius: 10px;
        width: 44px;
        height: 44px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    
    .metric-info {
        flex-grow: 1;
    }
    .metric-title {
        font-size: 12px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    .metric-value {
        font-size: 26px;
        font-weight: 750;
        color: #0f172a;
        margin: 4px 0 0 0;
        letter-spacing: -0.01em;
    }
    .metric-delta {
        font-size: 11px;
        color: #64748b;
        margin-top: 2px;
        font-weight: 500;
    }
    
    /* Minimalist Header with a Soft Top Accent Line */
    .app-header {
        background: #ffffff;
        padding: 24px 30px;
        border-radius: 14px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
        border-top: 4px solid #6366f1;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.02);
    }
    .app-title {
        font-size: 28px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .app-subtitle {
        font-size: 14px;
        color: #64748b;
        margin-top: 6px;
        font-weight: 400;
    }
    
    /* Section Headers */
    .section-header {
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 6px;
        margin-bottom: 18px;
        font-size: 18px;
        font-weight: 700;
        color: #4f46e5;
        letter-spacing: -0.01em;
    }
    
    /* Light Code Containers */
    .stCodeBlock {
        background-color: #f8fafc !important;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    /* Sidebar widget cards */
    .sidebar-widget {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 16px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.01);
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10b981;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.6s infinite;
        margin-right: 6px;
    }
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 6px rgba(16, 185, 129, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0);
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DATA LOADING (CACHED)
# ----------------------------------------------------

@st.cache_data
def load_raw_data():
    cols_to_use = ['Date', 'Dealer_Region', 'Company', 'Body Style', 'Model', 'Color', 'Price ($)']
    df = pd.read_csv("car_sales_extended_raw.csv", usecols=cols_to_use, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "").replace("-", "_") for c in df.columns]
    df['date_dt'] = pd.to_datetime(df['date'], format='mixed', dayfirst=True)
    df['month_start'] = df['date_dt'].dt.to_period('M').dt.to_timestamp()
    return df

@st.cache_data
def load_forecast_data():
    forecast_file = "reconciled_forecast.csv"
    if os.path.exists(forecast_file):
        df = pd.read_csv(forecast_file)
        df['ds'] = pd.to_datetime(df['ds'])
        return df
    return None

# Load the datasets
with st.spinner("Initializing performance data..."):
    df_raw = load_raw_data()
    df_forecast = load_forecast_data()

# ----------------------------------------------------
# SIDEBAR CONTROLS
# ----------------------------------------------------

# Compact Logo branding at the top of the sidebar with clean bottom border
st.sidebar.markdown("""
<div style='text-align: center; padding: 6px 0 8px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 10px;'>
    <h3 style='color:#4f46e5; font-weight:800; margin:0; font-size: 20px; letter-spacing:-0.02em;'>🚗 AUTOFCAST</h3>
    <p style='color:#64748b; font-size:9px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin:2px 0 0 0;'>Hierarchical Engine</p>
</div>
""", unsafe_allow_html=True)

# Navigation Workspace
app_mode = st.sidebar.radio(
    "Navigation Workspace",
    ["Overview Dashboard", "Interactive Drilldown & EDA", "Forecast Simulation", "Pipeline & Code Architecture"]
)

st.sidebar.markdown("---")

# INTERACTIVE SIDEBAR Presets widget
st.sidebar.markdown("<p style='color: #475569; font-size: 11px; font-weight:700; text-transform: uppercase; margin-bottom: 10px; letter-spacing: 0.05em;'>Active Market Scenario</p>", unsafe_allow_html=True)

# Presets mapping
PRESETS = {
    "Default Reconciled Forecast": {"growth": 0, "noise": 0},
    "Summer Sales Surge (+15% Growth)": {"growth": 15, "noise": 3},
    "Economic Contraction (-25% Drop)": {"growth": -25, "noise": 6},
    "High Volatility Disruption": {"growth": 5, "noise": 15}
}

scenario_preset = st.sidebar.selectbox(
    "Choose Preset Scenario",
    list(PRESETS.keys())
)

selected_preset = PRESETS[scenario_preset]

# Dynamically set session states when selection changes to resolve Streamlit gotchas
if 'prev_preset' not in st.session_state or st.session_state['prev_preset'] != scenario_preset:
    st.session_state['growth_slider'] = selected_preset["growth"]
    st.session_state['noise_slider'] = selected_preset["noise"]
    st.session_state['prev_preset'] = scenario_preset

active_growth = st.session_state.get('growth_slider', 0)
active_noise = st.session_state.get('noise_slider', 0)

# Displays preset summary badge directly in the sidebar
st.sidebar.markdown(f"""
<div style='background: rgba(99, 102, 241, 0.05); border: 1px solid rgba(99, 102, 241, 0.15); border-radius: 6px; padding: 10px; margin-top: 8px;'>
    <div style='font-size: 12px; color: #4f46e5; font-weight:700;'>Simulated Factors:</div>
    <div style='font-size: 13px; color: #334155; margin-top: 4px;'>Growth Rate: <strong>{active_growth:+.0f}%</strong></div>
    <div style='font-size: 13px; color: #334155;'>Noise Level: <strong>{active_noise}%</strong></div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Diagnostics widget with live pulse effect
st.sidebar.markdown("""
<div class="sidebar-widget">
    <p style='color: #64748b; font-size: 10px; font-weight:700; text-transform: uppercase; margin-top:0; margin-bottom: 10px; letter-spacing: 0.05em;'>System Diagnostics</p>
    <div style='display: flex; align-items: center; font-size: 13px; font-weight:600; color: #0f172a; margin-bottom: 8px;'>
        <span class="pulse-dot"></span> Pipeline Status: Active
    </div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Database Tier:</span><span style='color: #334155; font-weight: 600;'>Gold Validation</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Dataset Volume:</span><span style='color: #334155; font-weight: 600;'>69.3K rows</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Model Engine:</span><span style='color: #4f46e5; font-weight: 600;'>StatsForecast</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Engine Reconciler:</span><span style='color: #10b981; font-weight: 600;'>Bottom-Up</span></div>
</div>
""", unsafe_allow_html=True)

# Tech Stack Badges
st.sidebar.markdown("""
<div style='text-align: center; margin-top: 15px;'>
    <span style='background: rgba(99, 102, 241, 0.08); color: #6366f1; font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 20px; margin-right: 4px;'>Streamlit v1.58</span>
    <span style='background: rgba(16, 185, 129, 0.08); color: #10b981; font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 20px; margin-right: 4px;'>Nixtla</span>
    <span style='background: rgba(245, 158, 11, 0.08); color: #f59e0b; font-size: 10px; font-weight: 600; padding: 4px 8px; border-radius: 20px;'>Python 3</span>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Car Sales Hierarchical Forecasting</div>
    <div class="app-subtitle">An interactive visual explorer for time series hierarchical forecasting pipeline</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 1: OVERVIEW DASHBOARD
# ----------------------------------------------------
if app_mode == "Overview Dashboard":
    st.markdown("<div class='section-header'>📊 Executive Performance Overview</div>", unsafe_allow_html=True)
    
    # KPI Calculations
    total_sales = len(df_raw)
    total_revenue = df_raw['price_'].sum()
    avg_price = df_raw['price_'].mean()
    top_region = df_raw['dealer_region'].value_counts().index[0]
    
    regions_list = df_raw['dealer_region'].unique().tolist()
    total_reconciled_fcst_sum = 0
    total_simulated_fcst_sum = 0
    
    if df_forecast is not None:
        df_reg_fcst = df_forecast[df_forecast['unique_id'].isin(regions_list)]
        total_reconciled_fcst_sum = df_reg_fcst['AutoARIMA/BottomUp'].sum()
        
        sim_multiplier = 1 + (active_growth / 100.0)
        np.random.seed(42)
        noise_factors = 1.0 + (np.random.normal(0, active_noise / 100.0, len(df_reg_fcst)))
        total_simulated_fcst_sum = (df_reg_fcst['AutoARIMA/BottomUp'] * sim_multiplier * noise_factors).sum()

    # Row of Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background: rgba(99, 102, 241, 0.08);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#6366f1" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Sales Volume</div>
                <div class="metric-value">{total_sales:,}</div>
                <div class="metric-delta">Gross units (2022-2026)</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background: rgba(16, 185, 129, 0.08);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Total Revenue</div>
                <div class="metric-value">${total_revenue/1e6:.2f}M</div>
                <div class="metric-delta">Gross bookings</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background: rgba(245, 158, 11, 0.08);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/><circle cx="12" cy="13" r="3"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Simulated 2027 Sales</div>
                <div class="metric-value" style="color: #6366f1;">{int(round(total_simulated_fcst_sum)):,}</div>
                <div class="metric-delta">Scenario: {scenario_preset.split(" (")[0]}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box" style="background: rgba(239, 68, 68, 0.06);">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a8 8 0 0 0-8 8c0 5.25 8 12 8 12s8-6.75 8-12a8 8 0 0 0-8-8z"/><circle cx="12" cy="10" r="3"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Top Region</div>
                <div class="metric-value" style="font-size: 22px; padding-top: 2px;">{top_region}</div>
                <div class="metric-delta">Volume leader</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Core Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("##### Historical Sales and Future Scenario Projections")
        monthly_sales = df_raw.groupby('month_start').size().reset_index(name='sales')
        
        if df_forecast is not None:
            df_reg_fcst = df_forecast[df_forecast['unique_id'].isin(regions_list)]
            monthly_fcst = df_reg_fcst.groupby('ds')[['AutoARIMA/BottomUp']].sum().reset_index()
            monthly_fcst = monthly_fcst.sort_values('ds')
            
            sim_ratio = 1 + (active_growth / 100.0)
            np.random.seed(42)
            noise_factors = 1.0 + (np.random.normal(0, active_noise / 100.0, len(monthly_fcst)))
            monthly_fcst['Simulated'] = monthly_fcst['AutoARIMA/BottomUp'] * sim_ratio * noise_factors
        
        fig_trend = go.Figure()
        
        # Historical Path
        fig_trend.add_trace(go.Scatter(
            x=monthly_sales['month_start'],
            y=monthly_sales['sales'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(148, 163, 184, 0.03)',
            line=dict(color='#94a3b8', width=2),
            name="Historical Sales"
        ))
        
        # Reconciled Forecast Path
        fig_trend.add_trace(go.Scatter(
            x=monthly_fcst['ds'],
            y=monthly_fcst['AutoARIMA/BottomUp'],
            mode='lines+markers',
            line=dict(color='#6366f1', width=3),
            name="Base Reconciled Forecast"
        ))
        
        # Scenario Simulated Path
        if active_growth != 0 or active_noise != 0:
            fig_trend.add_trace(go.Scatter(
                x=monthly_fcst['ds'],
                y=monthly_fcst['Simulated'],
                mode='lines+markers',
                line=dict(color='#ec4899', width=2.5, dash='dot'),
                name="Simulated Scenario"
            ))
            
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#475569',
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.markdown("##### Sales Share by Region")
        region_sales = df_raw['dealer_region'].value_counts().reset_index()
        region_sales.columns = ['region', 'sales']
        
        # Donut chart with legend positioned horizontally below the chart to prevent overlap
        fig_pie = go.Figure(data=[go.Pie(
            labels=region_sales['region'],
            values=region_sales['sales'],
            hole=0.5,
            marker=dict(colors=['#818cf8', '#a78bfa', '#f472b6', '#fb7185', '#38bdf8']),
            textinfo='percent',
            textposition='inside',
            showlegend=True
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#475569',
            margin=dict(l=10, r=10, t=10, b=40),
            height=360,
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.05,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Anomaly Section
    st.markdown("##### 🚨 Historical Sales Anomalies")
    sales_series = df_raw.groupby('month_start').size().reset_index(name='sales')
    sales_mean = sales_series['sales'].mean()
    sales_std = sales_series['sales'].std()
    sales_series['z_score'] = (sales_series['sales'] - sales_mean) / sales_std
    anomalies = sales_series[sales_series['z_score'].abs() > 1.8]
    
    if len(anomalies) > 0:
        anom_cols = st.columns(len(anomalies))
        for idx, row in enumerate(anomalies.itertuples()):
            is_high = row.z_score > 0
            color = "#6366f1" if is_high else "#ef4444"
            arrow = "↑" if is_high else "↓"
            status = "Spike" if is_high else "Dip"
            with anom_cols[idx]:
                st.markdown(f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 4px solid {color}; border-radius: 8px; padding: 14px; text-align: left; box-shadow: 0 2px 8px rgba(0,0,0,0.01);">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">{status} Detected</div>
                    <div style="font-size: 14px; font-weight: 700; color: #0f172a; margin-top: 2px;">{row.month_start.strftime('%B %Y')}</div>
                    <div style="font-size: 12px; color: #334155; margin-top: 4px;">Sales: {row.sales} <span style="color: {color}; font-weight: 600;">({row.z_score:+.1f} SD)</span></div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No statistical outliers detected in the historical period.")


# ----------------------------------------------------
# TAB 2: INTERACTIVE DRILLDOWN & EDA
# ----------------------------------------------------
elif app_mode == "Interactive Drilldown & EDA":
    st.markdown("<div class='section-header'>🌳 Interactive Drilldown Hierarchy</div>", unsafe_allow_html=True)
    
    st.markdown("""
    Explore the hierarchical structure of sales. Click on any Region tile to drill down into Brands (Companies) and Body Styles.
    """)
    
    # Treemap Chart
    fig_tree = px.treemap(
        df_raw,
        path=['dealer_region', 'company', 'body_style'],
        values='price_',
        color='price_',
        color_continuous_scale=[[0, '#f5f3ff'], [0.5, '#c7d2fe'], [1.0, '#818cf8']],
        labels={'price_': 'Total Revenue ($)', 'dealer_region': 'Region', 'company': 'Brand', 'body_style': 'Body Style'}
    )
    fig_tree.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#475569',
        margin=dict(l=10, r=10, t=10, b=10),
        height=500
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Detailed segments
    s1, s2 = st.columns(2)
    
    with s1:
        st.markdown("##### Top 10 Vehicle Brands by Revenue")
        top_companies = df_raw.groupby('company')['price_'].sum().reset_index().sort_values('price_', ascending=False).head(10)
        fig_c = px.bar(top_companies, x='price_', y='company', orientation='h', color='price_', color_continuous_scale='Purples')
        fig_c.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#475569', 
            height=260, 
            coloraxis_showscale=False, 
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_c, use_container_width=True)
        
    with s2:
        st.markdown("##### Brand Price Distribution Matrix")
        top_companies_list = top_companies['company'].tolist()
        df_top_c = df_raw[df_raw['company'].isin(top_companies_list)]
        fig_box = px.box(df_top_c, x='company', y='price_', color_discrete_sequence=['#818cf8'])
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#475569', 
            height=260, 
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)')
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ----------------------------------------------------
# TAB 3: FORECAST SIMULATION
# ----------------------------------------------------
elif app_mode == "Forecast Simulation":
    st.markdown("<div class='section-header'>📈 Forecast Viewer & Scenario Planner</div>", unsafe_allow_html=True)
    
    if df_forecast is None:
        st.error("Missing prediction output file (`reconciled_forecast.csv`).")
    else:
        # Dynamic hierarchy selector panel
        st.markdown("##### Select Series node")
        
        # Grid layout for selectors
        col_sel_1, col_sel_2 = st.columns(2)
        
        with col_sel_1:
            level = st.selectbox(
                "Hierarchy Depth",
                [
                    "Region",
                    "Company (Brand)",
                    "Region & Company",
                    "Region, Company & Model",
                    "Region, Company, Model & Color (Leaf Node)"
                ]
            )
            
        selected_unique_id = ""
        
        with col_sel_2:
            if level == "Region":
                regions = sorted(df_raw['dealer_region'].unique().tolist())
                selected_region = st.selectbox("Select Region", regions)
                selected_unique_id = selected_region
                
            elif level == "Company (Brand)":
                companies = sorted(df_raw['company'].unique().tolist())
                selected_company = st.selectbox("Select Brand", companies)
                selected_unique_id = selected_company
                
            elif level == "Region & Company":
                r_c_col_1, r_c_col_2 = st.columns(2)
                with r_c_col_1:
                    regions = sorted(df_raw['dealer_region'].unique().tolist())
                    selected_region = st.selectbox("Select Region", regions)
                with r_c_col_2:
                    filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
                    selected_company = st.selectbox("Select Brand", filtered_companies)
                selected_unique_id = f"{selected_region}/{selected_company}"
                
            elif level == "Region, Company & Model":
                r_c_m_col_1, r_c_m_col_2, r_c_m_col_3 = st.columns(3)
                with r_c_m_col_1:
                    regions = sorted(df_raw['dealer_region'].unique().tolist())
                    selected_region = st.selectbox("Select Region", regions)
                with r_c_m_col_2:
                    filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
                    selected_company = st.selectbox("Select Brand", filtered_companies)
                with r_c_m_col_3:
                    filtered_models = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company)]['model'].unique().tolist())
                    selected_model = st.selectbox("Select Model", filtered_models)
                selected_unique_id = f"{selected_region}/{selected_company}/{selected_model}"
                
            elif level == "Region, Company, Model & Color (Leaf Node)":
                r_c_m_c_col_1, r_c_m_c_col_2, r_c_m_c_col_3, r_c_m_c_col_4 = st.columns(4)
                with r_c_m_c_col_1:
                    regions = sorted(df_raw['dealer_region'].unique().tolist())
                    selected_region = st.selectbox("Select Region", regions)
                with r_c_m_c_col_2:
                    filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
                    selected_company = st.selectbox("Select Brand", filtered_companies)
                with r_c_m_c_col_3:
                    filtered_models = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company)]['model'].unique().tolist())
                    selected_model = st.selectbox("Select Model", filtered_models)
                with r_c_m_c_col_4:
                    filtered_colors = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company) & (df_raw['model'] == selected_model)]['color'].unique().tolist())
                    selected_color = st.selectbox("Select Color", filtered_colors)
                selected_unique_id = f"{selected_region}/{selected_company}/{selected_model}/{selected_color}"

        st.markdown(f"<span style='color: #64748b;'>Active Series Node:</span> ` {selected_unique_id} `", unsafe_allow_html=True)
        
        # Filter historical data
        hist_df = df_raw.copy()
        if level == "Region":
            hist_df = hist_df[hist_df['dealer_region'] == selected_region]
        elif level == "Company (Brand)":
            hist_df = hist_df[hist_df['company'] == selected_company]
        elif level == "Region & Company":
            hist_df = hist_df[(hist_df['dealer_region'] == selected_region) & (hist_df['company'] == selected_company)]
        elif level == "Region, Company & Model":
            hist_df = hist_df[(hist_df['dealer_region'] == selected_region) & (hist_df['company'] == selected_company) & (hist_df['model'] == selected_model)]
        elif level == "Region, Company, Model & Color (Leaf Node)":
            hist_df = hist_df[(hist_df['dealer_region'] == selected_region) & (hist_df['company'] == selected_company) & (hist_df['model'] == selected_model) & (hist_df['color'] == selected_color)]
            
        hist_monthly = hist_df.groupby('month_start').size().reset_index(name='y')
        hist_monthly.columns = ['ds', 'y']
        hist_monthly = hist_monthly.sort_values('ds')
        
        series_fcst = df_forecast[df_forecast['unique_id'] == selected_unique_id].sort_values('ds')
        
        if len(series_fcst) == 0:
            st.error(f"⚠️ Forecast predictions not available for series `{selected_unique_id}`.")
        else:
            # ----------------------------------------------------
            # INTERACTIVE SCENARIO SIMULATOR
            # ----------------------------------------------------
            st.markdown("##### ⚡ Real-Time Scenario Planner")
            
            sim_col_1, sim_col_2 = st.columns(2)
            with sim_col_1:
                growth_slider = st.slider(
                    "Simulated Growth Shift (%)",
                    min_value=-50,
                    max_value=50,
                    key="growth_slider",
                    step=5,
                    help="Shift the reconciled forecast path."
                )
            with sim_col_2:
                noise_slider = st.slider(
                    "Add Volatility (%)",
                    min_value=0,
                    max_value=20,
                    key="noise_slider",
                    step=2,
                    help="Add simulated market noise."
                )
            
            # Apply growth and noise
            sim_ratio = 1 + (growth_slider / 100.0)
            np.random.seed(42)
            noise_factors = 1.0 + (np.random.normal(0, noise_slider / 100.0, len(series_fcst)))
            
            series_fcst['Simulated'] = series_fcst['AutoARIMA/BottomUp'] * sim_ratio * noise_factors
            
            # Reconcile plotting with subtle colors
            fig = go.Figure()
            
            # History (Muted gray)
            fig.add_trace(go.Scatter(
                x=hist_monthly['ds'],
                y=hist_monthly['y'],
                name='Historical Sales',
                line=dict(color='#94a3b8', width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            
            # Base AutoARIMA (Dashed light gray)
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA'],
                name='Base AutoARIMA Forecast',
                line=dict(color='#cbd5e1', width=1.5, dash='dash'),
                mode='lines'
            ))
            
            # Reconciled BottomUp (Pastel Indigo)
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA/BottomUp'],
                name='Reconciled Forecast',
                line=dict(color='#6366f1', width=3),
                mode='lines+markers',
                marker=dict(size=6)
            ))
            
            # Simulated Scenario (Pastel Rose Pink)
            if growth_slider != 0 or noise_slider != 0:
                fig.add_trace(go.Scatter(
                    x=series_fcst['ds'],
                    y=series_fcst['Simulated'],
                    name='Simulated Scenario',
                    line=dict(color='#ec4899', width=2.5, dash='dot'),
                    mode='lines+markers',
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Monthly Sales Volume",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#475569',
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.03)'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Muted statistics metrics
            fcst_sum_reconciled = series_fcst['AutoARIMA/BottomUp'].sum()
            fcst_sum_simulated = series_fcst['Simulated'].sum()
            hist_sum_last_year = hist_monthly[hist_monthly['ds'] >= '2026-01-01']['y'].sum()
            
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Historical Sales (2026)", f"{int(hist_sum_last_year)} units")
            with s_col2:
                st.metric("Reconciled Forecast (2027)", f"{int(round(fcst_sum_reconciled))} units")
            with s_col3:
                delta_sim = fcst_sum_simulated - fcst_sum_reconciled
                st.metric(
                    "Simulated Scenario Total", 
                    f"{int(round(fcst_sum_simulated))} units",
                    delta=f"{delta_sim:+.0f} vs base reconciled",
                    delta_color="off" # Grey delta
                )
                
            # Table details
            st.markdown("##### Forecast details matrix")
            display_cols = ['ds', 'AutoARIMA', 'AutoARIMA/BottomUp', 'Simulated']
            display_labels = ['Timeline', 'Unreconciled Base', 'Reconciled Bottom-Up', 'Simulated Path']
            
            display_df = series_fcst[display_cols].copy()
            display_df.columns = display_labels
            display_df['Timeline'] = display_df['Timeline'].dt.strftime('%B %Y')
            
            st.dataframe(display_df.style.format({
                'Unreconciled Base': '{:,.1f}',
                'Reconciled Bottom-Up': '{:,.1f}',
                'Simulated Path': '{:,.1f}'
            }), use_container_width=True)


# ----------------------------------------------------
# TAB 4: PIPELINE & CODE ARCHITECTURE
# ----------------------------------------------------
elif app_mode == "Pipeline & Code Architecture":
    st.markdown("<div class='section-header'>🛠️ Data Pipeline & Code Architecture</div>", unsafe_allow_html=True)
    
    st.markdown("""
    This project was originally developed inside **Databricks** as a hierarchical time series forecasting pipeline. 
    Below is a breakdown of the notebooks that make up the pipeline.
    """)
    
    tab_ingest, tab_prep, tab_baseline, tab_reconcile = st.tabs([
        "📥 1_Ingestion.ipynb",
        "⚙️ 2_Preprocessing.ipynb",
        "🤖 3_Baseline.ipynb (Base Models)",
        "⚖️ 4. Reconciliation Process"
    ])
    
    with tab_ingest:
        st.markdown("""
        **Objective:** Raw data validation, column cleaning, and schemas ingestion.
        
        **Key Steps:**
        - Load gold table `car_sales_hierarchy_filtered`.
        - Verify critical columns (e.g. `Date`, `Dealer_Region`, `Company`) are non-null.
        - Standardize column names (removing special characters like `$`, parentheses, spaces).
        - Write validation tables in Delta format.
        """)
        st.code("""
# Column normalization snippet from 1_Ingestion.ipynb
clean_columns = [
    col_name.strip()
    .replace(" ", "_")
    .replace("(", "")
    .replace(")", "")
    .replace("$", "")
    .replace("-", "_")
    for col_name in df.columns
]
df = df.toDF(*clean_columns)
        """, language="python")
        
    with tab_prep:
        st.markdown("""
        **Objective:** Convert raw transactional data into aggregated monthly hierarchies and remove sparse/low-volume combinations.
        
        **Key Steps:**
        - Aggregate transactions into monthly granularity (`ds = trunc(date, 'month')`).
        - Generate groupings and concat names to create hierarchical IDs (e.g. `Region/{region}/Company/{company}`).
        - Remove series with less than 12 active months to avoid model instability from sparse data.
        - Overwrite gold validation tables.
        """)
        st.code("""
# Filtering out sparse time series in 2_Preprocessing.ipynb
series_length = hier_df.groupBy("unique_id").count()
valid_series = series_length.filter(col("count") >= 12).select("unique_id")
hier_df = hier_df.join(valid_series, on="unique_id", how="inner")
        """, language="python")
        
    with tab_baseline:
        st.markdown("""
        **Objective:** Split train/validation/test sets, fit base forecasting models using Nixtla's `statsforecast`, and log runs via MLflow.
        
        **Key Steps:**
        - Train range: 2022-01-01 to 2024-12-31.
        - Validation range: 2025-01-01 to 2025-12-31.
        - Test range: 2026-01-01 to 2026-12-31.
        - Train AutoARIMA model on all series.
        - Save model outputs to MLflow model registry as PKL files.
        """)
        st.code("""
# StatsForecast AutoARIMA fitment in 3_Baseline.ipynb
sf = StatsForecast(
    models=[AutoARIMA(season_length=12)],
    freq="MS",
    n_jobs=-1
)
validation_forecast = sf.forecast(df=train_df, h=12)
        """, language="python")
        
    with tab_reconcile:
        st.markdown("""
        **Objective:** Generate a summation matrix and reconcile the base AutoARIMA forecasts using the Bottom-Up method.
        
        **Key Steps:**
        - Create the summation matrix $S$ and tags detailing the hierarchy mapping using `hierarchicalforecast.utils.aggregate`.
        - Fit AutoARIMA on all hierarchical levels (regions, brands, region/brand combinations, etc.).
        - Reconcile using `HierarchicalReconciliation` with `BottomUp()`.
        - Bottom-Up sum ensures that child predictions sum up exactly to parent level forecasts, maintaining mathematical consistency.
        """)
        st.code("""
# Reconcile forecasts using hierarchicalforecast in 3_Baseline.ipynb
reconciler = HierarchicalReconciliation(
    reconcilers=[BottomUp()]
)
reconciled_forecast = reconciler.reconcile(
    Y_hat_df=forecast_reconcile,
    S_df=S_df,
    tags=tags
)
        """, language="python")
