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

# Custom CSS for modern glassmorphism & premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Background */
    .stApp {
        background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%) !important;
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f19 !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Custom tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.4);
        padding: 6px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-weight: 600;
        border: none;
        padding: 0 16px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #3b82f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 22px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        margin-bottom: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        align-items: center;
        gap: 18px;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59, 130, 246, 0.4);
        box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
    }
    .metric-icon-box {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid rgba(59, 130, 246, 0.3);
        border-radius: 12px;
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .metric-info {
        flex-grow: 1;
    }
    .metric-title {
        font-size: 13px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #ffffff;
        margin: 4px 0 0 0;
        letter-spacing: -0.02em;
    }
    .metric-delta {
        font-size: 11px;
        color: #10b981;
        font-weight: 500;
        margin-top: 2px;
    }
    
    /* Header Block */
    .app-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e293b 100%);
        padding: 35px 40px;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    .app-header::before {
        content: "";
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(59, 130, 246, 0.2) 0%, rgba(0, 0, 0, 0) 70%);
        pointer-events: none;
    }
    .app-title {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
        letter-spacing: -0.03em;
    }
    .app-subtitle {
        font-size: 16px;
        color: #93c5fd;
        margin-top: 8px;
        font-weight: 400;
        opacity: 0.95;
    }
    
    /* Segment divider */
    .section-header {
        border-bottom: 2px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 8px;
        margin-bottom: 20px;
        font-weight: 700;
        color: #3b82f6;
    }
    
    /* Styled code containers */
    .stCodeBlock {
        background-color: #0b0f19 !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
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
st.sidebar.markdown("<div style='text-align: center; padding: 25px 0;'><h2 style='color:#3b82f6; font-weight:800; margin:0; letter-spacing:-0.03em;'>⚡ AUTOFCAST</h2><p style='color:#64748b; font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; margin-top:2px;'>Hierarchical Engine</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Workspace",
    ["Overview Dashboard", "Interactive Drilldown & EDA", "Forecast Simulation", "Pipeline & Code Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px;'>
    <p style='color: #94a3b8; font-size: 11px; font-weight:600; text-transform: uppercase; margin-bottom: 8px;'>Engine Diagnostics</p>
    <div style='display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;'><span style='color: #64748b;'>Transactions:</span><span style='color: #f8fafc; font-weight: 600;'>69,371</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;'><span style='color: #64748b;'>Unique Series:</span><span style='color: #f8fafc; font-weight: 600;'>4,548</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;'><span style='color: #64748b;'>Base Model:</span><span style='color: #3b82f6; font-weight: 600;'>AutoARIMA</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 13px; margin: 4px 0;'><span style='color: #64748b;'>Reconciliation:</span><span style='color: #10b981; font-weight: 600;'>Bottom-Up</span></div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🚗 Car Sales Hierarchical Forecasting</div>
    <div class="app-subtitle">A state-of-the-art interactive forecasting dashboard leveraging Nixtla's StatsForecast & Bottom-Up Reconciliation</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# TAB 1: OVERVIEW DASHBOARD
# ----------------------------------------------------
if app_mode == "Overview Dashboard":
    st.markdown("<h3 class='section-header'>📊 Executive Performance Overview</h3>", unsafe_allow_html=True)
    
    # KPI Calculations
    total_sales = len(df_raw)
    total_revenue = df_raw['price_'].sum()
    avg_price = df_raw['price_'].mean()
    top_region = df_raw['dealer_region'].value_counts().index[0]
    top_company = df_raw['company'].value_counts().index[0]
    
    # Row of Metrics
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Sales Volume</div>
                <div class="metric-value">{total_sales:,}</div>
                <div class="metric-delta">Gross Transactions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Total Revenue</div>
                <div class="metric-value">${total_revenue/1e6:.2f}M</div>
                <div class="metric-delta">Gross Bookings</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/><circle cx="12" cy="13" r="3"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Avg Price</div>
                <div class="metric-value">${avg_price:,.0f}</div>
                <div class="metric-delta">Average transaction</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a8 8 0 0 0-8 8c0 5.25 8 12 8 12s8-6.75 8-12a8 8 0 0 0-8-8z"/><circle cx="12" cy="10" r="3"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Top Region</div>
                <div class="metric-value" style="font-size: 24px; padding-top: 4px;">{top_region}</div>
                <div class="metric-delta">Market Share Leader</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Core Charts
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("#### Historical Sales Volatility & Trend")
        monthly_sales = df_raw.groupby('month_start').size().reset_index(name='sales')
        
        # Plotly layout with custom glass dark theme
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_sales['month_start'],
            y=monthly_sales['sales'],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.08)',
            line=dict(color='#3b82f6', width=3.5),
            name="Monthly Sales"
        ))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=10, r=10, t=10, b=10),
            height=360,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
            hovermode="x"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.markdown("#### Sales Distribution by Region")
        region_sales = df_raw['dealer_region'].value_counts().reset_index()
        region_sales.columns = ['region', 'sales']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=region_sales['region'],
            values=region_sales['sales'],
            hole=0.5,
            marker=dict(colors=['#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe', '#dbeafe']),
            textinfo='label+percent',
            textposition='inside',
            showlegend=False
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#cbd5e1',
            margin=dict(l=10, r=10, t=10, b=10),
            height=360
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Anomaly Detection Highlight Section
    st.markdown("#### 🚨 Historical Sales Anomaly Detection")
    sales_series = df_raw.groupby('month_start').size().reset_index(name='sales')
    sales_mean = sales_series['sales'].mean()
    sales_std = sales_series['sales'].std()
    
    # Define thresholds for outlier (z-score > 1.8)
    sales_series['z_score'] = (sales_series['sales'] - sales_mean) / sales_std
    anomalies = sales_series[sales_series['z_score'].abs() > 1.8]
    
    if len(anomalies) > 0:
        anom_cols = st.columns(len(anomalies))
        for idx, row in enumerate(anomalies.itertuples()):
            is_high = row.z_score > 0
            color = "#10b981" if is_high else "#ef4444"
            arrow = "↗" if is_high else "↘"
            status = "Spike" if is_high else "Drop"
            with anom_cols[idx]:
                st.markdown(f"""
                <div style="background: rgba(15,23,42,0.5); border: 1px solid {color}40; border-radius: 12px; padding: 15px; text-align: center;">
                    <span style="color: {color}; font-weight: 800; font-size: 14px;">{arrow} {status} Detected</span>
                    <h5 style="margin: 6px 0; color: #f8fafc;">{row.month_start.strftime('%B %Y')}</h5>
                    <span style="font-size: 12px; color: #94a3b8;">Sales: <strong>{row.sales}</strong> ({row.z_score:+.2f} SD)</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No statistical outliers detected in the historical period.")


# ----------------------------------------------------
# TAB 2: INTERACTIVE DRILLDOWN & EDA
# ----------------------------------------------------
elif app_mode == "Interactive Drilldown & EDA":
    st.markdown("<h3 class='section-header'>🌳 Interactive Drilldown Hierarchy</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    Use the **Interactive Treemap** below to explore the structure of the car sales. 
    Click on any **Region** block to zoom into specific **Brands (Companies)**, and click again to drill down to **Body Styles**.
    """)
    
    # Treemap Chart
    fig_tree = px.treemap(
        df_raw,
        path=['dealer_region', 'company', 'body_style'],
        values='price_',
        color='price_',
        color_continuous_scale='Blues',
        labels={'price_': 'Total Revenue ($)', 'dealer_region': 'Region', 'company': 'Brand', 'body_style': 'Body Style'}
    )
    fig_tree.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#cbd5e1',
        margin=dict(l=10, r=10, t=10, b=10),
        height=550
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Detailed segments
    st.markdown("#### Segment breakdown matrix")
    s1, s2 = st.columns(2)
    
    with s1:
        st.markdown("##### Top 10 Vehicle Companies by Net Revenue")
        top_companies = df_raw.groupby('company')['price_'].sum().reset_index().sort_values('price_', ascending=False).head(10)
        fig_c = px.bar(top_companies, x='price_', y='company', orientation='h', color='price_', color_continuous_scale='Blues')
        fig_c.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=300, coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_c, use_container_width=True)
        
    with s2:
        st.markdown("##### Brand Price Distribution Analysis")
        top_companies_list = top_companies['company'].tolist()
        df_top_c = df_raw[df_raw['company'].isin(top_companies_list)]
        fig_box = px.box(df_top_c, x='company', y='price_', color='company', color_discrete_sequence=px.colors.qualitative.Safe)
        fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#cbd5e1', height=300, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)


# ----------------------------------------------------
# TAB 3: FORECAST SIMULATION
# ----------------------------------------------------
elif app_mode == "Forecast Simulation":
    st.markdown("<h3 class='section-header'>📈 Forecast Viewer & Scenario Planner</h3>", unsafe_allow_html=True)
    
    if df_forecast is None:
        st.error("Missing prediction output file (`reconciled_forecast.csv`). Please run the pipeline script.")
    else:
        # Dynamic hierarchy selector panel
        st.markdown("#### Define Selector Parameters")
        
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

        st.markdown(f"Currently Analyzing Series: ` {selected_unique_id} `")
        
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
            st.error(f"⚠️ Could not find forecast predictions for series `{selected_unique_id}`. This series might have been pruned as sparse (under 12 active sales months) during preprocessing.")
        else:
            # ----------------------------------------------------
            # INTERACTIVE SCENARIO SIMULATOR
            # ----------------------------------------------------
            st.markdown("#### ⚡ Real-Time Scenario Planner")
            
            sim_col_1, sim_col_2 = st.columns(2)
            with sim_col_1:
                growth_slider = st.slider(
                    "Simulated Growth Shift (%)",
                    min_value=-50,
                    max_value=50,
                    value=0,
                    step=5,
                    help="Shift the reconciled forecast path upwards or downwards based on mock economic scenarios."
                )
            with sim_col_2:
                noise_slider = st.slider(
                    "Add Scenario Volatility (%)",
                    min_value=0,
                    max_value=20,
                    value=0,
                    step=2,
                    help="Introduce random market volatility into the simulated path."
                )
            
            # Apply growth and noise simulated projections
            sim_ratio = 1 + (growth_slider / 100.0)
            np.random.seed(42)  # Maintain stable noise paths
            noise_factors = 1.0 + (np.random.normal(0, noise_slider / 100.0, len(series_fcst)))
            
            series_fcst['Simulated'] = series_fcst['AutoARIMA/BottomUp'] * sim_ratio * noise_factors
            
            # Re-plot interactive charts
            fig = go.Figure()
            
            # History
            fig.add_trace(go.Scatter(
                x=hist_monthly['ds'],
                y=hist_monthly['y'],
                name='Historical Sales',
                line=dict(color='#3b82f6', width=3),
                mode='lines+markers'
            ))
            
            # Unreconciled AutoARIMA
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA'],
                name='Base AutoARIMA Forecast',
                line=dict(color='#f97316', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            # Reconciled BottomUp
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA/BottomUp'],
                name='Reconciled Bottom-Up Forecast',
                line=dict(color='#10b981', width=3),
                mode='lines+markers'
            ))
            
            # Simulated Scenario
            if growth_slider != 0 or noise_slider != 0:
                fig.add_trace(go.Scatter(
                    x=series_fcst['ds'],
                    y=series_fcst['Simulated'],
                    name='Simulated Scenario Path',
                    line=dict(color='#ec4899', width=3, dash='dot'),
                    mode='lines+markers'
                ))
            
            fig.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Monthly Sales Volume",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#cbd5e1',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.04)'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=480
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show summary stats
            fcst_sum_reconciled = series_fcst['AutoARIMA/BottomUp'].sum()
            fcst_sum_simulated = series_fcst['Simulated'].sum()
            hist_sum_last_year = hist_monthly[hist_monthly['ds'] >= '2026-01-01']['y'].sum()
            
            s_col1, s_col2, s_col3 = st.columns(3)
            with s_col1:
                st.metric("Historical Sales (2026)", f"{int(hist_sum_last_year)} cars")
            with s_col2:
                st.metric("Reconciled Forecast (2027)", f"{int(round(fcst_sum_reconciled))} cars")
            with s_col3:
                delta_sim = fcst_sum_simulated - fcst_sum_reconciled
                st.metric(
                    "Simulated Scenario Total", 
                    f"{int(round(fcst_sum_simulated))} cars",
                    delta=f"{delta_sim:+.0f} vs base reconciled"
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
    st.markdown("<h3 class='section-header'>🛠️ Data Pipeline & Code Architecture</h3>", unsafe_allow_html=True)
    
    st.markdown("""
    This project was originally developed inside **Databricks** as a hierarchical time series forecasting pipeline. 
    Below is a breakdown of the three notebooks that make up the pipeline and how they map to this web dashboard.
    """)
    
    tab_ingest, tab_prep, tab_baseline, tab_reconcile = st.tabs([
        "📥 1. Ingestion.ipynb",
        "⚙️ 2. Preprocessing.ipynb",
        "🤖 3. Baseline.ipynb (Base Models)",
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
# Column normalization snippet from Ingestion.ipynb
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
# Filtering out sparse time series in Preprocessing.ipynb
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
# StatsForecast AutoARIMA fitment in Baseline.ipynb
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
# Reconcile forecasts using hierarchicalforecast in Baseline.ipynb
reconciler = HierarchicalReconciliation(
    reconcilers=[BottomUp()]
)
reconciled_forecast = reconciler.reconcile(
    Y_hat_df=forecast_reconcile,
    S_df=S_df,
    tags=tags
)
        """, language="python")
