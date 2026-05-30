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

# Custom CSS for subtle, minimalist & high-end UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Layout Background */
    .stApp {
        background-color: #090d16 !important;
        color: #cbd5e1;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #05070b !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Custom tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #131924;
        padding: 5px;
        border-radius: 8px;
        border: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 38px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #64748b;
        font-weight: 500;
        font-size: 14px;
        border: none;
        padding: 0 14px;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #38bdf8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        color: #38bdf8 !important;
        box-shadow: none;
        border: 1px solid rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Subtle Metric Cards */
    .metric-card {
        background: #131924;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .metric-card:hover {
        border-color: #334155;
        background: #18202e;
    }
    .metric-icon-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        width: 42px;
        height: 42px;
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
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #f1f5f9;
        margin: 4px 0 0 0;
        letter-spacing: -0.01em;
    }
    .metric-delta {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
    }
    
    /* Clean Minimalist Header */
    .app-header {
        background: #131924;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        border: 1px solid #1e293b;
    }
    .app-title {
        font-size: 30px;
        font-weight: 700;
        color: #f1f5f9;
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
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
        margin-bottom: 18px;
        font-size: 18px;
        font-weight: 600;
        color: #94a3b8;
    }
    
    /* Styled code block borders */
    .stCodeBlock {
        background-color: #05070b !important;
        border-radius: 8px;
        border: 1px solid #1e293b;
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
st.sidebar.markdown("<div style='text-align: center; padding: 20px 0;'><h3 style='color:#f8fafc; font-weight:700; margin:0; letter-spacing:-0.02em;'>🚗 AUTOFCAST</h3><p style='color:#475569; font-size:10px; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-top:2px;'>Hierarchical Engine</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation Workspace",
    ["Overview Dashboard", "Interactive Drilldown & EDA", "Forecast Simulation", "Pipeline & Code Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='background: #131924; border: 1px solid #1e293b; border-radius: 8px; padding: 14px;'>
    <p style='color: #475569; font-size: 10px; font-weight:600; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.05em;'>Engine Diagnostics</p>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Transactions:</span><span style='color: #cbd5e1; font-weight: 500;'>69,371</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Unique Series:</span><span style='color: #cbd5e1; font-weight: 500;'>4,548</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Base Model:</span><span style='color: #94a3b8; font-weight: 500;'>AutoARIMA</span></div>
    <div style='display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0;'><span style='color: #64748b;'>Reconciliation:</span><span style='color: #38bdf8; font-weight: 500;'>Bottom-Up</span></div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# MAIN HEADER
# ----------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🚗 Car Sales Hierarchical Forecasting</div>
    <div class="app-subtitle">A sleek, mathematical forecasting pipeline leveraging Nixtla's StatsForecast & Bottom-Up Reconciliation</div>
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
    
    # Row of Metrics (Minimalist style)
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9l-1.4 2.9A3.7 3.7 0 0 0 2 12v4c0 .6.4 1 1 1h2"/><circle cx="7" cy="17" r="2"/><path d="M9 17h6"/><circle cx="17" cy="17" r="2"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Sales Volume</div>
                <div class="metric-value">{total_sales:,}</div>
                <div class="metric-delta">Gross units</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"/><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
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
            <div class="metric-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/><circle cx="12" cy="13" r="3"/></svg>
            </div>
            <div class="metric-info">
                <div class="metric-title">Avg Price</div>
                <div class="metric-value">${avg_price:,.0f}</div>
                <div class="metric-delta">Per vehicle</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-icon-box">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a8 8 0 0 0-8 8c0 5.25 8 12 8 12s8-6.75 8-12a8 8 0 0 0-8-8z"/><circle cx="12" cy="10" r="3"/></svg>
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
        st.markdown("##### Historical Monthly Sales Volume")
        monthly_sales = df_raw.groupby('month_start').size().reset_index(name='sales')
        
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly_sales['month_start'],
            y=monthly_sales['sales'],
            mode='lines',
            line=dict(color='#475569', width=2), # Subtle gray line
            name="Monthly Sales"
        ))
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
            hovermode="x"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        st.markdown("##### Sales Share by Region")
        region_sales = df_raw['dealer_region'].value_counts().reset_index()
        region_sales.columns = ['region', 'sales']
        
        # Muted layout colors
        fig_pie = go.Figure(data=[go.Pie(
            labels=region_sales['region'],
            values=region_sales['sales'],
            hole=0.6,
            marker=dict(colors=['#1e293b', '#334155', '#475569', '#64748b', '#94a3b8']), # Muted grayscale/steel tones
            textinfo='percent',
            textposition='inside',
            showlegend=True
        )])
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Anomaly Detection Highlight Section (Subtle slate panels)
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
            color = "#38bdf8" if is_high else "#94a3b8"
            arrow = "↑" if is_high else "↓"
            status = "Spike" if is_high else "Dip"
            with anom_cols[idx]:
                st.markdown(f"""
                <div style="background: #131924; border: 1px solid #1e293b; border-left: 3px solid {color}; border-radius: 6px; padding: 12px; text-align: left;">
                    <div style="font-size: 11px; color: #64748b; font-weight: 600; text-transform: uppercase;">{status} Detected</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f1f5f9; margin-top: 2px;">{row.month_start.strftime('%B %Y')}</div>
                    <div style="font-size: 12px; color: #cbd5e1; margin-top: 4px;">Sales: {row.sales} <span style="color: {color};">({row.z_score:+.1f} SD)</span></div>
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
    Explore the hierarchical distribution of sales. Click on any Region tile to drill down into Brands (Companies) and Body Styles.
    """)
    
    # Treemap Chart (Using custom subtle color scale)
    fig_tree = px.treemap(
        df_raw,
        path=['dealer_region', 'company', 'body_style'],
        values='price_',
        color='price_',
        color_continuous_scale=[[0, '#131924'], [0.5, '#475569'], [1.0, '#38bdf8']], # Custom dark to steel-blue scale
        labels={'price_': 'Total Revenue ($)', 'dealer_region': 'Region', 'company': 'Brand', 'body_style': 'Body Style'}
    )
    fig_tree.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        margin=dict(l=10, r=10, t=10, b=10),
        height=500
    )
    st.plotly_chart(fig_tree, use_container_width=True)

    # Detailed segments
    s1, s2 = st.columns(2)
    
    with s1:
        st.markdown("##### Top 10 Vehicle Brands by Revenue")
        top_companies = df_raw.groupby('company')['price_'].sum().reset_index().sort_values('price_', ascending=False).head(10)
        fig_c = px.bar(top_companies, x='price_', y='company', orientation='h', color='price_', color_continuous_scale='Cividis')
        fig_c.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#94a3b8', 
            height=260, 
            coloraxis_showscale=False, 
            yaxis=dict(autorange="reversed"),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig_c, use_container_width=True)
        
    with s2:
        st.markdown("##### Brand Price Distribution Matrix")
        top_companies_list = top_companies['company'].tolist()
        df_top_c = df_raw[df_raw['company'].isin(top_companies_list)]
        fig_box = px.box(df_top_c, x='company', y='price_', color_discrete_sequence=['#475569'])
        fig_box.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font_color='#94a3b8', 
            height=260, 
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)')
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
            # MINIMALIST SCENARIO SIMULATOR
            # ----------------------------------------------------
            st.markdown("##### ⚡ Real-Time Scenario Planner")
            
            sim_col_1, sim_col_2 = st.columns(2)
            with sim_col_1:
                growth_slider = st.slider(
                    "Simulated Growth Shift (%)",
                    min_value=-55,
                    max_value=55,
                    value=0,
                    step=5,
                    help="Shift the reconciled forecast path."
                )
            with sim_col_2:
                noise_slider = st.slider(
                    "Add Volatility (%)",
                    min_value=0,
                    max_value=20,
                    value=0,
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
            
            # History (Steel gray)
            fig.add_trace(go.Scatter(
                x=hist_monthly['ds'],
                y=hist_monthly['y'],
                name='Historical Sales',
                line=dict(color='#64748b', width=2),
                mode='lines+markers',
                marker=dict(size=4)
            ))
            
            # Base AutoARIMA (Dashed slate)
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA'],
                name='Base AutoARIMA Forecast',
                line=dict(color='#475569', width=1.5, dash='dash'),
                mode='lines'
            ))
            
            # Reconciled BottomUp (Sleek sky blue)
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA/BottomUp'],
                name='Reconciled Forecast',
                line=dict(color='#38bdf8', width=2.5),
                mode='lines+markers',
                marker=dict(size=5)
            ))
            
            # Simulated Scenario (Subtle soft purple)
            if growth_slider != 0 or noise_slider != 0:
                fig.add_trace(go.Scatter(
                    x=series_fcst['ds'],
                    y=series_fcst['Simulated'],
                    name='Simulated Scenario',
                    line=dict(color='#818cf8', width=2, dash='dot'),
                    mode='lines+markers',
                    marker=dict(size=4)
                ))
            
            fig.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Monthly Sales Volume",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.03)'),
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
                    delta_color="off" # Grey delta instead of red/green
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
