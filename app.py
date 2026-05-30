import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page config
st.set_page_config(
    page_title="Car Sales Hierarchical Forecasting",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism & premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid #334155;
    }
    
    /* Card design */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        margin-bottom: 20px;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: rgba(59, 130, 246, 0.5);
    }
    .metric-title {
        font-size: 14px;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 8px;
    }
    .metric-delta {
        font-size: 12px;
        color: #10b981;
        margin-top: 4px;
    }
    
    /* App Title Header */
    .app-header {
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        padding: 40px;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .app-title {
        font-size: 40px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .app-subtitle {
        font-size: 18px;
        color: #93c5fd;
        margin-top: 8px;
        font-weight: 300;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# DATA LOADING (CACHED)
# ----------------------------------------------------

@st.cache_data
def load_raw_data():
    # Load only necessary columns to minimize memory footprint on Streamlit Cloud
    cols_to_use = ['Date', 'Dealer_Region', 'Company', 'Body Style', 'Model', 'Color', 'Price ($)']
    df = pd.read_csv("car_sales_extended_raw.csv", usecols=cols_to_use, low_memory=False)
    
    # Clean columns
    df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "").replace("$", "").replace("-", "_") for c in df.columns]
    
    # Parse dates
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
with st.spinner("Loading sales dataset..."):
    df_raw = load_raw_data()
    df_forecast = load_forecast_data()

# ----------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------
st.sidebar.markdown("<div style='text-align: center; padding: 20px 0;'><h2 style='color:#60a5fa; font-weight:800; margin:0;'>🚗 AUTOFCAST</h2><p style='color:#94a3b8; font-size:12px;'>Hierarchical Forecasting System</p></div>", unsafe_allow_html=True)
st.sidebar.markdown("---")

app_mode = st.sidebar.radio(
    "Navigation",
    ["Overview & EDA", "Hierarchical Forecast Viewer", "Pipeline & Code Architecture"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Dataset Info:**
- **Transactions:** 69,371 rows
- **Date Range:** Jan 2022 – Dec 2026
- **Hierarchy Levels:** 5 (Region, Brand, Combo, Model, Color)
- **Total Series:** 4,548 series
""")

# ----------------------------------------------------
# HEADER BLOCK
# ----------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">🚗 Car Sales Hierarchical Forecasting</div>
    <div class="app-subtitle">Interactive dashboard showcasing Databricks forecasting pipeline adapted to Streamlit Cloud</div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------
# MODE 1: OVERVIEW & EDA
# ----------------------------------------------------
if app_mode == "Overview & EDA":
    st.markdown("### 📊 Market Performance & Exploratory Analysis")
    
    # KPI metrics calculation
    total_sales = len(df_raw)
    total_revenue = df_raw['price_'].sum()
    avg_price = df_raw['price_'].mean()
    top_region = df_raw['dealer_region'].value_counts().index[0]
    top_company = df_raw['company'].value_counts().index[0]
    
    # Render KPI Cards in columns
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Sales</div>
            <div class="metric-value">{total_sales:,}</div>
            <div class="metric-delta">Transactions</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">${total_revenue/1e6:.2f}M</div>
            <div class="metric-delta">Gross revenue</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg Car Price</div>
            <div class="metric-value">${avg_price:,.2f}</div>
            <div class="metric-delta">Per vehicle</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Region</div>
            <div class="metric-value">{top_region}</div>
            <div class="metric-delta">Highest volume</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Top Brand</div>
            <div class="metric-value">{top_company}</div>
            <div class="metric-delta">Most popular</div>
        </div>
        """, unsafe_allow_html=True)

    # Charts section
    c1, c2 = st.columns([2, 1])
    
    with c1:
        # Historical monthly sales trend
        st.markdown("#### Monthly Sales Volume Trend")
        monthly_sales = df_raw.groupby('month_start').size().reset_index(name='sales')
        
        fig_trend = px.line(
            monthly_sales, 
            x='month_start', 
            y='sales', 
            labels={'month_start': 'Month', 'sales': 'Cars Sold'},
            color_discrete_sequence=['#3b82f6']
        )
        fig_trend.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            margin=dict(l=20, r=20, t=10, b=20),
            height=380
        )
        # Add smooth line & area
        fig_trend.data[0].update(mode='lines+markers', line=dict(width=3))
        st.plotly_chart(fig_trend, use_container_width=True)

    with c2:
        # Sales by Dealer Region
        st.markdown("#### Regional Sales Share")
        region_sales = df_raw['dealer_region'].value_counts().reset_index()
        region_sales.columns = ['region', 'sales']
        
        fig_region = px.pie(
            region_sales, 
            values='sales', 
            names='region',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig_region.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            margin=dict(l=10, r=10, t=10, b=10),
            height=380,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig_region, use_container_width=True)

    c3, c4 = st.columns([1, 1])
    
    with c3:
        # Top 15 Brands
        st.markdown("#### Top 15 Brands by Sales Volume")
        brand_sales = df_raw['company'].value_counts().head(15).reset_index()
        brand_sales.columns = ['brand', 'sales']
        
        fig_brand = px.bar(
            brand_sales,
            x='sales',
            y='brand',
            orientation='h',
            color='sales',
            color_continuous_scale='Blues',
            labels={'sales': 'Cars Sold', 'brand': 'Brand'}
        )
        fig_brand.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=True, gridcolor='#334155'),
            yaxis=dict(showgrid=False),
            margin=dict(l=20, r=20, t=10, b=20),
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_brand, use_container_width=True)
        
    with c4:
        # Sales by Body Style
        st.markdown("#### Sales by Body Style")
        body_sales = df_raw['body_style'].value_counts().reset_index()
        body_sales.columns = ['body_style', 'sales']
        
        fig_body = px.bar(
            body_sales,
            x='body_style',
            y='sales',
            color='sales',
            color_continuous_scale='Purples',
            labels={'sales': 'Cars Sold', 'body_style': 'Body Style'}
        )
        fig_body.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#f8fafc',
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#334155'),
            margin=dict(l=20, r=20, t=10, b=20),
            height=380,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_body, use_container_width=True)


# ----------------------------------------------------
# MODE 2: HIERARCHICAL FORECAST VIEWER
# ----------------------------------------------------
elif app_mode == "Hierarchical Forecast Viewer":
    st.markdown("### 📈 Hierarchical Forecasting & Reconciliation")
    
    if df_forecast is None:
        st.warning("⚠️ Reconciled forecast data file (`reconciled_forecast.csv`) is missing or has not been generated yet. Please run the forecasting pipeline first!")
    else:
        st.markdown("""
        Select a hierarchy level and narrow down your filters. The system will load the precomputed **AutoARIMA** forecasts 
        and compare them against the **Bottom-Up Reconciled** forecasts which align predictions across all hierarchy levels.
        """)
        
        # Selectbox to determine hierarchy depth
        level = st.selectbox(
            "Select Hierarchy Resolution Level",
            [
                "Region",
                "Company (Brand)",
                "Region & Company",
                "Region, Company & Model",
                "Region, Company, Model & Color (Leaf Node)"
            ]
        )
        
        # Populate dynamic selectors based on selected level
        selected_unique_id = ""
        
        if level == "Region":
            regions = sorted(df_raw['dealer_region'].unique().tolist())
            selected_region = st.selectbox("Select Dealer Region", regions)
            selected_unique_id = selected_region
            
        elif level == "Company (Brand)":
            companies = sorted(df_raw['company'].unique().tolist())
            selected_company = st.selectbox("Select Brand/Company", companies)
            selected_unique_id = selected_company
            
        elif level == "Region & Company":
            regions = sorted(df_raw['dealer_region'].unique().tolist())
            selected_region = st.selectbox("Select Dealer Region", regions)
            
            # Filter companies available in this region
            filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
            selected_company = st.selectbox("Select Brand/Company", filtered_companies)
            selected_unique_id = f"{selected_region}/{selected_company}"
            
        elif level == "Region, Company & Model":
            regions = sorted(df_raw['dealer_region'].unique().tolist())
            selected_region = st.selectbox("Select Dealer Region", regions)
            
            filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
            selected_company = st.selectbox("Select Brand/Company", filtered_companies)
            
            filtered_models = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company)]['model'].unique().tolist())
            selected_model = st.selectbox("Select Model", filtered_models)
            selected_unique_id = f"{selected_region}/{selected_company}/{selected_model}"
            
        elif level == "Region, Company, Model & Color (Leaf Node)":
            regions = sorted(df_raw['dealer_region'].unique().tolist())
            selected_region = st.selectbox("Select Dealer Region", regions)
            
            filtered_companies = sorted(df_raw[df_raw['dealer_region'] == selected_region]['company'].unique().tolist())
            selected_company = st.selectbox("Select Brand/Company", filtered_companies)
            
            filtered_models = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company)]['model'].unique().tolist())
            selected_model = st.selectbox("Select Model", filtered_models)
            
            filtered_colors = sorted(df_raw[(df_raw['dealer_region'] == selected_region) & (df_raw['company'] == selected_company) & (df_raw['model'] == selected_model)]['color'].unique().tolist())
            selected_color = st.selectbox("Select Color", filtered_colors)
            selected_unique_id = f"{selected_region}/{selected_company}/{selected_model}/{selected_color}"

        st.markdown(f"**Target Series unique_id:** `{selected_unique_id}`")
        
        # 1. Filter historical sales from raw data
        # We group by month to get actual monthly transaction counts
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
        
        # 2. Get forecast data for this series
        series_fcst = df_forecast[df_forecast['unique_id'] == selected_unique_id].sort_values('ds')
        
        if len(series_fcst) == 0:
            st.error(f"Could not find forecast predictions for series `{selected_unique_id}` in `reconciled_forecast.csv`. This series might have been filtered out as sparse (less than 12 months of sales) during preprocessing.")
        else:
            # Merging forecast and history (for final year overlap check or just listing)
            # Create Plotly Chart
            fig = go.Figure()
            
            # Historical Sales line
            fig.add_trace(go.Scatter(
                x=hist_monthly['ds'],
                y=hist_monthly['y'],
                name='Historical Sales',
                line=dict(color='#3b82f6', width=3),
                mode='lines+markers'
            ))
            
            # Base AutoARIMA Forecast
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA'],
                name='Base AutoARIMA Forecast',
                line=dict(color='#f97316', width=2, dash='dash'),
                mode='lines+markers'
            ))
            
            # Bottom-Up Reconciled Forecast
            fig.add_trace(go.Scatter(
                x=series_fcst['ds'],
                y=series_fcst['AutoARIMA/BottomUp'],
                name='Bottom-Up Reconciled Forecast',
                line=dict(color='#10b981', width=3),
                mode='lines+markers'
            ))
            
            fig.update_layout(
                title=f"Sales Forecast for '{selected_unique_id}' (Next 12 Months)",
                xaxis_title="Date",
                yaxis_title="Monthly Sales Volume (Cars Sold)",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#f8fafc',
                xaxis=dict(showgrid=True, gridcolor='#334155'),
                yaxis=dict(showgrid=True, gridcolor='#334155'),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Summary Metrics for next year
            fcst_sum_base = series_fcst['AutoARIMA'].sum()
            fcst_sum_reconciled = series_fcst['AutoARIMA/BottomUp'].sum()
            hist_sum_last_year = hist_monthly[hist_monthly['ds'] >= '2026-01-01']['y'].sum()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    label="Last Year Historical Sales (2026)",
                    value=f"{int(hist_sum_last_year)} cars"
                )
            with col2:
                growth_reconciled = ((fcst_sum_reconciled - hist_sum_last_year) / hist_sum_last_year * 100) if hist_sum_last_year > 0 else 0
                st.metric(
                    label="Reconciled Forecast (Next 12 Months)",
                    value=f"{int(round(fcst_sum_reconciled))} cars",
                    delta=f"{growth_reconciled:+.1f}% vs Last Year"
                )
            with col3:
                growth_base = ((fcst_sum_base - hist_sum_last_year) / hist_sum_last_year * 100) if hist_sum_last_year > 0 else 0
                st.metric(
                    label="Unreconciled AutoARIMA Forecast",
                    value=f"{int(round(fcst_sum_base))} cars",
                    delta=f"{growth_base:+.1f}% vs Last Year"
                )
                
            # Show predictions table
            st.markdown("#### Forecast Details (Raw Values)")
            display_df = series_fcst[['ds', 'AutoARIMA', 'AutoARIMA/BottomUp']].copy()
            display_df.columns = ['Date', 'Unreconciled AutoARIMA', 'Reconciled (Bottom-Up)']
            display_df['Date'] = display_df['Date'].dt.strftime('%b %Y')
            st.dataframe(display_df.style.format({
                'Unreconciled AutoARIMA': '{:,.2f}',
                'Reconciled (Bottom-Up)': '{:,.2f}'
            }), use_container_width=True)


# ----------------------------------------------------
# MODE 3: PIPELINE & CODE ARCHITECTURE
# ----------------------------------------------------
elif app_mode == "Pipeline & Code Architecture":
    st.markdown("### 🛠️ Data Pipeline & Code Architecture")
    
    st.markdown("""
    This project was originally developed inside **Databricks** as a hierarchical time series forecasting pipeline. 
    Below is a breakdown of the three notebooks that make up the pipeline and how they map to this web dashboard.
    """)
    
    # Carousel of code modules / explanations
    st.markdown("""
    ### 📂 Notebook Code Breakdown
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
        - Fit AutoARIMA on all hierarchical levels (individual regions, individual brands, region/brand combinations, etc.).
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
