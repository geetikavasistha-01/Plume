import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

# Set page config
st.set_page_config(
    page_title="Methodology",
    page_icon="📖",
    layout="wide"
)

# Custom CSS for dark theme
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0b0c10;
        color: #c5c6c7;
    }
    
    .page-title {
        background: linear-gradient(45deg, #45a29e, #66fcf1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    .methodology-header {
        color: #66fcf1;
        font-weight: 600;
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid rgba(69, 162, 158, 0.3);
        padding-bottom: 0.3rem;
    }
    
    /* LaTeX custom spacing */
    .katex-display {
        background: rgba(31, 40, 51, 0.4);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(69, 162, 158, 0.15);
        margin: 1rem 0;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Import helper
from dashboard import helper

# Render sidebar region selector and get selected region
selected_region, region_slug, bbox, paths = helper.render_sidebar_and_get_region()

st.markdown("<div class='page-title'>Methodology & Architecture</div>", unsafe_allow_html=True)

# Model training caveat note
st.info("Note: The CNN-LSTM forecasting model was trained on Delhi-NCR data. Other regions are processed in an inference-only capacity using this pre-trained model (transfer learning), which may carry lower spatial prediction accuracy.")

# 1. Pipeline diagram description
st.markdown("<div class='methodology-header'>Data & Model Pipeline Workflow</div>", unsafe_allow_html=True)
st.markdown("The end-to-end data ingestion, preprocessing, forecasting, and anomaly detection workflow operates on the following structure:")

# Load the pipeline flowchart image
assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
image_path = os.path.join(assets_dir, "pipeline_flowchart.jpg")
st.image(image_path, caption="End-to-End ISRO AQI & HCHO Pipeline Workflow Diagram", use_container_width=True)

st.divider()

# 2. Pipeline Stages in columns
st.markdown("<div class='methodology-header'>Pipeline Stages Breakdown</div>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    with st.container(border=True):
        st.markdown("### 🛰️ Ingestion")
        st.markdown("**GEE Download**")
        st.markdown("Queries raw collections on a 0.05° resolution.")
        with st.expander("Technical Details"):
            st.markdown("""
            - Pulls Sentinel-5P TROPOMI OFFL level-3 collections.
            - Pulls ECMWF ERA5 Land hourly reanalysis.
            - Daily composites are computed and clipped to Delhi-NCR bounds.
            """)
            
with col2:
    with st.container(border=True):
        st.markdown("### 🧮 Preprocess")
        st.markdown("**Grid Alignment**")
        st.markdown("Interpolates layers and generates sequences.")
        with st.expander("Details"):
            st.markdown("""
            - Fills data gaps and masks clouds via nearest-neighbor interpolation.
            - Calculates the Proxy AQI target based on standardized pollutant thresholds.
            - Normalizes variables using z-score values and structures a 7-day lookback tensor.
            """)
            
with col3:
    with st.container(border=True):
        st.markdown("### 🧠 Forecasting")
        st.markdown("**CNN-LSTM Network**")
        st.markdown("Runs next-day spatial regression forecasts.")
        with st.expander("Details"):
            st.markdown("""
            - Uses 1D Convolutional layers to capture short-term meteorological trends.
            - An LSTM layer integrates sequence states to model atmospheric accumulation.
            - Outputs predicted gridded AQI values.
            """)
            
with col4:
    with st.container(border=True):
        st.markdown("### 🔍 Anomalies")
        st.markdown("**Isolation Forest**")
        st.markdown("Identifies high-density chemical hotspots.")
        with st.expander("Details"):
            st.markdown("""
            - Fits an Isolation Forest model on the daily multi-pollutant space [HCHO, NO2, AOD].
            - Flags spatial outliers with short path lengths.
            - Discards low-concentration outliers to focus exclusively on highly polluted areas.
            """)

st.divider()

# 3. Proxy AQI Target Formulation
st.markdown("<div class='methodology-header'>Proxy AQI Formulation & Sub-Indices</div>", unsafe_allow_html=True)

left_math, right_chart = st.columns([1, 1])

with left_math:
    st.markdown("""
    Since real-time monitoring station datasets are sparse, the target Air Quality Index is calculated as a **Proxy Target** using ambient column densities. 
    First, individual pollutant values are scaled to normalized sub-indices ($I_{\text{pollutant}}$) between $0$ and $100+$:
    """)
    
    st.latex(r"I_{\text{NO}_2} = \min\left(\frac{\text{NO}_2}{0.0003}, 1.0\right) \times 150.0")
    st.latex(r"I_{\text{HCHO}} = \min\left(\frac{\text{HCHO}}{0.0004}, 1.0\right) \times 100.0")
    st.latex(r"I_{\text{AOD}} = \min\left(\frac{\text{AOD} + 0.5}{3.5}, 1.0\right) \times 200.0")
    
    st.markdown("The final **Proxy AQI** combines these sub-indices using a weighted-maximum aggregation:")
    st.latex(r"\text{Proxy AQI} = \text{Clip}\left(\max\left(I_{\text{NO}_2}, I_{\text{AOD}}\right) + 0.15 \times I_{\text{HCHO}}, 10.0, 500.0\right)")

with right_chart:
    processed_path = paths['processed_grid']
    ds = None
    if os.path.exists(processed_path):
        try:
            ds = xr.open_dataset(processed_path)
        except Exception:
            pass
            
    if ds is not None:
        try:
            # Find the date index with non-zero HCHO
            time_idx = 0
            for idx in range(len(ds.time) - 1, -1, -1):
                if float(ds.isel(time=idx)['hcho'].mean()) > 1e-7:
                    time_idx = idx
                    break
            ds_latest = ds.isel(time=time_idx)
            
            # Find index of max HCHO to display a representative hotspot cell
            hcho_grid = ds_latest['hcho'].values
            flat_idx = np.nanargmax(hcho_grid)
            lat_idx, lon_idx = np.unravel_index(flat_idx, hcho_grid.shape)
            
            hcho_val = float(ds_latest['hcho'].values[lat_idx, lon_idx])
            no2_val = float(ds_latest['no2'].values[lat_idx, lon_idx])
            aod_val = float(ds_latest['aod'].values[lat_idx, lon_idx])
            aqi_val = float(ds_latest['aqi'].values[lat_idx, lon_idx])
            
            # Subindices
            i_no2 = min(no2_val / 0.0003, 1.0) * 150.0
            i_hcho = min(hcho_val / 0.0004, 1.0) * 100.0
            i_aod = min((aod_val + 0.5) / 3.5, 1.0) * 200.0
            
            lat_coord = float(ds_latest.lat.values[lat_idx])
            lon_coord = float(ds_latest.lon.values[lon_idx])
            cell_label = f"Real Hotspot (Lat {lat_coord:.3f}, Lon {lon_coord:.3f})"
        except Exception:
            ds = None
            
    if ds is None:
        # Illustrative values
        i_no2 = 82.5
        i_hcho = 62.0
        i_aod = 145.0
        aqi_val = max(i_no2, i_aod) + 0.15 * i_hcho
        cell_label = "Illustrative Hotspot Cell"
        
    # Plotly horizontal bar chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=["NO₂ Sub-Index", "HCHO Sub-Index", "AOD Sub-Index", "Overall Proxy AQI"],
        x=[i_no2, i_hcho, i_aod, aqi_val],
        orientation='h',
        marker=dict(
            color=['#45a29e', '#c77dff', '#ff7e00', '#66fcf1'],
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        text=[f"{i_no2:.1f}", f"{i_hcho:.1f}", f"{i_aod:.1f}", f"{aqi_val:.1f}"],
        textposition='auto',
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#1f2833',
        plot_bgcolor='#1f2833',
        margin=dict(l=40, r=40, t=50, b=40),
        title=dict(
            text=f"Sub-Index Composition: {cell_label}",
            font=dict(size=13, color='#66fcf1')
        ),
        xaxis_title='Value / Index Points',
        height=280
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# 4. Model Architecture Diagram
st.markdown("<div class='methodology-header'>Deep Learning Architecture (CNN-1D + LSTM)</div>", unsafe_allow_html=True)

left_desc, right_arch = st.columns([1, 1])

with left_desc:
    st.markdown("""
    To forecast tomorrow's gridded AQI, the temporal sequence of each grid cell must be processed. We implement a convolutional-recurrent model:
    
    1. **Conv1D Layer:** Slides a 1D filter kernel (size 3) over the 7 days of historical inputs to extract local gradients (e.g. wind shifts, cooling/warming periods).
    2. **Regularization:** Batch Normalization stabilizes training variables, and a Dropout layer (0.1) prevents dependency memorization.
    3. **LSTM Layer:** Integrates the intermediate sequence outputs over the 7-day window to model atmospheric accumulation or decay.
    4. **Dense Layer:** Maps recurrent states to a hidden layer of size 16.
    5. **Output Head:** A single linear dense unit generates the continuous AQI forecast value.
    """)

with right_arch:
    arch_image_path = os.path.join(assets_dir, "model_architecture.png")
    st.image(arch_image_path, caption="CNN-1D + LSTM Model Architecture Details", use_container_width=True)

st.divider()

# 5. Limitations & Future Work formatted
st.markdown("<div class='methodology-header'>Limitations & Future Extensions</div>", unsafe_allow_html=True)

col_limit, col_future = st.columns(2)

with col_limit:
    st.markdown("<h4 style='color:#c77dff;'>Current Limitations</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **Proxy-Only Validation:** The target variable is an atmospheric proxy and has not been cross-calibrated against ground-level CPCB monitor stations.
    - **Aerosol Index Substitution:** Sentinel-5P absorbing aerosol index is used in place of direct INSAT-3D Aerosol Optical Depth (AOD).
    - **Data Pipeline Latency:** Sentinel-5P L3 offline collections exhibit a 3-5 day latency in GEE, meaning active hotspot models use observations offset by several days.
    """)

with col_future:
    st.markdown("<h4 style='color:#66fcf1;'>Future Extensions</h4>", unsafe_allow_html=True)
    st.markdown("""
    - **CPCB Station Alignment:** Ingest real-time PM2.5 and PM10 measurements from CPCB APIs (data.gov.in) to calibrate GEE proxies.
    - **INSAT-3D Integration:** Link direct INSAT-3D hourly AOD measurements once available to enhance aerosol inputs.
    - **National-Scale Expansion:** Deploy the gridding and forecasting layers over a national India bounding frame.
    - **Decoupled serving layer:** Expose predictions via a FastAPI endpoint for external pipeline integrations.
    """)
