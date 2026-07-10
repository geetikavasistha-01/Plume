import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import plotly.graph_objects as go
from dashboard.icon_utils import inject_material_icons, icon

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

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

# Helper function to load wind rose
@st.cache_data
def load_wind_rose(wind_rose_path):
    if not os.path.exists(wind_rose_path):
        return None
    with open(wind_rose_path, 'r') as f:
        return json.load(f)

def show(selected_region, region_slug, bbox, paths):
    st.markdown(custom_css, unsafe_allow_html=True)
    inject_material_icons(st)
    
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
            st.markdown("### :material/satellite_alt: Ingestion")
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
            st.markdown("### :material/calculate: Preprocess")
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
            st.markdown("### :material/psychology: Forecasting")
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
            st.markdown("### :material/travel_explore: Anomalies")
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
                
                st.markdown(f"""
                <div class='card' style='border:1px solid #45a29e;'>
                    <h4 style='color:#66fcf1;margin-top:0;'>Representative Grid Cell AQI Breakout</h4>
                    <p style='font-size:0.85rem;color:#8b8c8d;margin-bottom:1rem;'>Coordinates: <code>{lat_coord:.4f}°N, {lon_coord:.4f}°E</code></p>
                    <table style='width:100%; border-collapse:collapse; text-align:left; font-size:0.9rem;'>
                        <tr style='border-bottom:1px solid rgba(69, 162, 158, 0.2);'>
                            <th style='padding:0.4rem 0;'>Parameter</th>
                            <th style='padding:0.4rem 0;'>Observation</th>
                            <th style='padding:0.4rem 0;'>Sub-Index</th>
                        </tr>
                        <tr style='border-bottom:1px solid rgba(69, 162, 158, 0.1);'>
                            <td style='padding:0.4rem 0;'>NO₂ Column</td>
                            <td><code>{no2_val:.2e}</code> mol/m²</td>
                            <td><b>{i_no2:.1f}</b></td>
                        </tr>
                        <tr style='border-bottom:1px solid rgba(69, 162, 158, 0.1);'>
                            <td style='padding:0.4rem 0;'>HCHO Column</td>
                            <td><code>{hcho_val:.2e}</code> mol/m²</td>
                            <td><b>{i_hcho:.1f}</b></td>
                        </tr>
                        <tr style='border-bottom:1px solid rgba(69, 162, 158, 0.1);'>
                            <td style='padding:0.4rem 0;'>Aero. Index (AOD)</td>
                            <td><code>{aod_val:.2f}</code></td>
                            <td><b>{i_aod:.1f}</b></td>
                        </tr>
                        <tr style='background:rgba(102, 252, 241, 0.05);'>
                            <td style='padding:0.5rem 0; font-weight:600;'>Proxy AQI</td>
                            <td>-</td>
                            <td style='color:#66fcf1;'><b>{aqi_val:.1f}</b></td>
                        </tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.caption(f"Could not load latest representative cell math: {e}")
        else:
            st.info("Load grid data first to see representative sub-index math.")
            
    st.divider()
    
    # 4. Wind Rose Math & Advection Equations
    st.markdown("<div class='methodology-header'>Spatio-Temporal Wind Transport Advection</div>", unsafe_allow_html=True)
    
    left_wr, right_traj = st.columns([1, 1])
    
    df_wr = None
    wr_data = load_wind_rose(paths['wind_rose'])
    if wr_data is not None:
        df_wr = pd.DataFrame(wr_data.get('wind_rose', []))
        
    with left_wr:
        st.markdown("""
        We analyze meteorological wind transport by correlating regional predicted AQI against daily local wind directions. 
        The **Wind Rose Analysis** aggregates wind directions into 8 primary sectors:
        """)
        
        if df_wr is not None:
            fig_wr = go.Figure()
            fig_wr.add_trace(go.Barpolar(
                r=df_wr['frequency'] * 100,
                theta=df_wr['direction'],
                name='Wind Directions',
                marker=dict(
                    color=df_wr['mean_aqi'],
                    colorscale='Cividis',
                    colorbar=dict(title='Mean AQI', orientation='h', y=-0.2),
                    showscale=True
                )
            ))
            fig_wr.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2833',
                plot_bgcolor='#1f2833',
                polar=dict(
                    radialaxis=dict(showticklabels=True, ticks=''),
                    angularaxis=dict(direction='clockwise', period=8)
                ),
                height=300,
                margin=dict(l=30, r=30, t=30, b=30)
            )
            st.plotly_chart(fig_wr, use_container_width=True)
        else:
            st.info("Wind Rose data not available for this region. Run transport analysis.")
            
    with right_traj:
        st.markdown("""
        We also compute simplified **24-hour advection trajectories** for detected chemical hotspots to identify downwind dispersion and likely upwind sources. 
        Using the U/V wind speed components (in m/s), coordinates are projected over a 24-hour timeframe:
        """)
        st.latex(r"\Delta x_{\text{deg}} = \frac{u \times 86400}{111000 \times \cos(\text{lat})}, \quad \Delta y_{\text{deg}} = \frac{v \times 86400}{111000}")
        
        if df_wr is not None:
            trajectories = wr_data.get('hotspot_trajectories', [])
            if len(trajectories) > 0:
                df_traj = pd.DataFrame(trajectories)
                df_traj_show = df_traj[['location', 'u', 'v', 'fwd_lat', 'fwd_lon']].copy()
                df_traj_show.columns = ['Nearest Location', 'U (m/s)', 'V (m/s)', 'Fwd Lat', 'Fwd Lon']
                df_traj_show['U (m/s)'] = df_traj_show['U (m/s)'].map(lambda x: f"{x:.2f}")
                df_traj_show['V (m/s)'] = df_traj_show['V (m/s)'].map(lambda x: f"{x:.2f}")
                df_traj_show['Fwd Lat'] = df_traj_show['Fwd Lat'].map(lambda x: f"{x:.3f}")
                df_traj_show['Fwd Lon'] = df_traj_show['Fwd Lon'].map(lambda x: f"{x:.3f}")
                
                st.dataframe(df_traj_show, use_container_width=True, hide_index=True)
            else:
                st.info("No trajectories available for this region.")
        else:
            st.info("Trajectory advection data not loaded.")
            
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
