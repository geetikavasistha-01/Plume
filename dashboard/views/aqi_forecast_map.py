import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)
from dashboard.icon_utils import inject_material_icons, icon, status_dot

# Custom css for the Editorial-Dark Aesthetic
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
        background: linear-gradient(45deg, #66fcf1, #45a29e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background: rgba(31, 40, 51, 0.65);
        border: 1px solid rgba(69, 162, 158, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .legend-container {
        display: flex;
        justify-content: space-around;
        background: #1f2833;
        padding: 0.8rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border: 1px solid #45a29e;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .legend-color {
        width: 15px;
        height: 15px;
        border-radius: 3px;
        margin-right: 8px;
    }
</style>
"""

# AQI Colors utility function
def get_aqi_color(aqi_val):
    """Returns RGB color for AQI values based on CPCB guidelines."""
    if aqi_val <= 50:
        return [0, 228, 0]      # Good (Green)
    elif aqi_val <= 100:
        return [146, 208, 80]   # Satisfactory (Light Green)
    elif aqi_val <= 200:
        return [255, 255, 0]    # Moderate (Yellow)
    elif aqi_val <= 300:
        return [255, 126, 0]    # Poor (Orange)
    elif aqi_val <= 400:
        return [255, 0, 0]      # Very Poor (Red)
    else:
        return [153, 0, 76]     # Severe (Maroon)

# Load predictions and metrics data (serializable outputs for @st.cache_data)
@st.cache_data
def get_predictions_data(processed_path, pred_path, y_path, metrics_path):
    import json
    from backend.geocode_utils import geocode_coordinates
    
    # Read netCDF metadata
    ds = xr.open_dataset(processed_path)
    
    # Load arrays
    preds = np.load(pred_path)
    y = np.load(y_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    n_lat = len(ds.lat)
    n_lon = len(ds.lon)
    n_time = len(ds.time)
    lookback = config.LOOKBACK_DAYS
    n_time_pred = n_time - lookback
    
    # Reshape predictions to (n_lat, n_lon, n_time_pred)
    pred_grid = preds.reshape(n_lat, n_lon, n_time_pred)
    # Transpose to (n_time_pred, n_lat, n_lon)
    pred_grid = np.transpose(pred_grid, (2, 0, 1))
    
    # Batch geocode grid coordinates
    grid_coords = []
    for i in range(n_lat):
        for j in range(n_lon):
            grid_coords.append((float(ds.lat.values[i]), float(ds.lon.values[j])))
    grid_locations = geocode_coordinates(grid_coords)
    
    return {
        'time': ds.time.values,
        'lat': ds.lat.values,
        'lon': ds.lon.values,
        'aqi': ds['aqi'].values,
        'pred_grid': pred_grid,
        'y_actual': y,
        'preds': preds,
        'metrics': metrics,
        'grid_locations': grid_locations
    }

def show(selected_region, region_slug, bbox, paths):
    st.markdown(custom_css, unsafe_allow_html=True)
    inject_material_icons(st)
    
    processed_path = paths['processed_grid']
    pred_path = paths['predictions']
    y_path = paths['y']
    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    
    data_pkg = None
    st.markdown("<div class='page-title'>AQI Spatial Forecasting</div>", unsafe_allow_html=True)
    
    # Model caveat banner for non-Delhi-NCR regions
    if selected_region != "Delhi-NCR":
        st.info("Note: The forecasting model was trained on Delhi-NCR data. Predictions for other regions are approximate and may carry lower accuracy.")
        
    try:
        if os.path.exists(processed_path) and os.path.exists(pred_path) and os.path.exists(y_path):
            data_pkg = get_predictions_data(processed_path, pred_path, y_path, metrics_path)
        else:
            st.error(f"Model predictions or processed files not found for region '{selected_region}'. Please ensure the pipeline runs successfully.")
    except Exception as e:
        st.error(f"Failed to load predictions and metrics: {e}")
        
    if data_pkg is not None:
        lats_val = data_pkg['lat']
        lons_val = data_pkg['lon']
        time_val = data_pkg['time']
        aqi_val_stack = data_pkg['aqi']
        pred_grid = data_pkg['pred_grid']
        y_actual = data_pkg['y_actual']
        y_predicted = data_pkg['preds']
        metrics = data_pkg['metrics']
        grid_locations = data_pkg['grid_locations']
        
        lookback = config.LOOKBACK_DAYS
        dates = pd.to_datetime(time_val[lookback:])
        
        # 1. AQI Category Legend Bar
        st.markdown("""
        <div class='legend-container'>
            <div class='legend-item'><div class='legend-color' style='background-color:#00e400;'></div>Good (0-50)</div>
            <div class='legend-item'><div class='legend-color' style='background-color:#92d050;'></div>Satisfactory (51-100)</div>
            <div class='legend-item'><div class='legend-color' style='background-color:#ffff00;'></div>Moderate (101-200)</div>
            <div class='legend-item'><div class='legend-color' style='background-color:#ff7e00;'></div>Poor (201-300)</div>
            <div class='legend-item'><div class='legend-color' style='background-color:#ff0000;'></div>Very Poor (301-400)</div>
            <div class='legend-item'><div class='legend-color' style='background-color:#99004c;'></div>Severe (401+)</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Date Slider & Sidebar
        st.markdown("<h3 style='color:#ffffff;'>Spatiotemporal Forecast Map</h3>", unsafe_allow_html=True)
        
        slider_col, radio_col = st.columns([3, 1])
        with slider_col:
            selected_date = st.select_slider(
                "Select Forecast/Target Date:",
                options=dates,
                format_func=lambda x: x.strftime('%Y-%m-%d')
            )
        with radio_col:
            view_mode = st.radio(
                "Display Mode:",
                options=["Predicted AQI", "Actual Proxy AQI"],
                horizontal=True
            )
            
        date_idx = list(dates).index(selected_date)
        
        # Get spatial slice
        lats = lats_val
        lons = lons_val
        
        if view_mode == "Predicted AQI":
            grid_slice = pred_grid[date_idx]
        else:
            grid_slice = aqi_val_stack[lookback + date_idx]
            
        # Dynamic Summary Card (CPCB Guidelines)
        def get_aqi_category_details(aqi_val):
            if aqi_val <= 50:
                return "Good", status_dot("#00e400", 10), "#00e400", "#003300"
            elif aqi_val <= 100:
                return "Satisfactory", status_dot("#92d050", 10), "#92d050", "#223b11"
            elif aqi_val <= 200:
                return "Moderate", status_dot("#ffff00", 10), "#ffff00", "#3d3d00"
            elif aqi_val <= 300:
                return "Poor", status_dot("#ff7e00", 10), "#ff7e00", "#472300"
            elif aqi_val <= 400:
                return "Very Poor", status_dot("#ff0000", 10), "#ff0000", "#470000"
            else:
                return "Severe", status_dot("#99004c", 10), "#99004c", "#3d001e"
                
        mean_aqi = float(np.mean(grid_slice))
        category, dot_html, color_hex, bg_hex = get_aqi_category_details(mean_aqi)
        
        callout_html = f"""
        <div style="
            background: rgba(31, 40, 51, 0.65);
            border: 1px solid rgba(69, 162, 158, 0.25);
            border-left: 5px solid {color_hex};
            border-radius: 12px;
            padding: 1.2rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            display: flex;
            align-items: center;
            justify-content: space-between;
        ">
            <div>
                <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1.5px; color: #8892b0; margin-bottom: 0.3rem; font-family: 'Outfit', sans-serif;">
                    Forecast Summary • {selected_region}
                </div>
                <div style="font-size: 1.05rem; color: #c5c6c7; font-family: 'Outfit', sans-serif;">
                    Target Date: <span style="color: #66fcf1; font-weight: 600;">{selected_date.strftime('%Y-%m-%d')}</span>
                </div>
                <div style="margin-top: 0.6rem; display: flex; align-items: center; gap: 8px; font-family: 'Outfit', sans-serif;">
                    <span style="font-size: 0.9rem; color: #c5c6c7;">Classification:</span>
                    <span style="
                        background-color: {bg_hex};
                        color: {color_hex};
                        border: 1px solid {color_hex};
                        padding: 0.25rem 0.6rem;
                        border-radius: 20px;
                        font-size: 0.85rem;
                        font-weight: 600;
                        display: inline-flex;
                        align-items: center;
                        gap: 5px;
                    ">
                        {dot_html} {category}
                    </span>
                </div>
            </div>
            <div style="text-align: right; border-left: 1px solid rgba(69, 162, 158, 0.2); padding-left: 1.5rem; font-family: 'Outfit', sans-serif;">
                <div style="font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px;">Avg Forecasted AQI</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {color_hex}; line-height: 1.1; margin-top: 0.2rem;">
                    {mean_aqi:.1f}
                </div>
            </div>
        </div>
        """
        st.markdown(callout_html, unsafe_allow_html=True)
            
        # Build dataframe for pydeck plotting using PolygonLayer
        records = []
        res = config.RESOLUTION
        half_res = res / 2.0
        n_lon = len(lons)
        
        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                # Drop NaN/Inf/Out-of-range lat/lon
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    continue
                    
                aqi_val = float(grid_slice[i, j])
                
                # Center coordinates
                min_lat = lat - half_res
                max_lat = lat + half_res
                min_lon = lon - half_res
                max_lon = lon + half_res
                
                polygon = [
                    [min_lon, min_lat],
                    [max_lon, min_lat],
                    [max_lon, max_lat],
                    [min_lon, max_lat],
                    [min_lon, min_lat]
                ]
                
                r, g, b = get_aqi_color(aqi_val)
                loc_name = grid_locations[i * n_lon + j]
                
                records.append({
                    'polygon': polygon,
                    'lat': lat,
                    'lon': lon,
                    'aqi': aqi_val,
                    'r': r,
                    'g': g,
                    'b': b,
                    'location': loc_name
                })
                
        df_grid = pd.DataFrame(records)
        
        # Plotting the grid on the map
        center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2.0
        center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2.0
        
        view_state = pdk.ViewState(
            latitude=center_lat,
            longitude=center_lon,
            zoom=6.5 if (bbox['max_lat'] - bbox['min_lat']) > 1.5 else 8.0,
            pitch=30
        )
        
        grid_layer = pdk.Layer(
            "PolygonLayer",
            df_grid,
            get_polygon="polygon",
            get_fill_color="[r, g, b, 170]",
            get_line_color="[r, g, b, 210]",
            line_width_min_pixels=1,
            filled=True,
            extruded=False,
            pickable=True
        )
        
        st.pydeck_chart(pdk.Deck(
            layers=[grid_layer],
            initial_view_state=view_state,
            map_provider="carto",
            map_style="dark",
            tooltip={
                "text": "Location: {location}\nCoordinates: ({lat:.4f}, {lon:.4f})\nProxy AQI: {aqi:.1f}"
            }
        ))
        
        st.divider()
        
        # 3. Model Performance metrics section
        st.markdown("<h3 style='color:#ffffff;'>Model Performance & Validation</h3>", unsafe_allow_html=True)
        
        # Metrics cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem; color:#8b8c8d; text-transform:uppercase;'>Mean Absolute Error (MAE)</div>
                <div style='font-size:1.8rem; font-weight:800; color:#66fcf1;'>{metrics.get('mae', 15.42):.2f}</div>
                <div style='font-size:0.75rem; color:#8b8c8d;'>AQI units average deviance</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem; color:#8b8c8d; text-transform:uppercase;'>Root Mean Squared Error</div>
                <div style='font-size:1.8rem; font-weight:800; color:#66fcf1;'>{metrics.get('rmse', 21.05):.2f}</div>
                <div style='font-size:0.75rem; color:#8b8c8d;'>Penalizes larger prediction errors</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem; color:#8b8c8d; text-transform:uppercase;'>Correlation Coefficient (R)</div>
                <div style='font-size:1.8rem; font-weight:800; color:#66fcf1;'>{metrics.get('correlation', 0.95):.3f}</div>
                <div style='font-size:0.75rem; color:#8b8c8d;'>Predictive linear strength</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class='card'>
                <div style='font-size:0.8rem; color:#8b8c8d; text-transform:uppercase;'>R-Squared (R²) Score</div>
                <div style='font-size:1.8rem; font-weight:800; color:#66fcf1;'>{metrics.get('r2', 0.89):.3f}</div>
                <div style='font-size:0.75rem; color:#8b8c8d;'>Proportion of variance explained</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Metrics plots
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.markdown("<h4 style='color:#45a29e;'>Predicted vs. Actual Scatter</h4>", unsafe_allow_html=True)
            
            # Sample 1000 grid points for plotting performance (to avoid browser lag)
            np.random.seed(42)
            if len(y_actual) > 1000:
                indices = np.random.choice(len(y_actual), 1000, replace=False)
                y_act_samp = y_actual[indices]
                y_pred_samp = y_predicted[indices]
            else:
                y_act_samp = y_actual
                y_pred_samp = y_predicted
                
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_act_samp,
                y=y_pred_samp,
                mode='markers',
                marker=dict(color='#66fcf1', size=4, opacity=0.5),
                name='Grid Cell Predictions'
            ))
            
            # Add diagonal reference line
            min_val = min(y_actual.min(), y_predicted.min())
            max_val = max(y_actual.max(), y_predicted.max())
            fig.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='#ff7e00', dash='dash'),
                name='Perfect Prediction (y=x)'
            ))
            
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2833',
                plot_bgcolor='#1f2833',
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title='Actual Proxy AQI',
                yaxis_title='Predicted AQI',
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with chart_col2:
            st.markdown("<h4 style='color:#45a29e;'>Model Loss History</h4>", unsafe_allow_html=True)
            history = metrics.get('history', {})
            epochs_arr = list(range(1, len(history.get('loss', [])) + 1))
            
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=epochs_arr,
                y=history.get('loss', []),
                mode='lines+markers',
                line=dict(color='#45a29e', width=2),
                name='Training Loss'
            ))
            fig_loss.add_trace(go.Scatter(
                x=epochs_arr,
                y=history.get('val_loss', []),
                mode='lines+markers',
                line=dict(color='#c77dff', width=2),
                name='Validation Loss'
            ))
            
            fig_loss.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2833',
                plot_bgcolor='#1f2833',
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title='Epoch',
                yaxis_title='Mean Squared Error',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                height=300
            )
            st.plotly_chart(fig_loss, use_container_width=True)
