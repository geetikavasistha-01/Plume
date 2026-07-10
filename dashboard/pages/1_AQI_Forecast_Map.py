import os
import sys
import json
import numpy as np
import pandas as pd
import xarray as xr
import streamlit as st
import pydeck as pdk
import plotly.graph_objects as go
import tensorflow as tf

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

# Set page config
st.set_page_config(
    page_title="AQI Forecast Map",
    page_icon="🌍",
    layout="wide"
)

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
st.markdown(custom_css, unsafe_allow_html=True)

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

# Load Keras model resource
@st.cache_resource
def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)

# Load predictions and metrics data (serializable outputs for @st.cache_data)
@st.cache_data
def get_predictions_data(processed_path, X_path, y_path, metrics_path, model_path):
    import json
    # Read netCDF metadata
    ds = xr.open_dataset(processed_path)
    # Load model locally via cache resource
    model = load_keras_model(model_path)
    
    # Load arrays
    X = np.load(X_path)
    y = np.load(y_path)
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
        
    # Run prediction
    preds = model.predict(X, batch_size=4096, verbose=0).flatten()
    
    n_lat = len(ds.lat)
    n_lon = len(ds.lon)
    n_time = len(ds.time)
    lookback = config.LOOKBACK_DAYS
    n_time_pred = n_time - lookback
    
    # Reshape predictions to (n_lat, n_lon, n_time_pred)
    pred_grid = preds.reshape(n_lat, n_lon, n_time_pred)
    # Transpose to (n_time_pred, n_lat, n_lon)
    pred_grid = np.transpose(pred_grid, (2, 0, 1))
    
    return {
        'time': ds.time.values,
        'lat': ds.lat.values,
        'lon': ds.lon.values,
        'aqi': ds['aqi'].values,
        'pred_grid': pred_grid,
        'y_actual': y,
        'preds': preds,
        'metrics': metrics
    }

# Import helper
from dashboard import helper

# Render sidebar region selector and get selected region
selected_region, region_slug, bbox, paths = helper.render_sidebar_and_get_region()

# Main program
processed_path = paths['processed_grid']
model_path = os.path.join(config.MODELS_DIR, "cnn_lstm_aqi.keras")
X_path = paths['X']
y_path = paths['y']
metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")

data_pkg = None
st.markdown("<div class='page-title'>AQI Spatial Forecasting</div>", unsafe_allow_html=True)

# Model caveat banner for non-Delhi-NCR regions
if selected_region != "Delhi-NCR":
    st.info("Note: The forecasting model was trained on Delhi-NCR data. Predictions for other regions are approximate and may carry lower accuracy.")

try:
    if os.path.exists(processed_path) and os.path.exists(model_path) and os.path.exists(X_path):
        data_pkg = get_predictions_data(processed_path, X_path, y_path, metrics_path, model_path)
    else:
        st.error(f"Model resources or processed files not found for region '{selected_region}'. Please ensure the pipeline runs successfully.")
except Exception as e:
    st.error(f"Failed to load model and run predictions: {e}")

if data_pkg is not None:
    lats_val = data_pkg['lat']
    lons_val = data_pkg['lon']
    time_val = data_pkg['time']
    aqi_val_stack = data_pkg['aqi']
    pred_grid = data_pkg['pred_grid']
    y_actual = data_pkg['y_actual']
    y_predicted = data_pkg['preds']
    metrics = data_pkg['metrics']
    
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
        
    # Build dataframe for pydeck plotting using PolygonLayer
    records = []
    res = config.RESOLUTION
    half_res = res / 2.0
    
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            # Drop NaN/Inf/Out-of-range lat/lon
            if not np.isfinite(lat) or not np.isfinite(lon):
                continue
            if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                continue
                
            aqi_val = float(grid_slice[i, j])
            if not np.isfinite(aqi_val):
                continue
                
            color = get_aqi_color(aqi_val)
            
            # Corners in [lon, lat] order
            c1 = [float(lon - half_res), float(lat - half_res)]
            c2 = [float(lon + half_res), float(lat - half_res)]
            c3 = [float(lon + half_res), float(lat + half_res)]
            c4 = [float(lon - half_res), float(lat + half_res)]
            
            # Verify exactly 4 valid corner points
            if not (all(np.isfinite(c1)) and all(np.isfinite(c2)) and all(np.isfinite(c3)) and all(np.isfinite(c4))):
                continue
                
            polygon = [c1, c2, c3, c4, c1]  # Closed ring (first and last match)
            
            records.append({
                'polygon': polygon,
                'aqi': aqi_val,
                'r': color[0],
                'g': color[1],
                'b': color[2]
            })
            
    map_df = pd.DataFrame(records)
    
    # Render map
    center_lat = float(np.mean(lats))
    center_lon = float(np.mean(lons))
    
    view_state = pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=6.5 if (bbox['max_lat'] - bbox['min_lat']) > 1.5 else 7.8,
        pitch=40
    )
    
    grid_layer = pdk.Layer(
        "PolygonLayer",
        map_df,
        get_polygon="polygon",
        get_fill_color="[r, g, b, 180]",
        get_line_color="[r, g, b, 80]",
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
        tooltip={"text": "AQI: {aqi}"}
    ))
    
    # 3. Model Performance Section (Leaned down & packed)
    st.markdown("---")
    st.markdown("<h2 style='color:#ffffff;'>CNN-LSTM Model Performance Analysis</h2>", unsafe_allow_html=True)
    
    metric_col, chart_col1, chart_col2 = st.columns([1, 2, 2])
    
    with metric_col:
        st.markdown("<h4 style='color:#45a29e;'>Validation Metrics</h4>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>Mean Absolute Error</div>
            <div class='card-value'>{metrics.get('val_mae', 0.0):.3f}</div>
            <div class='card-desc'>Average prediction offset in AQI points</div>
        </div>
        
        <div class='card'>
            <div class='card-title'>RMSE</div>
            <div class='card-value'>{metrics.get('val_rmse', 0.0):.3f}</div>
            <div class='card-desc'>Root Mean Squared Error (penalizes large errors)</div>
        </div>
        
        <div class='card'>
            <div class='card-title'>R² Score / Corr</div>
            <div class='card-value'>{metrics.get('correlation', 0.0):.4f}</div>
            <div class='card-desc'>Prediction-Target Pearson correlation</div>
        </div>
        """, unsafe_allow_html=True)
        
    with chart_col1:
        st.markdown("<h4 style='color:#45a29e;'>Predicted vs. Actual AQI</h4>", unsafe_allow_html=True)
        # Sample predictions to keep plot fast
        sample_indices = np.random.choice(len(y_actual), min(1000, len(y_actual)), replace=False)
        y_act_samp = y_actual[sample_indices]
        y_pred_samp = y_predicted[sample_indices]
        
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
