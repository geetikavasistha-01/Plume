import os
import sys
import subprocess
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

def run_pipeline(region_name, bbox=None):
    """Executes the ingestion, preprocessing, and hotspot steps sequentially."""
    python_path = sys.executable
    backend_dir = os.path.join(config.BASE_DIR, "backend")
    
    # 1. Ingestion
    ingest_cmd = [python_path, os.path.join(backend_dir, "data_ingest.py"), "--region", region_name]
    if bbox:
        ingest_cmd += [
            "--lat-min", str(bbox['min_lat']),
            "--lat-max", str(bbox['max_lat']),
            "--lon-min", str(bbox['min_lon']),
            "--lon-max", str(bbox['max_lon'])
        ]
    
    # 2. Preprocess
    preprocess_cmd = [python_path, os.path.join(backend_dir, "preprocess.py"), "--region", region_name]
    if bbox:
        preprocess_cmd += [
            "--lat-min", str(bbox['min_lat']),
            "--lat-max", str(bbox['max_lat']),
            "--lon-min", str(bbox['min_lon']),
            "--lon-max", str(bbox['max_lon'])
        ]
        
    # 3. Hotspot
    hotspot_cmd = [python_path, os.path.join(backend_dir, "hotspot.py"), "--region", region_name]
    if bbox:
        hotspot_cmd += [
            "--lat-min", str(bbox['min_lat']),
            "--lat-max", str(bbox['max_lat']),
            "--lon-min", str(bbox['min_lon']),
            "--lon-max", str(bbox['max_lon'])
        ]
        
    try:
        # Step 1: Ingest
        with st.spinner(f"Step 1/3: Ingesting Sentinel-5P + ERA5 GEE data for '{region_name}'... (1-2 mins)"):
            result = subprocess.run(ingest_cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
        # Step 2: Preprocess
        with st.spinner("Step 2/3: Regridding variables & calculating Proxy AQI..."):
            result = subprocess.run(preprocess_cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
        # Step 3: Hotspot
        with st.spinner("Step 3/3: Running Isolation Forest hotspot detector..."):
            result = subprocess.run(hotspot_cmd, capture_output=True, text=True, check=True)
            print(result.stdout)
            
        st.success(f"Pipeline executed successfully for '{region_name}'!")
        
        # Clear Streamlit cache and trigger rerun
        st.cache_data.clear()
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Pipeline execution failed: {e.stderr if e.stderr else e.stdout}")
        st.stop()

def render_sidebar_and_get_region():
    """Renders a common sidebar region selector and coordinates display."""
    st.sidebar.markdown("<h1 style='text-align: center; margin-bottom: 0;'>📡</h1>", unsafe_allow_html=True)
    st.sidebar.markdown("<h2 style='text-align: center; color:#66fcf1; font-weight:800; margin-top:0;'>ISRO AQI & HCHO</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("---")
    
    region_options = list(config.PRESET_REGIONS.keys()) + ["Custom..."]
    
    # Persist selected region in session state
    if 'selected_region' not in st.session_state:
        st.session_state.selected_region = "Delhi-NCR"
        
    default_idx = region_options.index(st.session_state.selected_region) if st.session_state.selected_region in region_options else 0
    
    selected_region = st.sidebar.selectbox(
        "Select Region",
        region_options,
        index=default_idx
    )
    
    bbox = None
    is_custom = selected_region == "Custom..."
    
    if is_custom:
        st.sidebar.markdown("### Custom Bounding Box")
        lat_min = st.sidebar.number_input("Lat Min", value=27.0, min_value=-90.0, max_value=90.0, step=0.1)
        lat_max = st.sidebar.number_input("Lat Max", value=29.5, min_value=-90.0, max_value=90.0, step=0.1)
        lon_min = st.sidebar.number_input("Lon Min", value=76.5, min_value=-180.0, max_value=180.0, step=0.1)
        lon_max = st.sidebar.number_input("Lon Max", value=78.5, min_value=-180.0, max_value=180.0, step=0.1)
        
        # Validations
        if lat_min >= lat_max:
            st.sidebar.error("Error: Lat Min must be less than Lat Max")
            st.stop()
        if lon_min >= lon_max:
            st.sidebar.error("Error: Lon Min must be less than Lon Max")
            st.stop()
            
        lat_size = lat_max - lat_min
        lon_size = lon_max - lon_min
        
        if lat_size > 5.0 or lon_size > 5.0:
            st.sidebar.warning("Warning: Large box size (>5°). Pull might be slow.")
        if lat_size > 10.0 or lon_size > 10.0:
            st.sidebar.error("Error: Box size > 10° is blocked to prevent GEE quota failures.")
            st.stop()
            
        bbox = {
            'min_lat': lat_min,
            'max_lat': lat_max,
            'min_lon': lon_min,
            'max_lon': lon_max
        }
        
        # Custom region slug
        custom_name = f"Custom_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
        region_slug = config.get_region_slug(custom_name)
        
        if st.sidebar.button("Process Custom Region"):
            st.session_state.selected_region = "Custom..."
            run_pipeline(custom_name, bbox)
    else:
        st.session_state.selected_region = selected_region
        region_slug = config.get_region_slug(selected_region)
        region_coords = config.PRESET_REGIONS[selected_region]
        bbox = {
            'min_lat': region_coords['lat_min'],
            'max_lat': region_coords['lat_max'],
            'min_lon': region_coords['lon_min'],
            'max_lon': region_coords['lon_max']
        }
        
    paths = config.get_paths(region_slug)
    
    # For presets, run pipeline automatically if cached files are missing
    if not is_custom:
        if not os.path.exists(paths['processed_grid']) or not os.path.exists(paths['hotspots']):
            st.sidebar.info(f"Cache missing for '{selected_region}'. Triggering pipeline...")
            run_pipeline(selected_region)
            
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Study Region:** {selected_region}")
    st.sidebar.markdown(f"**Bounding Box:**")
    st.sidebar.markdown(f"Lat: `{bbox['min_lat']:.2f} - {bbox['max_lat']:.2f}`")
    st.sidebar.markdown(f"Lon: `{bbox['min_lon']:.2f} - {bbox['max_lon']:.2f}`")
    st.sidebar.markdown(f"**Temporal Resolution:** Daily")
    st.sidebar.markdown(f"**Spatial Grid Size:** {config.RESOLUTION}° (~5km)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data Sources:**")
    st.sidebar.markdown("- Sentinel-5P TROPOMI")
    st.sidebar.markdown("- ECMWF ERA5 Land hourly")
    st.sidebar.markdown("---")
    st.sidebar.markdown("<div style='font-size:0.85rem; color:#8b8c8d;'>Developed by Kunoichi (GeekyKunoichi)<br>Platform: Antigravity AI IDE</div>", unsafe_allow_html=True)
    
    return selected_region, region_slug, bbox, paths
