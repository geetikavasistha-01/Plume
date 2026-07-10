import os
import sys
import subprocess
import streamlit as st

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)
from dashboard.icon_utils import icon

def run_pipeline(region_name, bbox=None):
    """Executes the ingestion, preprocessing, and hotspot steps sequentially."""
    python_path = sys.executable
    backend_dir = os.path.join(config.BASE_DIR, "backend")
    
    # 1. Ingestion (GEE)
    ingest_cmd = [python_path, os.path.join(backend_dir, "data_ingest.py"), "--region", region_name]
    # 2. CPCB ground stations Ingestion
    cpcb_cmd = [python_path, os.path.join(backend_dir, "cpcb_ingest.py"), "--region", region_name]
    # 3. Preprocess
    preprocess_cmd = [python_path, os.path.join(backend_dir, "preprocess.py"), "--region", region_name]
    # 4. Model Inference Predictions Caching
    inference_cmd = [python_path, os.path.join(backend_dir, "inference.py"), "--region", region_name]
    # 5. Hotspot Outlier Detection
    hotspot_cmd = [python_path, os.path.join(backend_dir, "hotspot.py"), "--region", region_name]
    # 6. NASA FIRMS Active Fires Ingestion
    fire_ingest_cmd = [python_path, os.path.join(backend_dir, "fire_ingest.py"), "--region", region_name]
    # 7. Fire and HCHO Correlation Analysis
    fire_analysis_cmd = [python_path, os.path.join(backend_dir, "fire_analysis.py"), "--region", region_name]
    # 8. Wind Transport Analysis
    transport_cmd = [python_path, os.path.join(backend_dir, "transport_analysis.py"), "--region", region_name]
    
    extra_args = []
    if bbox:
        extra_args = [
            "--lat-min", str(bbox['min_lat']),
            "--lat-max", str(bbox['max_lat']),
            "--lon-min", str(bbox['min_lon']),
            "--lon-max", str(bbox['max_lon'])
        ]
        ingest_cmd += extra_args
        cpcb_cmd += extra_args
        preprocess_cmd += extra_args
        inference_cmd += extra_args
        hotspot_cmd += extra_args
        fire_ingest_cmd += extra_args
        fire_analysis_cmd += extra_args
        transport_cmd += extra_args
        
    try:
        # One-time warning message
        info_placeholder = st.empty()
        info_placeholder.info(f"'{region_name}' hasn't been loaded before — running full data pipeline. This usually takes 2-3 minutes.")
        
        # Check for st.status compatibility
        if hasattr(st, "status"):
            with st.status(f"Setting up '{region_name}'...", expanded=True) as status:
                status.write(f"{icon('satellite_alt', 18, '#66fcf1')} Step 1/8: Fetching satellite + meteorology data from Earth Engine...")
                result = subprocess.run(ingest_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('domain', 18, '#66fcf1')} Step 2/8: Ingesting CPCB ground monitoring station data...")
                result = subprocess.run(cpcb_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('calculate', 18, '#66fcf1')} Step 3/8: Regridding variables and calculating targets...")
                result = subprocess.run(preprocess_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('psychology', 18, '#66fcf1')} Step 4/8: Running CNN-LSTM model predictions...")
                result = subprocess.run(inference_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('travel_explore', 18, '#66fcf1')} Step 5/8: Running Isolation Forest chemical hotspot detector...")
                result = subprocess.run(hotspot_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('local_fire_department', 18, '#66fcf1')} Step 6/8: Fetching MODIS/VIIRS active fire points...")
                result = subprocess.run(fire_ingest_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('bar_chart', 18, '#66fcf1')} Step 7/8: Correlating active fires with HCHO plumes...")
                result = subprocess.run(fire_analysis_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.write(f"{icon('air', 18, '#66fcf1')} Step 8/8: Simulating wind rose advection trajectories...")
                result = subprocess.run(transport_cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                
                status.update(label=f"'{region_name}' ready!", state="complete", expanded=False)
        else:
            # Fallback
            status_text = st.empty()
            status_text.info(f"{icon('satellite_alt', 18)} Step 1/8: Ingesting satellite data from Earth Engine...")
            subprocess.run(ingest_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('domain', 18)} Step 2/8: Ingesting CPCB ground monitoring stations...")
            subprocess.run(cpcb_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('calculate', 18)} Step 3/8: Preprocessing grid alignment...")
            subprocess.run(preprocess_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('psychology', 18)} Step 4/8: Running CNN-LSTM model predictions...")
            subprocess.run(inference_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('travel_explore', 18)} Step 5/8: Running Isolation Forest hotspot detector...")
            subprocess.run(hotspot_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('local_fire_department', 18)} Step 6/8: Ingesting MODIS/VIIRS active fires...")
            subprocess.run(fire_ingest_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('bar_chart', 18)} Step 7/8: Running active fire correlation analysis...")
            subprocess.run(fire_analysis_cmd, capture_output=True, text=True, check=True)
            status_text.info(f"{icon('air', 18)} Step 8/8: Running wind advection calculations...")
            subprocess.run(transport_cmd, capture_output=True, text=True, check=True)
            status_text.empty()
            
        info_placeholder.empty()
        st.success(f"Pipeline executed successfully for '{region_name}'!")
        
        # Clear Streamlit cache and trigger rerun
        st.cache_data.clear()
        st.rerun()
    except subprocess.CalledProcessError as e:
        st.error(f"Pipeline execution failed: {e.stderr if e.stderr else e.stdout}")
        st.stop()

def render_sidebar_and_get_region():
    """Renders region selector and metadata details inside collapsible sidebar sections."""
    st.sidebar.markdown("<h3 style='margin-top: 0; color:#66fcf1;'>Configuration</h3>", unsafe_allow_html=True)
    
    region_options = list(config.PRESET_REGIONS.keys()) + ["Search Location...", "Custom Coordinates..."]
    
    # Initialize session states
    if 'selected_region_mode' not in st.session_state:
        st.session_state.selected_region_mode = "Delhi-NCR"
    if 'resolved_region_name' not in st.session_state:
        st.session_state.resolved_region_name = ""
    if 'resolved_bbox' not in st.session_state:
        st.session_state.resolved_bbox = None
        
    default_idx = region_options.index(st.session_state.selected_region_mode) if st.session_state.selected_region_mode in region_options else 0
    
    selected_mode = st.sidebar.selectbox(
        "Select Region Mode",
        region_options,
        index=default_idx
    )
    
    st.session_state.selected_region_mode = selected_mode
    
    bbox = None
    selected_region = None
    region_slug = None
    
    if selected_mode == "Custom Coordinates...":
        st.sidebar.markdown("#### Bounding Box Input")
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
            
        bbox = {
            'min_lat': lat_min,
            'max_lat': lat_max,
            'min_lon': lon_min,
            'max_lon': lon_max
        }
        
        selected_region = f"Custom_{lat_min}_{lat_max}_{lon_min}_{lon_max}"
        region_slug = config.get_region_slug(selected_region)
        
        paths = config.get_paths(region_slug)
        if not os.path.exists(paths['processed_grid']) or not os.path.exists(paths['hotspots']):
            if st.sidebar.button("Process Custom Coordinates"):
                run_pipeline(selected_region, bbox)
            else:
                st.sidebar.info("Cache missing. Click 'Process Custom Coordinates' to run pipeline.")
                st.stop()
            
    elif selected_mode == "Search Location...":
        st.sidebar.markdown("#### Search Indian Location")
        search_query = st.sidebar.text_input(
            "Enter City, District, or State name:",
            value=st.session_state.resolved_region_name,
            placeholder="e.g. Hyderabad, Nagpur, Rajasthan"
        )
        st.sidebar.caption("Search any Indian city, district, or state to resolve its bounding box dynamically.")
        
        if search_query:
            # Resolve name if search query changed or resolved_bbox is not set
            if (st.session_state.resolved_region_name != search_query) or (st.session_state.resolved_bbox is None):
                from backend.region_resolver import resolve_region_bounds
                try:
                    resolved = resolve_region_bounds(search_query)
                    st.session_state.resolved_region_name = search_query
                    st.session_state.resolved_bbox = resolved
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                    st.stop()
                    
            bbox = st.session_state.resolved_bbox
            selected_region = search_query
            region_slug = config.get_region_slug(selected_region)
            
            # Validation on resolved bounds
            lat_size = bbox['max_lat'] - bbox['min_lat']
            lon_size = bbox['max_lon'] - bbox['min_lon']
            
            if lat_size > 5.0 or lon_size > 5.0:
                st.sidebar.warning("Warning: Large box size (>5°). Pull might be slow.")
            if lat_size > 10.0 or lon_size > 10.0:
                st.sidebar.error("Error: Resolved box size > 10° is blocked to prevent GEE quota failures.")
                st.stop()
                
            paths = config.get_paths(region_slug)
            
            # Check cache
            if not os.path.exists(paths['processed_grid']) or not os.path.exists(paths['hotspots']):
                st.sidebar.info(f"Cache missing for '{selected_region}'. Triggering pipeline...")
                run_pipeline(selected_region, bbox)
        else:
            st.sidebar.info("Enter a location name to search.")
            st.stop()
            
    else:
        # Preset region
        selected_region = selected_mode
        region_slug = config.get_region_slug(selected_region)
        region_coords = config.PRESET_REGIONS[selected_region]
        bbox = {
            'min_lat': region_coords['lat_min'],
            'max_lat': region_coords['lat_max'],
            'min_lon': region_coords['lon_min'],
            'max_lon': region_coords['lon_max']
        }
        paths = config.get_paths(region_slug)
        if not os.path.exists(paths['processed_grid']) or not os.path.exists(paths['hotspots']):
            st.sidebar.info(f"Cache missing for '{selected_region}'. Triggering pipeline...")
            run_pipeline(selected_region, bbox)
            
    paths = config.get_paths(region_slug)
    
    st.sidebar.markdown("---")
    
    # Collapsible Region Details
    with st.sidebar.expander("📍 Region Details", expanded=False):
        st.markdown(f"**Study Region:**\n`{selected_region}`")
        st.markdown("**Bounding Box:**")
        st.markdown(f"Lat: `{bbox['min_lat']:.2f} - {bbox['max_lat']:.2f}`")
        st.markdown(f"Lon: `{bbox['min_lon']:.2f} - {bbox['max_lon']:.2f}`")
        st.markdown("**Temporal Resolution:**\nDaily")
        st.markdown(f"**Spatial Grid Size:**\n{config.RESOLUTION}° (~5km)")
        
    # Collapsible Data Sources
    with st.sidebar.expander("📚 Data Sources", expanded=False):
        st.markdown("""
        - **Sentinel-5P** (HCHO, NO2, UV AI)
        - **ERA5 Land** (Temp, Winds, Precip)
        - **CPCB CAAQM** Ground Stations
        - **NASA FIRMS** Active Fires
        """)
        
    return selected_region, region_slug, bbox, paths
