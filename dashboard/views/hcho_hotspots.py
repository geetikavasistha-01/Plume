import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
from dashboard.icon_utils import inject_material_icons, icon, status_dot

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

# Custom CSS for dark aesthetic
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
        background: linear-gradient(45deg, #c77dff, #ff7e00);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 1.5rem;
    }
    
    .card {
        background: rgba(31, 40, 51, 0.65);
        border: 1px solid rgba(199, 125, 255, 0.3);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .card-title {
        font-size: 0.9rem;
        color: #c77dff;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .card-value {
        font-size: 2rem;
        color: #ffffff;
        font-weight: 800;
    }
</style>
"""

# Helper function to load hotspots
@st.cache_data
def load_hotspots(hotspots_path):
    if not os.path.exists(hotspots_path):
        return None
    with open(hotspots_path, 'r') as f:
        return json.load(f)

# Helper function to load fire correlation
@st.cache_data
def load_fire_correlation(fire_correlation_path):
    if not os.path.exists(fire_correlation_path):
        return None
    with open(fire_correlation_path, 'r') as f:
        return json.load(f)

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
    
    hotspots_data = load_hotspots(paths['hotspots'])
    
    st.markdown("<div class='page-title'>Formaldehyde (HCHO) Hotspot Detection</div>", unsafe_allow_html=True)
    
    # Model caveat banner for non-Delhi-NCR regions
    if selected_region != "Delhi-NCR":
        st.info("Note: The forecasting model was trained on Delhi-NCR data. Predictions for other regions are approximate and may carry lower accuracy.")
        
    if hotspots_data is None:
        st.error(f"Hotspots data not found for region '{selected_region}'. Please ensure the pipeline runs successfully.")
    else:
        hotspots_list = hotspots_data.get('hotspots', [])
        data_date = hotspots_data.get('data_date', 'N/A')
        
        # 0. Plain language summary
        if len(hotspots_list) > 0:
            top_place = hotspots_list[0].get('location', f"Lat {hotspots_list[0]['lat']:.3f}, Lon {hotspots_list[0]['lon']:.3f}")
            summary_text = f"On <strong>{data_date}</strong>, we detected <strong>{len(hotspots_list)} out of {hotspots_data.get('total_grid_cells', 0)}</strong> areas in <strong>{selected_region}</strong> with unusually high formaldehyde (HCHO) levels. The most affected area is near <strong>{top_place}</strong>."
        else:
            summary_text = f"On <strong>{data_date}</strong>, we found <strong>no significant</strong> formaldehyde (HCHO) hotspot anomalies across the <strong>{hotspots_data.get('total_grid_cells', 0)}</strong> grid cells of <strong>{selected_region}</strong>."
            
        st.markdown(f"<div style='font-size: 1.1rem; line-height: 1.6; margin-bottom: 1.5rem; background: rgba(199, 125, 255, 0.1); border-left: 4px solid #c77dff; padding: 1rem; border-radius: 6px;'>{summary_text}</div>", unsafe_allow_html=True)
        
        # 1. Hotspots statistics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Detected Hotspots</div>
                <div class='card-value'>{len(hotspots_list)} / {hotspots_data.get('total_grid_cells', 0)}</div>
                <div class='text' style='font-size:0.85rem; color:#8b8c8d;'>Grid cells with anomalous HCHO columns</div>
            </div>
            """, unsafe_allow_html=True)
            
        hcho_median = hotspots_data.get('hcho_spatial_median', 0.0)
        hcho_mean = hotspots_data.get('hcho_spatial_mean', 0.0)
        hcho_95th = hotspots_data.get('hcho_spatial_95th', 0.0)
        
        with col2:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>HCHO Spatial Summary</div>
                <div class='card-value'>{hcho_mean:.2e}</div>
                <div class='text' style='font-size:0.85rem; color:#8b8c8d; margin-bottom: 0.3rem;'>Mean concentration (mol/m²)</div>
                <div style='font-size:0.8rem; color:#c5c6c7; display:flex; justify-content:space-between;'>
                    <span>Median: <code>{hcho_median:.2e}</code></span>
                    <span>95%tile: <code>{hcho_95th:.2e}</code></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class='card'>
                <div class='card-title'>Observations Date</div>
                <div class='card-value'>{data_date}</div>
                <div class='text' style='font-size:0.85rem; color:#8b8c8d;'>Most recent valid satellite overpass</div>
            </div>
            """, unsafe_allow_html=True)
            
        if len(hotspots_list) == 0:
            st.info("No hotspots detected for the selected threshold.")
        else:
            df_hotspots = pd.DataFrame(hotspots_list)
            
            # Build dataframe for pydeck plotting using PolygonLayer
            records = []
            res = config.get_resolution(bbox)
            half_res = res / 2.0
            
            for idx, row in df_hotspots.iterrows():
                lat = row['lat']
                lon = row['lon']
                
                # Drop NaN/Inf/Out-of-range lat/lon
                if not np.isfinite(lat) or not np.isfinite(lon):
                    continue
                if lat < -90 or lat > 90 or lon < -180 or lon > 180:
                    continue
                    
                hcho = row['hcho']
                no2 = row['no2']
                aod = row['aod']
                aqi = row['aqi']
                score = row['hotspot_score']
                
                # Color mapping: Low score -> orange-red, High score -> glowing magenta
                factor = score / 100.0
                r = 255
                g = int(126 * (1 - factor))
                b = int(180 * factor)
                
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
                
                loc_name = row.get('location', f"Lat {lat:.3f}, Lon {lon:.3f}")
                
                records.append({
                    'polygon': polygon,
                    'lat': lat,
                    'lon': lon,
                    'hcho': hcho,
                    'aqi': aqi,
                    'location': loc_name,
                    'hotspot_score': score,
                    'r': r,
                    'g': g,
                    'b': b
                })
                
            df_map_hotspots = pd.DataFrame(records)
            
            # 2. Main Page Layout: Create Tabs
            tab1, tab2 = st.tabs([":material/travel_explore: Spatial Hotspots", ":material/local_fire_department: Biomass Burning Analysis"])
            
            with tab1:
                left_col, right_col = st.columns([3, 2])
                
                with left_col:
                    st.markdown("<h3 style='color:#ffffff;'>Hotspot Spatial Distribution</h3>", unsafe_allow_html=True)
                    
                    if len(df_map_hotspots) > 0:
                        center_lat = float(df_map_hotspots['lat'].mean())
                        center_lon = float(df_map_hotspots['lon'].mean())
                    else:
                        center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2.0
                        center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2.0
                        
                    view_state = pdk.ViewState(
                        latitude=center_lat,
                        longitude=center_lon,
                        zoom=6.5 if (bbox['max_lat'] - bbox['min_lat']) > 1.5 else 8.0,
                        pitch=30
                    )
                    
                    hotspots_layer = pdk.Layer(
                        "PolygonLayer",
                        df_map_hotspots,
                        get_polygon="polygon",
                        get_fill_color="[r, g, b, 200]",
                        get_line_color="[r, g, b, 240]",
                        line_width_min_pixels=1.5,
                        filled=True,
                        extruded=False,
                        pickable=True
                    )
                    
                    st.pydeck_chart(pdk.Deck(
                        layers=[hotspots_layer],
                        initial_view_state=view_state,
                        map_provider="carto",
                        map_style="dark",
                        tooltip={
                            "text": "Location: {location}\nHotspot Score: {hotspot_score:.1f}\nHCHO: {hcho:.2e}\nAQI: {aqi:.1f}"
                        }
                    ))
                    
                    # Map Legend
                    st.markdown(f"""
                    <div style='background: rgba(31, 40, 51, 0.65); padding: 0.8rem; border-radius: 8px; margin-top: 1rem; border: 1px solid rgba(199, 125, 255, 0.3);'>
                        <div style='font-weight: 600; font-size: 0.85rem; color: #c77dff; margin-bottom: 0.5rem;'>{icon('local_fire_department', 16, '#c77dff')} HOTSPOT ANOMALY SEVERITY GRADIENT</div>
                        <div style='display: flex; align-items: center; justify-content: space-between;'>
                            <span style='font-size: 0.8rem; color: #8b8c8d;'>Moderate (Score 40-70)</span>
                            <div style='height: 12px; flex-grow: 1; margin: 0 10px; border-radius: 3px; background: linear-gradient(to right, rgb(255, 126, 0), rgb(255, 0, 180));'></div>
                            <span style='font-size: 0.8rem; color: #8b8c8d;'>Severe (Score 90-100)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with right_col:
                    st.markdown("<h3 style='color:#ffffff;'>Top Hotspot Locations Ranked</h3>", unsafe_allow_html=True)
                    
                    # Show ranked hotspots table (top 15)
                    rank_df = df_hotspots.head(15).copy()
                    rank_df['Rank'] = np.arange(1, len(rank_df) + 1)
                    
                    # Add Severity Category
                    def get_severity_category(score):
                        if score <= 40:
                            return f"{status_dot('#00e400', 10)} Low"
                        elif score <= 70:
                            return f"{status_dot('#ffff00', 10)} Moderate"
                        elif score <= 90:
                            return f"{status_dot('#ff7e00', 10)} High"
                        else:
                            return f"{status_dot('#ff0000', 10)} Severe"
                            
                    rank_df['Severity'] = rank_df['hotspot_score'].map(get_severity_category)
                    rank_df['Nearest Area'] = rank_df['location'].fillna("N/A")
                    
                    rank_df = rank_df.rename(columns={
                        'lat': 'Latitude',
                        'lon': 'Longitude',
                        'hcho': 'HCHO (mol/m²)',
                        'aqi': 'Proxy AQI',
                        'hotspot_score': 'Anomaly Score'
                    })
                    
                    # Select specific columns
                    display_cols = ['Rank', 'Nearest Area', 'Anomaly Score', 'Severity', 'Proxy AQI']
                    st.dataframe(rank_df[display_cols], use_container_width=True, hide_index=True)
                    
            with tab2:
                # Biomass Burning tab logic
                st.markdown("<h3 style='color:#ffffff;'>Biomass Burning & Fire Detections</h3>", unsafe_allow_html=True)
                
                corr_data = load_fire_correlation(paths['fire_correlation'])
                wind_data = load_wind_rose(paths['wind_rose'])
                
                if corr_data is None:
                    st.warning("Biomass burning correlation data not compiled for this region.")
                else:
                    g_corr = corr_data.get('global_correlation', 0.0)
                    d_corr = corr_data.get('daily_correlation', 0.0)
                    
                    # 1. Statistics Cards
                    bb_col1, bb_col2, bb_col3 = st.columns(3)
                    with bb_col1:
                        st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>Spatio-Temporal Correlation (R)</div>
                            <div class='card-value'>{g_corr:.3f}</div>
                            <div class='text' style='font-size:0.85rem; color:#8b8c8d;'>Pearson R (Grid Fire Counts vs HCHO Columns)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with bb_col2:
                        st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>Daily Mean Correlation (R)</div>
                            <div class='card-value'>{d_corr:.3f}</div>
                            <div class='text' style='font-size:0.85rem; color:#8b8c8d;'>Temporal correlation (Mean Fires vs Mean HCHO)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    with bb_col3:
                        threshold = corr_data.get('statistical_threshold', 0.0)
                        st.markdown(f"""
                        <div class='card'>
                            <div class='card-title'>Anomaly Threshold</div>
                            <div class='card-value'>{threshold:.1f}</div>
                            <div class='text' style='font-size:0.85rem; color:#8b8c8d;'>Fires/day statistical upper-bound limit</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # 2. Daily Time Series Plots
                    st.markdown("<h4 style='color:#c77dff;'>Regional Fire Counts & HCHO Time Series</h4>", unsafe_allow_html=True)
                    
                    df_ts = pd.DataFrame(corr_data.get('time_series', []))
                    if not df_ts.empty:
                        df_ts['date'] = pd.to_datetime(df_ts['date'])
                        
                        # Plotly dual-axis chart
                        fig_ts = go.Figure()
                        
                        fig_ts.add_trace(go.Bar(
                            x=df_ts['date'],
                            y=df_ts['fire_count'],
                            name='Active Fires',
                            marker_color='rgba(255, 126, 0, 0.75)',
                            yaxis='y'
                        ))
                        
                        fig_ts.add_trace(go.Scatter(
                            x=df_ts['date'],
                            y=df_ts['mean_hcho'],
                            name='Mean HCHO',
                            line=dict(color='#c77dff', width=2),
                            yaxis='y2'
                        ))
                        
                        fig_ts.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='#1f2833',
                            plot_bgcolor='#1f2833',
                            margin=dict(l=40, r=40, t=20, b=40),
                            height=320,
                            yaxis=dict(
                                title=dict(text='Active Fire Count', font=dict(color='#ff7e00')),
                                tickfont=dict(color='#ff7e00')
                            ),
                            yaxis2=dict(
                                title=dict(text='HCHO Column (mol/m²)', font=dict(color='#c77dff')),
                                tickfont=dict(color='#c77dff'),
                                overlaying='y',
                                side='right'
                            ),
                            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                        )
                        st.plotly_chart(fig_ts, use_container_width=True)
                        
                    # 3. Fire Map & Trajectory Simulation (Row 2)
                    st.markdown("<h3 style='color:#ffffff;'>Biomass Burning Source Localization & Wind Transport</h3>", unsafe_allow_html=True)
                    map_col, period_col = st.columns([3, 2])
                    
                    with map_col:
                        st.markdown("<h4 style='color:#c77dff;'>Fires & Hotspots Distribution Map</h4>", unsafe_allow_html=True)
                        
                        # Try to load raw fire CSV cache
                        fire_csv_path = paths['fire_cache']
                        if os.path.exists(fire_csv_path):
                            df_fires_raw = pd.read_csv(fire_csv_path)
                        else:
                            df_fires_raw = pd.DataFrame()
                            
                        if df_fires_raw.empty:
                            st.info("No raw fire point detections available for this region.")
                        else:
                            center_lat_fires = float(df_fires_raw['latitude'].mean())
                            center_lon_fires = float(df_fires_raw['longitude'].mean())
                            
                            view_state_fires = pdk.ViewState(
                                latitude=center_lat_fires,
                                longitude=center_lon_fires,
                                zoom=6.5 if (bbox['max_lat'] - bbox['min_lat']) > 1.5 else 8.0,
                                pitch=30
                            )
                            
                            fires_layer = pdk.Layer(
                                "ScatterplotLayer",
                                df_fires_raw,
                                get_position="[longitude, latitude]",
                                get_color="[255, 126, 0, 180]", # bright orange
                                get_radius=2500,
                                pickable=True
                            )
                            
                            # Hotspots Polygon Layer (same as tab1)
                            hotspots_poly_layer = pdk.Layer(
                                "PolygonLayer",
                                df_map_hotspots,
                                get_polygon="polygon",
                                get_fill_color="[255, 0, 180, 120]", # translucent magenta
                                get_line_color="[255, 0, 180, 200]",
                                line_width_min_pixels=1.5,
                                filled=True,
                                pickable=True
                            )
                            
                            st.pydeck_chart(pdk.Deck(
                                layers=[hotspots_poly_layer, fires_layer],
                                initial_view_state=view_state_fires,
                                map_provider="carto",
                                map_style="dark",
                                tooltip={
                                    "text": "Latitude: {latitude:.3f}\nLongitude: {longitude:.3f}\nFRP: {frp:.1f} MW\nSatellite: {satellite}"
                                }
                            ))
                            
                            st.caption("Orange markers represent daily MODIS/VIIRS thermal fire detections. Magenta squares indicate predicted HCHO hotspot cells.")
                            
                    with period_col:
                        st.markdown("<h4 style='color:#c77dff;'>Statistical Biomass Burning Events</h4>", unsafe_allow_html=True)
                        periods = corr_data.get('burning_periods', [])
                        if len(periods) == 0:
                            st.info("No active biomass burning periods identified (daily fire counts remained below threshold).")
                        else:
                            df_periods = pd.DataFrame(periods)
                            df_periods = df_periods.rename(columns={
                                'start_date': 'Start Date',
                                'end_date': 'End Date',
                                'total_fires': 'Total Active Fires',
                                'mean_hcho': 'Average HCHO'
                            })
                            df_periods['Average HCHO'] = df_periods['Average HCHO'].map(lambda x: f"{x:.2e}")
                            st.dataframe(df_periods, use_container_width=True, hide_index=True)
                            st.markdown("""
                            **Biomass Burning Analysis:**
                            Periods are flagged when the daily regional active fire counts exceed the **Statistical Threshold** (mean + 2 standard deviations).
                            High correlation values (`R > 0.3`) strongly indicate that crop residue burning or forest fires are primary drivers of formaldehyde (HCHO) precursor emissions.
                            """)
