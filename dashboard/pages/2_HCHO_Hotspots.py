import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import importlib
from backend import config
importlib.reload(config)

# Set page config
st.set_page_config(
    page_title="HCHO Hotspots",
    page_icon="🔥",
    layout="wide"
)

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
st.markdown(custom_css, unsafe_allow_html=True)

# Import helper
from dashboard import helper

# Render sidebar region selector and get selected region
selected_region, region_slug, bbox, paths = helper.render_sidebar_and_get_region()

# Helper function to load hotspots
@st.cache_data
def load_hotspots(hotspots_path):
    if not os.path.exists(hotspots_path):
        return None
    with open(hotspots_path, 'r') as f:
        return json.load(f)

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
        summary_text = f"On **{data_date}**, we detected **{len(hotspots_list)} out of {hotspots_data.get('total_grid_cells', 0)}** areas in **{selected_region}** with unusually high formaldehyde (HCHO) levels. The most affected area is near **{top_place}**."
    else:
        summary_text = f"On **{data_date}**, we found **no significant** formaldehyde (HCHO) hotspot anomalies across the **{hotspots_data.get('total_grid_cells', 0)}** grid cells of **{selected_region}**."
        
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
        # Filter hotspots to drop NaN/Inf/Out-of-range lat/lon
        records = []
        res = config.RESOLUTION
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
                'lat': lat,
                'lon': lon,
                'hcho': hcho,
                'no2': no2,
                'aod': aod,
                'aqi': aqi,
                'hotspot_score': score,
                'r': r,
                'g': g,
                'b': b
            })
            
        df_map_hotspots = pd.DataFrame(records)
        
        # 2. Main Page Layout: Map on left, ranked table on right
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
            st.markdown("""
            <div style='background: rgba(31, 40, 51, 0.65); padding: 0.8rem; border-radius: 8px; margin-top: 1rem; border: 1px solid rgba(199, 125, 255, 0.3);'>
                <div style='font-weight: 600; font-size: 0.85rem; color: #c77dff; margin-bottom: 0.5rem;'>🔥 HOTSPOT ANOMALY SEVERITY GRADIENT</div>
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
                    return "🟢 Low"
                elif score <= 70:
                    return "🟡 Moderate"
                elif score <= 90:
                    return "🟠 High"
                else:
                    return "🔴 Severe"
            
            rank_df['Severity'] = rank_df['hotspot_score'].map(get_severity_category)
            rank_df['Nearest Area'] = rank_df['location'].fillna("N/A")
            
            rank_df = rank_df.rename(columns={
                'lat': 'Latitude',
                'lon': 'Longitude',
                'hcho': 'HCHO (mol/m²)',
                'aqi': 'Proxy AQI',
                'hotspot_score': 'Severity Score'
            })
            
            # Format outputs
            rank_df['HCHO (mol/m²)'] = rank_df['HCHO (mol/m²)'].map(lambda x: f"{x:.2e}")
            rank_df['Proxy AQI'] = rank_df['Proxy AQI'].map(lambda x: f"{x:.1f}")
            rank_df['Severity Score'] = rank_df['Severity Score'].map(lambda x: f"{x:.1f}")
            
            st.dataframe(
                rank_df[['Rank', 'Nearest Area', 'Severity', 'HCHO (mol/m²)', 'Proxy AQI', 'Severity Score']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Nearest Area": st.column_config.TextColumn(
                        "Nearest Area",
                        help="Resolved geographical place name using offline reverse geocoding."
                    ),
                    "Severity": st.column_config.TextColumn(
                        "Severity",
                        help="Categorical label representing the severity of the HCHO anomaly."
                    ),
                    "Proxy AQI": st.column_config.TextColumn(
                        "Proxy AQI",
                        help="An estimated Air Quality Index derived from satellite pollutant readings (not a ground-station measurement)."
                    ),
                    "Severity Score": st.column_config.TextColumn(
                        "Severity Score",
                        help="Isolation Forest anomaly score representing how outlying this grid cell is compared to others."
                    )
                }
            )
            
        # 3. Distribution details at bottom
        st.markdown("---")
        st.markdown("<h3 style='color:#ffffff;'>Hotspot Chemical Context Analysis</h3>", unsafe_allow_html=True)
        
        hist_col, scatter_col = st.columns(2)
        
        with hist_col:
            st.markdown("<h4 style='color:#c77dff;'>HCHO Concentration Distribution</h4>", unsafe_allow_html=True)
            # Create a histogram using plotly
            fig_hist = px.histogram(
                df_hotspots, 
                x='hcho', 
                nbins=30,
                color_discrete_sequence=['#c77dff'],
                labels={'hcho': 'HCHO Concentration (mol/m²)'}
            )
            fig_hist.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2833',
                plot_bgcolor='#1f2833',
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title='HCHO Concentration (mol/m²)',
                yaxis_title='Frequency (Grid Cells)',
                height=300
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
        with scatter_col:
            st.markdown("<h4 style='color:#c77dff;'>HCHO vs. NO₂ Anomaly Space</h4>", unsafe_allow_html=True)
            # Scatter plot of HCHO vs NO2 colored by severity score
            fig_scat = px.scatter(
                df_hotspots,
                x='no2',
                y='hcho',
                color='hotspot_score',
                color_continuous_scale='Portland',
                labels={
                    'no2': 'NO₂ Column (mol/m²)',
                    'hcho': 'HCHO Column (mol/m²)',
                    'hotspot_score': 'Severity'
                }
            )
            fig_scat.update_layout(
                template='plotly_dark',
                paper_bgcolor='#1f2833',
                plot_bgcolor='#1f2833',
                margin=dict(l=40, r=40, t=20, b=40),
                xaxis_title='NO₂ Concentration (mol/m²)',
                yaxis_title='HCHO Concentration (mol/m²)',
                height=300
            )
            st.plotly_chart(fig_scat, use_container_width=True)
            
        st.markdown("""
        **Hotspot Analysis Note:** Isolation Forest operates over the 3D pollutant space `[HCHO, NO2, AOD]` to flag grid cells that exhibit unusual multi-variate signatures. 
        Cells exhibiting high HCHO densities and corresponding high values of NO2/AOD are isolated first, leading to elevated **Severity Scores**. 
        These anomalies typically correspond to localized chemical combustion sources, urban traffic intersections, or industrial discharge plumes in the NCR.
        """)
