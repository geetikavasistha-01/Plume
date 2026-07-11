import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_synthetic_data(bbox, region_name):
    """Generates a realistic synthetic dataset for a given bounding box."""
    print(f"Generating synthetic datasets for region '{region_name}'...")
    
    # Dimensions
    res = config.get_resolution(bbox)
    lats = np.arange(bbox['min_lat'], bbox['max_lat'] + 0.01, res)
    lons = np.arange(bbox['min_lon'], bbox['max_lon'] + 0.01, res)
    dates = pd.date_range(start=config.START_DATE, end=config.END_DATE - timedelta(days=1))
    
    n_lat = len(lats)
    n_lon = len(lons)
    n_time = len(dates)
    
    # Create coordinates grid
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    
    np.random.seed(42)
    
    # Target center (Delhi-like hotspot centered in the region)
    center_lat = (bbox['min_lat'] + bbox['max_lat']) / 2.0
    center_lon = (bbox['min_lon'] + bbox['max_lon']) / 2.0
    dist_from_center = np.sqrt((lat_grid - center_lat)**2 + (lon_grid - center_lon)**2)
    
    # HCHO (Formaldehyde) - range 0 to 0.0008 mol/m^2
    spatial_hcho = 0.0003 * np.exp(-dist_from_center**2 / 0.1) + 0.0001
    hcho_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        temp_cycle = 1.0 + 0.2 * np.sin(2 * np.pi * t / 30)
        noise = np.random.normal(0, 0.05, (n_lat, n_lon))
        hcho_data[t] = np.clip(spatial_hcho * temp_cycle + noise * 0.00005, 0, 0.001)
        
    # NO2 - range 0 to 0.0005 mol/m^2
    spatial_no2 = 0.00025 * np.exp(-dist_from_center**2 / 0.05) + 0.00005
    no2_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        temp_cycle = 1.0 + 0.3 * np.sin(2 * np.pi * t / 7)
        noise = np.random.normal(0, 0.05, (n_lat, n_lon))
        no2_data[t] = np.clip(spatial_no2 * temp_cycle + noise * 0.00003, 0, 0.0008)
        
    # AOD (Aerosol Index) - range 0 to 4
    spatial_aod = 1.8 * np.exp(-dist_from_center**2 / 0.3) + 0.5
    aod_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        temp_cycle = 1.0 + 0.15 * np.sin(2 * np.pi * t / 20)
        noise = np.random.normal(0, 0.1, (n_lat, n_lon))
        aod_data[t] = np.clip(spatial_aod * temp_cycle + noise * 0.2, -0.5, 5.0)

    # Temperature 2m (in Kelvin, range 300 to 318 K)
    temp_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        base_temp = 305.0 + 5.0 * (t / n_time)
        temp_data[t] = base_temp + 3.0 * np.sin(2 * np.pi * t / 15) + np.random.normal(0, 0.5, (n_lat, n_lon))

    # Wind U/V (range -5 to 8 m/s)
    u_wind_data = np.zeros((n_time, n_lat, n_lon))
    v_wind_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        u_wind_data[t] = 2.0 * np.sin(2 * np.pi * t / 10) + np.random.normal(0, 1.0, (n_lat, n_lon))
        v_wind_data[t] = -1.0 * np.cos(2 * np.pi * t / 12) + np.random.normal(0, 1.0, (n_lat, n_lon))

    # Dewpoint temperature (Kelvin)
    dewpoint_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        dewpoint_data[t] = temp_data[t] - (10.0 + np.random.normal(0, 1.0, (n_lat, n_lon)))

    # Precipitation (meters, sparse events)
    precip_data = np.zeros((n_time, n_lat, n_lon))
    for t in range(n_time):
        if np.random.rand() < 0.15:
            precip_data[t] = np.clip(np.random.exponential(0.01, (n_lat, n_lon)), 0, 0.05)

    # Combine into Xarray Dataset
    ds = xr.Dataset(
        data_vars={
            'hcho': (['time', 'lat', 'lon'], hcho_data),
            'no2': (['time', 'lat', 'lon'], no2_data),
            'aod': (['time', 'lat', 'lon'], aod_data),
            'temp': (['time', 'lat', 'lon'], temp_data),
            'u_wind': (['time', 'lat', 'lon'], u_wind_data),
            'v_wind': (['time', 'lat', 'lon'], v_wind_data),
            'dewpoint': (['time', 'lat', 'lon'], dewpoint_data),
            'precip': (['time', 'lat', 'lon'], precip_data),
        },
        coords={
            'time': dates,
            'lat': lats,
            'lon': lons
        },
        attrs={
            'description': f'ISRO AQI HCHO Pipeline Region Dataset - {region_name} (Synthetic)',
            'region': region_name,
            'is_demo': 'true'
        }
    )
    return ds

def pull_gee_data(bbox, region_name):
    """Pulls gridded variables from Google Earth Engine for a specific bbox."""
    import ee
    import geemap
    
    print(f"Initializing Earth Engine for region '{region_name}' with Project ID: {config.GEE_PROJECT_ID}...")
    try:
        import streamlit as st
        # If running inside Streamlit Cloud and the gcp secrets block is defined
        if "gcp" in st.secrets:
            print("Authenticating with service account credentials from st.secrets['gcp']...")
            from google.oauth2 import service_account
            gcp_info = st.secrets["gcp"]
            credentials = service_account.Credentials.from_service_account_info(dict(gcp_info))
            ee.Initialize(credentials, project=config.GEE_PROJECT_ID)
        else:
            # Fallback to local machine GEE config
            ee.Initialize(project=config.GEE_PROJECT_ID)
    except Exception as e:
        print(f"Streamlit secrets auth failed or not available ({e}). Attempting default credentials...")
        ee.Initialize(project=config.GEE_PROJECT_ID)
    
    aoi = ee.Geometry.Rectangle([
        bbox['min_lon'],
        bbox['min_lat'],
        bbox['max_lon'],
        bbox['max_lat']
    ])
    
    dates = [config.START_DATE + timedelta(days=i) for i in range(60)]
    start_date_str = dates[0].strftime('%Y-%m-%d')
    end_date_str = (dates[-1] + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # GEE Collections mapping
    collections = {
        'hcho': ('COPERNICUS/S5P/OFFL/L3_HCHO', 'tropospheric_HCHO_column_number_density'),
        'no2': ('COPERNICUS/S5P/OFFL/L3_NO2', 'tropospheric_NO2_column_number_density'),
        'aod': ('COPERNICUS/S5P/OFFL/L3_AER_AI', 'absorbing_aerosol_index'),
        'temp': ('ECMWF/ERA5_LAND/HOURLY', 'temperature_2m'),
        'u_wind': ('ECMWF/ERA5_LAND/HOURLY', 'u_component_of_wind_10m'),
        'v_wind': ('ECMWF/ERA5_LAND/HOURLY', 'v_component_of_wind_10m'),
        'dewpoint': ('ECMWF/ERA5_LAND/HOURLY', 'dewpoint_temperature_2m'),
        'precip': ('ECMWF/ERA5_LAND/HOURLY', 'total_precipitation')
    }
    
    res = config.get_resolution(bbox)
    scale = int(res * 100000)
    print(f"Using gridding resolution: {res}° (scale={scale}m) for region '{region_name}'")
    
    grids = {}
    
    # Create server-side list of dates for mapping
    date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    date_list = ee.List(date_strings)
    
    for var, (coll_name, band_name) in collections.items():
        print(f"Fetching variable '{var}' from GEE collection '{coll_name}'...")
        
        # Load collection once for the entire range
        full_coll = ee.ImageCollection(coll_name) \
            .filterDate(start_date_str, end_date_str) \
            .select(band_name)
            
        def get_daily_mean(d_str):
            d_str = ee.String(d_str)
            d1 = ee.Date(d_str)
            d2 = d1.advance(1, 'day')
            daily = full_coll.filterDate(d1, d2)
            
            mean_img = ee.Algorithms.If(
                daily.size().gt(0),
                daily.mean(),
                ee.Image.constant(0.0).rename(band_name)
            )
            return ee.Image(mean_img).set('system:time_start', d1.millis())
            
        # Map over the dates list on the GEE server
        daily_images = ee.ImageCollection(date_list.map(get_daily_mean))
        combined_img = daily_images.toBands()
        
        print(f"Converting GEE {var} image to numpy...")
        arr = geemap.ee_to_numpy(combined_img, region=aoi, scale=scale)
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.nan_to_num(arr, nan=0.0)
        grids[var] = arr
        print(f"Loaded '{var}' with shape {arr.shape}")
        
    n_time, n_lat, n_lon = grids['hcho'].shape
    lats = np.linspace(bbox['min_lat'], bbox['max_lat'], n_lat)
    lons = np.linspace(bbox['min_lon'], bbox['max_lon'], n_lon)
    
    ds = xr.Dataset(
        data_vars={
            'hcho': (['time', 'lat', 'lon'], grids['hcho']),
            'no2': (['time', 'lat', 'lon'], grids['no2']),
            'aod': (['time', 'lat', 'lon'], grids['aod']),
            'temp': (['time', 'lat', 'lon'], grids['temp']),
            'u_wind': (['time', 'lat', 'lon'], grids['u_wind']),
            'v_wind': (['time', 'lat', 'lon'], grids['v_wind']),
            'dewpoint': (['time', 'lat', 'lon'], grids['dewpoint']),
            'precip': (['time', 'lat', 'lon'], grids['precip']),
        },
        coords={
            'time': pd.date_range(start=config.START_DATE, periods=n_time),
            'lat': lats,
            'lon': lons
        },
        attrs={
            'description': f'ISRO AQI HCHO Pipeline Region Dataset - {region_name} (Live GEE)',
            'region': region_name,
            'is_demo': 'false'
        }
    )
    return ds

def main():
    parser = argparse.ArgumentParser(description="Ingest GEE / Synthetic data")
    parser.add_argument("--region", type=str, default="Delhi-NCR", help="Region name")
    parser.add_argument("--lat-min", type=float, default=None, help="Custom lat min")
    parser.add_argument("--lat-max", type=float, default=None, help="Custom lat max")
    parser.add_argument("--lon-min", type=float, default=None, help="Custom lon min")
    parser.add_argument("--lon-max", type=float, default=None, help="Custom lon max")
    parser.add_argument("--use-cache", action="store_true", help="Skip pull if cache exists")
    args = parser.parse_args()
    
    region_name = args.region
    is_custom = region_name.lower() == "custom"
    
    # Bounding box resolution
    if args.lat_min is not None and args.lat_max is not None and args.lon_min is not None and args.lon_max is not None:
        bbox = {
            'min_lat': args.lat_min,
            'max_lat': args.lat_max,
            'min_lon': args.lon_min,
            'max_lon': args.lon_max
        }
    elif region_name in config.PRESET_REGIONS:
        region_coords = config.PRESET_REGIONS[region_name]
        bbox = {
            'min_lat': region_coords['lat_min'],
            'max_lat': region_coords['lat_max'],
            'min_lon': region_coords['lon_min'],
            'max_lon': region_coords['lon_max']
        }
    elif is_custom:
        print("Error: custom region requires bounding box inputs.")
        sys.exit(1)
    else:
        print(f"Error: Unknown region '{region_name}' and no bounding box provided.")
        sys.exit(1)
        
    region_slug = config.get_region_slug(region_name)
    paths = config.get_paths(region_slug)
    cache_file = paths['raw_grid']
    
    if args.use_cache and os.path.exists(cache_file):
        print(f"Cached data found at {cache_file} for region '{region_name}'. Skipping ingestion.")
        return
        
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    
    try:
        ds = pull_gee_data(bbox, region_name)
        print(f"Successfully pulled GEE data for region '{region_name}'.")
    except Exception as e:
        print(f"Error during GEE data pull: {e}. Falling back to synthetic.")
        ds = get_synthetic_data(bbox, region_name)
        
    ds.to_netcdf(cache_file)
    print(f"Saved dataset to cache file at {cache_file}")

if __name__ == "__main__":
    main()
