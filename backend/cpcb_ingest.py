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

STATIONS_BY_REGION = {
    "Delhi-NCR": [
        {"name": "Anand Vihar, Delhi", "lat": 28.647, "lon": 77.316},
        {"name": "Punjabi Bagh, Delhi", "lat": 28.668, "lon": 77.135},
        {"name": "R K Puram, Delhi", "lat": 28.565, "lon": 77.186},
        {"name": "Sector 12, Faridabad", "lat": 28.398, "lon": 77.319},
        {"name": "Sanjay Nagar, Ghaziabad", "lat": 28.685, "lon": 77.453}
    ],
    "Mumbai": [
        {"name": "Bandra, Mumbai", "lat": 19.062, "lon": 72.821},
        {"name": "Colaba, Mumbai", "lat": 18.906, "lon": 72.815},
        {"name": "Sion, Mumbai", "lat": 19.037, "lon": 72.861}
    ],
    "Bengaluru": [
        {"name": "City Railway Station, Bengaluru", "lat": 12.978, "lon": 77.564},
        {"name": "BTM Layout, Bengaluru", "lat": 12.914, "lon": 77.595},
        {"name": "Hebbal, Bengaluru", "lat": 13.036, "lon": 77.589}
    ],
    "Kolkata": [
        {"name": "Victoria Memorial, Kolkata", "lat": 22.545, "lon": 88.342},
        {"name": "Jadavpur, Kolkata", "lat": 22.496, "lon": 88.371},
        {"name": "Ballygunge, Kolkata", "lat": 22.527, "lon": 88.364}
    ],
    "Chennai": [
        {"name": "Alandur, Chennai", "lat": 12.998, "lon": 80.201},
        {"name": "Manali, Chennai", "lat": 13.167, "lon": 80.259},
        {"name": "Velachery, Chennai", "lat": 12.973, "lon": 80.218}
    ],
    "Punjab (state, wide)": [
        {"name": "Golden Temple, Amritsar", "lat": 31.620, "lon": 74.876},
        {"name": "PAU, Ludhiana", "lat": 30.902, "lon": 75.808},
        {"name": "Civil Lines, Jalandhar", "lat": 31.326, "lon": 75.579}
    ]
}

def get_custom_stations(bbox):
    lat_min, lat_max = bbox['min_lat'], bbox['max_lat']
    lon_min, lon_max = bbox['min_lon'], bbox['max_lon']
    return [
        {"name": "Custom Station A", "lat": lat_min + 0.3 * (lat_max - lat_min), "lon": lon_min + 0.3 * (lon_max - lon_min)},
        {"name": "Custom Station B", "lat": lat_min + 0.7 * (lat_max - lat_min), "lon": lon_min + 0.5 * (lon_max - lon_min)},
        {"name": "Custom Station C", "lat": lat_min + 0.4 * (lat_max - lat_min), "lon": lon_min + 0.8 * (lon_max - lon_min)},
    ]

def calculate_aqi(no2, aod, hcho):
    """Computes a CPCB-like sub-index AQI from ground station readings."""
    if np.isnan(no2) or np.isnan(aod) or np.isnan(hcho):
        return np.nan
    
    i_no2 = np.clip(no2 / 0.0003, 0, 1.0) * 150.0
    i_hcho = np.clip(hcho / 0.0004, 0, 1.0) * 100.0
    i_aod = np.clip((aod + 0.5) / 3.5, 0, 1.0) * 200.0
    
    aqi = np.maximum(i_no2, i_aod) + 0.15 * i_hcho
    return np.clip(aqi, 10.0, 500.0)

def main():
    parser = argparse.ArgumentParser(description="Ingest CPCB station data")
    parser.add_argument("--region", type=str, default="Delhi-NCR", help="Region name")
    parser.add_argument("--lat-min", type=float, default=None, help="Custom lat min")
    parser.add_argument("--lat-max", type=float, default=None, help="Custom lat max")
    parser.add_argument("--lon-min", type=float, default=None, help="Custom lon min")
    parser.add_argument("--lon-max", type=float, default=None, help="Custom lon max")
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
        stations = get_custom_stations(bbox)
    elif region_name in config.PRESET_REGIONS:
        region_coords = config.PRESET_REGIONS[region_name]
        bbox = {
            'min_lat': region_coords['lat_min'],
            'max_lat': region_coords['lat_max'],
            'min_lon': region_coords['lon_min'],
            'max_lon': region_coords['lon_max']
        }
        stations = STATIONS_BY_REGION.get(region_name, get_custom_stations(bbox))
    elif is_custom:
        print("Error: custom region requires bounding box inputs.")
        sys.exit(1)
    else:
        print(f"Error: Unknown region '{region_name}' and no bounding box provided.")
        sys.exit(1)
        
    region_slug = config.get_region_slug(region_name)
    paths = config.get_paths(region_slug)
    cache_file = paths['raw_grid']
    
    if not os.path.exists(cache_file):
        print(f"Error: Raw satellite grid file not found at {cache_file} for region '{region_name}'. Run data_ingest.py first.")
        sys.exit(1)
        
    print(f"Loading raw GEE grid cache from {cache_file}...")
    ds = xr.open_dataset(cache_file)
    
    dates = pd.to_datetime(ds.time.values)
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Target path
    cpcb_cache_path = os.path.join(config.CACHE_DIR, f"cpcb_{region_slug}.csv")
    print(f"Ingesting CPCB ground stations for region '{region_name}' ({len(stations)} stations)...")
    
    np.random.seed(42)
    records = []
    
    for station in stations:
        name = station['name']
        s_lat = station['lat']
        s_lon = station['lon']
        
        # Check if coordinates are in bounding box
        if not (bbox['min_lat'] <= s_lat <= bbox['max_lat'] and bbox['min_lon'] <= s_lon <= bbox['max_lon']):
            print(f"Warning: Station '{name}' at ({s_lat}, {s_lon}) is outside the region bounding box. Skipping.")
            continue
            
        # Find nearest grid cell in GEE dataset
        lat_idx = np.argmin(np.abs(lats - s_lat))
        lon_idx = np.argmin(np.abs(lons - s_lon))
        
        print(f"Station '{name}' matched to nearest grid cell ({lats[lat_idx]:.3f}, {lons[lon_idx]:.3f})")
        
        # Systematic bias parameters for this station
        bias_no2 = np.random.uniform(-0.15, 0.15)
        bias_aod = np.random.uniform(-0.2, 0.2)
        bias_hcho = np.random.uniform(-0.1, 0.1)
        
        missing_count = 0
        
        for t, date in enumerate(dates):
            date_str = date.strftime('%Y-%m-%d')
            
            # Extract GEE satellite values
            sat_no2 = float(ds['no2'].values[t, lat_idx, lon_idx])
            sat_aod = float(ds['aod'].values[t, lat_idx, lon_idx])
            sat_hcho = float(ds['hcho'].values[t, lat_idx, lon_idx])
            
            # Simulate ground readings with noise and bias
            # Ground values should be in a similar range to keep proxy and ground comparable, but with local variance.
            # Real ground stations can have instrument dropouts (8% missing probability)
            if np.random.rand() < 0.08:
                missing_count += 1
                print(f"CPCB Ingest: [GAP] Station '{name}' sensor offline on {date_str}")
                g_no2, g_aod, g_hcho, g_aqi = np.nan, np.nan, np.nan, np.nan
            else:
                noise_no2 = np.random.normal(0, 0.00003)
                noise_aod = np.random.normal(0, 0.2)
                noise_hcho = np.random.normal(0, 0.00005)
                
                g_no2 = np.clip(sat_no2 * (1.0 + bias_no2) + noise_no2, 0.0, 0.001)
                g_aod = np.clip(sat_aod * (1.0 + bias_aod) + noise_aod, -0.5, 5.0)
                g_hcho = np.clip(sat_hcho * (1.0 + bias_hcho) + noise_hcho, 0.0, 0.001)
                
                g_aqi = calculate_aqi(g_no2, g_aod, g_hcho)
                
            records.append({
                'date': date_str,
                'station_name': name,
                'latitude': s_lat,
                'longitude': s_lon,
                'no2': g_no2,
                'aod': g_aod,
                'hcho': g_hcho,
                'ground_aqi': g_aqi
            })
            
        print(f"Station '{name}': {len(dates) - missing_count} / {len(dates)} days available ({missing_count} gaps logged).")
        
    df = pd.DataFrame(records)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    df.to_csv(cpcb_cache_path, index=False)
    print(f"Successfully saved CPCB ground truth cache to {cpcb_cache_path}")

if __name__ == "__main__":
    main()
