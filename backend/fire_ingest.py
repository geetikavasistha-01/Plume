import os
import sys
import argparse
import requests
import io
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def generate_synthetic_fires(bbox, start_date, end_date):
    """Generates a realistic spatial-temporal dataset of active fire points within a bounding box."""
    print("Generating high-fidelity synthetic active fire detections...")
    np.random.seed(101)
    
    dates = pd.date_range(start=start_date, end=end_date)
    records = []
    
    # Define 2-3 spatial fire source clusters (e.g. crop fields, industrial outskirts)
    num_clusters = 3
    clusters = []
    for _ in range(num_clusters):
        lat_c = np.random.uniform(bbox['min_lat'] + 0.1, bbox['max_lat'] - 0.1)
        lon_c = np.random.uniform(bbox['min_lon'] + 0.1, bbox['max_lon'] - 0.1)
        # Intensity scale (how active this cluster is)
        intensity = np.random.uniform(5.0, 30.0)
        clusters.append((lat_c, lon_c, intensity))
        
    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        # Fires peak in mid-May to early-June, and die down in July due to monsoon/rains
        t_factor = 1.0 - (date - start_date).days / (end_date - start_date).days
        # Seasonal trend function with peak
        seasonal_mult = np.clip(1.5 * np.exp(-((date - start_date).days - 15)**2 / 120.0), 0.05, 2.0)
        
        for lat_c, lon_c, base_intensity in clusters:
            # Number of daily fires in this cluster
            n_fires = int(np.random.poisson(base_intensity * seasonal_mult))
            
            for _ in range(n_fires):
                # Spread around cluster center
                f_lat = np.clip(np.random.normal(lat_c, 0.15), bbox['min_lat'], bbox['max_lat'])
                f_lon = np.clip(np.random.normal(lon_c, 0.15), bbox['min_lon'], bbox['max_lon'])
                
                # Confidence: 50% to 100%
                conf = float(np.random.uniform(50, 100))
                # FRP (Fire Radiative Power) in MW: exponential distribution
                frp = float(np.random.exponential(25.0) + 2.0)
                # Brightness in Kelvin: 300K to 450K
                bright = float(310.0 + np.random.exponential(15.0))
                
                satellite = np.random.choice(["MODIS", "VIIRS"], p=[0.3, 0.7])
                
                records.append({
                    'latitude': f_lat,
                    'longitude': f_lon,
                    'confidence': conf,
                    'brightness': bright,
                    'frp': frp,
                    'acq_date': date_str,
                    'satellite': satellite
                })
                
    df = pd.DataFrame(records)
    print(f"Generated {len(df)} synthetic active fire points.")
    return df

def fetch_firms_data(bbox, start_date, end_date, api_key):
    """Queries NASA FIRMS API for active fire CSV data over the date range."""
    # Bounding box format: west,south,east,north
    area_str = f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}"
    
    # Sources to query
    sources = ["MODIS_NRT", "VIIRS_SNPP_NRT"]
    all_dfs = []
    
    # The API day_range can be up to 10 days, but standard is 1-5. Let's query in 5-day intervals.
    chunk_size = 5
    current_date = start_date
    
    while current_date <= end_date:
        # Calculate actual chunk size (up to chunk_size or remaining days)
        remaining_days = (end_date - current_date).days + 1
        query_days = min(chunk_size, remaining_days)
        date_str = current_date.strftime('%Y-%m-%d')
        
        for source in sources:
            print(f"Fetching FIRMS fire data: {source} for {query_days} days starting {date_str}...")
            url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/{source}/{area_str}/{query_days}/{date_str}"
            
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    csv_text = response.text
                    if "latitude" in csv_text:
                        df_chunk = pd.read_csv(io.StringIO(csv_text))
                        # Rename columns if needed to match standard
                        if 'acq_date' in df_chunk.columns:
                            satellite_label = "MODIS" if "MODIS" in source else "VIIRS"
                            df_chunk['satellite'] = satellite_label
                            # Map brightness
                            if 'bright_ti4' in df_chunk.columns:
                                df_chunk = df_chunk.rename(columns={'bright_ti4': 'brightness'})
                            elif 'brightness' not in df_chunk.columns:
                                df_chunk['brightness'] = 330.0
                                
                            all_dfs.append(df_chunk[['latitude', 'longitude', 'confidence', 'brightness', 'frp', 'acq_date', 'satellite']])
                    else:
                        print(f"No fire detections found for {source} in this chunk.")
                else:
                    print(f"Error calling FIRMS API ({response.status_code}): {response.text}")
            except Exception as e:
                print(f"Exception during FIRMS API call: {e}")
                
        current_date += timedelta(days=query_days)
        
    if all_dfs:
        merged_df = pd.concat(all_dfs, ignore_index=True)
        print(f"Retrieved {len(merged_df)} active fire points from NASA FIRMS API.")
        return merged_df
    else:
        raise ValueError("No data could be retrieved from FIRMS API.")

def main():
    parser = argparse.ArgumentParser(description="Ingest MODIS/VIIRS active fire counts")
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
    fire_csv_path = paths['fire_cache']
    
    api_key = os.environ.get("FIRMS_API_KEY")
    
    # Parse dates from GEE raw file to align ranges
    raw_grid_path = paths['raw_grid']
    if os.path.exists(raw_grid_path):
        ds = xr.open_dataset(raw_grid_path)
        dates = pd.to_datetime(ds.time.values)
        start_date = dates.min()
        end_date = dates.max()
        print(f"Aligned fire date range to GEE time dimension: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    else:
        start_date = config.START_DATE
        end_date = config.END_DATE
        print(f"GEE Raw file not found. Defaulting date range to config dates: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
    if api_key:
        print(f"Found FIRMS API Key in environment. Querying NASA FIRMS web services...")
        try:
            df = fetch_firms_data(bbox, start_date, end_date, api_key)
        except Exception as e:
            print(f"Failed to fetch FIRMS API data: {e}. Falling back to synthetic generator.")
            df = generate_synthetic_fires(bbox, start_date, end_date)
    else:
        print("Warning: FIRMS_API_KEY environment variable not set. Falling back to synthetic generator.")
        df = generate_synthetic_fires(bbox, start_date, end_date)
        
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    df.to_csv(fire_csv_path, index=False)
    print(f"Successfully cached fire points to {fire_csv_path}")

if __name__ == "__main__":
    main()
