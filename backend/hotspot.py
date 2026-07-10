import os
import sys
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.ensemble import IsolationForest

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from geocode_utils import geocode_coordinates

def main():
    parser = argparse.ArgumentParser(description="Find HCHO Hotspots via Isolation Forest")
    parser.add_argument("--region", type=str, default="Delhi-NCR", help="Region name")
    parser.add_argument("--lat-min", type=float, default=None, help="Custom lat min")
    parser.add_argument("--lat-max", type=float, default=None, help="Custom lat max")
    parser.add_argument("--lon-min", type=float, default=None, help="Custom lon min")
    parser.add_argument("--lon-max", type=float, default=None, help="Custom lon max")
    args = parser.parse_args()
    
    region_name = args.region
    is_custom = region_name.lower() == "custom"
    
    if is_custom:
        region_name = f"Custom_{args.lat_min}_{args.lat_max}_{args.lon_min}_{args.lon_max}"
        
    region_slug = config.get_region_slug(region_name)
    paths = config.get_paths(region_slug)
    processed_grid_path = paths['processed_grid']
    
    if not os.path.exists(processed_grid_path):
        print(f"Error: Processed grid not found at {processed_grid_path} for region '{region_name}'. Preprocess first.")
        sys.exit(1)
        
    print(f"Loading processed grid for region '{region_name}' from {processed_grid_path}...")
    ds = xr.open_dataset(processed_grid_path)
    
    # Search backwards for the latest time step with valid satellite data
    time_idx = len(ds.time) - 1
    valid_data_found = False
    for idx in range(len(ds.time) - 1, -1, -1):
        if float(ds.isel(time=idx)['hcho'].mean()) > 1e-7:
            time_idx = idx
            valid_data_found = True
            break
            
    if not valid_data_found:
        print("Warning: No valid HCHO data found in any time step. Defaulting to last index.")
        time_idx = len(ds.time) - 1
        
    ds_latest = ds.isel(time=time_idx)
    target_date_str = str(pd.to_datetime(ds_latest.time.values).strftime('%Y-%m-%d'))
    print(f"Selected date for hotspot detection: {target_date_str} (time index {time_idx})")
    
    lats = ds_latest.lat.values
    lons = ds_latest.lon.values
    
    # Flatten the spatial grid into a table of features
    records = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            hcho = float(ds_latest['hcho'].values[i, j])
            no2 = float(ds_latest['no2'].values[i, j])
            aod = float(ds_latest['aod'].values[i, j])
            aqi = float(ds_latest['aqi'].values[i, j])
            
            records.append({
                'lat': float(lat),
                'lon': float(lon),
                'hcho': hcho,
                'no2': no2,
                'aod': aod,
                'aqi': aqi
            })
            
    df = pd.DataFrame(records)
    
    # Features for Isolation Forest
    features = ['hcho', 'no2', 'aod']
    X = df[features].values
    
    # Fit Isolation Forest
    print("Fitting Isolation Forest model for hotspot detection...")
    model = IsolationForest(n_estimators=100, contamination=0.08, random_state=42)
    df['anomaly_pred'] = model.fit_predict(X)
    
    df['raw_score'] = model.decision_function(X)
    
    hcho_median = df['hcho'].median()
    hcho_95th = df['hcho'].quantile(0.95)
    df['is_hotspot'] = (df['anomaly_pred'] == -1) & (df['hcho'] > hcho_median)
    
    min_dec = df['raw_score'].min()
    max_dec = df['raw_score'].max()
    if max_dec != min_dec:
        df['hotspot_score'] = 100.0 * (max_dec - df['raw_score']) / (max_dec - min_dec)
    else:
        df['hotspot_score'] = 50.0
    
    # Filter out hotspots list
    hotspots_df = df[df['is_hotspot']].copy()
    hotspots_df = hotspots_df.sort_values(by='hotspot_score', ascending=False)
    
    hotspots_list = hotspots_df[['lat', 'lon', 'hcho', 'no2', 'aod', 'aqi', 'hotspot_score']].to_dict(orient='records')
    
    # Geocode coordinates offline
    coords = [(h['lat'], h['lon']) for h in hotspots_list]
    locations = geocode_coordinates(coords)
    for h, loc in zip(hotspots_list, locations):
        h['location'] = loc
    
    # Construct a GeoJSON structure
    geojson_features = []
    for h in hotspots_list:
        geojson_features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [h['lon'], h['lat']]
            },
            'properties': {
                'hcho': h['hcho'],
                'no2': h['no2'],
                'aod': h['aod'],
                'aqi': h['aqi'],
                'hotspot_score': h['hotspot_score']
            }
        })
        
    geojson = {
        'type': 'FeatureCollection',
        'features': geojson_features
    }
    
    # Save output
    output_data = {
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_date': target_date_str,
        'total_grid_cells': len(df),
        'hotspots_count': len(hotspots_list),
        'hcho_spatial_median': float(hcho_median),
        'hcho_spatial_mean': float(df['hcho'].mean()),
        'hcho_spatial_95th': float(hcho_95th),
        'hotspots': hotspots_list,
        'geojson': geojson
    }
    
    output_path = paths['hotspots']
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Detected {len(hotspots_list)} hotspots for date {target_date_str}.")
    print(f"Saved hotspots data to {output_path}")

if __name__ == "__main__":
    main()
