import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def get_wind_direction_bin(u, v):
    """
    Computes wind direction in degrees (0-360) where wind blows FROM,
    and returns its binned sector.
    """
    if u == 0 and v == 0:
        return "Calm"
    # Angle in radians
    angle_rad = np.arctan2(v, u)
    # Convert to degrees (0 to 360)
    angle_deg = np.degrees(angle_rad)
    # Wind direction is where wind is blowing FROM
    # vector (u, v) points to where wind is blowing TO
    # so wind direction is opposite: (270 - angle) % 360
    wind_from_deg = (270.0 - angle_deg) % 360.0
    
    # 8 bins
    bins = {
        "N": (337.5, 22.5),
        "NE": (22.5, 67.5),
        "E": (67.5, 112.5),
        "SE": (112.5, 157.5),
        "S": (157.5, 202.5),
        "SW": (202.5, 247.5),
        "W": (247.5, 292.5),
        "NW": (292.5, 337.5)
    }
    
    for sector, (low, high) in bins.items():
        if sector == "N":
            if wind_from_deg >= low or wind_from_deg < high:
                return sector
        else:
            if low <= wind_from_deg < high:
                return sector
                
    return "Calm"

def main():
    parser = argparse.ArgumentParser(description="Run wind transport and advection analysis")
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
    hotspots_path = paths['hotspots']
    output_json_path = paths['wind_rose']
    
    if not os.path.exists(processed_grid_path):
        print(f"Error: Processed grid file not found at {processed_grid_path}. Run preprocess.py first.")
        sys.exit(1)
        
    if not os.path.exists(hotspots_path):
        print(f"Error: Hotspots JSON file not found at {hotspots_path}. Run hotspot.py first.")
        sys.exit(1)
        
    print(f"Loading processed grid dataset from {processed_grid_path}...")
    with xr.open_dataset(processed_grid_path) as ds_open:
        ds = ds_open.load()
        
    print(f"Loading hotspots data from {hotspots_path}...")
    with open(hotspots_path, 'r') as f:
        hotspots_data = json.load(f)
        
    dates = pd.to_datetime(ds.time.values)
    lats = ds.lat.values
    lons = ds.lon.values
    
    # Use the same time index selected for hotspot detection
    data_date_str = hotspots_data.get('data_date')
    data_date = pd.to_datetime(data_date_str)
    
    t_indices = np.where(dates == data_date)[0]
    if len(t_indices) == 0:
        print(f"Warning: Hotspot date {data_date_str} not found in grid dataset times. Using last step.")
        t_idx = len(dates) - 1
    else:
        t_idx = t_indices[0]
        
    # Extract wind vectors and pollutant values for the whole dataset to compile wind rose
    print("Compiling wind rose summary...")
    u_wind = ds['u_wind'].values
    v_wind = ds['v_wind'].values
    hcho = ds['hcho'].values
    aqi = ds['aqi'].values
    
    # Flatten across time, lat, lon
    u_flat = u_wind.flatten()
    v_flat = v_wind.flatten()
    hcho_flat = hcho.flatten()
    aqi_flat = aqi.flatten()
    
    # Calculate directions and speeds
    speeds = np.sqrt(u_flat**2 + v_flat**2)
    directions = []
    
    for u, v in zip(u_flat, v_flat):
        directions.append(get_wind_direction_bin(u, v))
        
    df_wind = pd.DataFrame({
        'direction': directions,
        'speed': speeds,
        'hcho': hcho_flat,
        'aqi': aqi_flat
    })
    
    # Filter out Calm and NaNs
    df_wind = df_wind[(df_wind['direction'] != "Calm") & np.isfinite(df_wind['hcho']) & np.isfinite(df_wind['aqi'])]
    
    # Group by direction sector
    sectors_summary = []
    total_samples = len(df_wind)
    
    if total_samples > 0:
        grouped = df_wind.groupby('direction')
        for direction_bin in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            if direction_bin in grouped.groups:
                grp = grouped.get_group(direction_bin)
                sectors_summary.append({
                    'direction': direction_bin,
                    'frequency': float(len(grp) / total_samples),
                    'mean_hcho': float(grp['hcho'].mean()),
                    'mean_aqi': float(grp['aqi'].mean()),
                    'mean_speed': float(grp['speed'].mean())
                })
            else:
                sectors_summary.append({
                    'direction': direction_bin,
                    'frequency': 0.0,
                    'mean_hcho': 0.0,
                    'mean_aqi': 0.0,
                    'mean_speed': 0.0
                })
    else:
        print("Warning: No valid wind/pollutant samples found. Creating empty rose.")
        for direction_bin in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            sectors_summary.append({
                'direction': direction_bin,
                'frequency': 0.125,
                'mean_hcho': 0.0,
                'mean_aqi': 0.0,
                'mean_speed': 0.0
            })
            
    # Calculate advection trajectories for top 5 hotspots
    print("Calculating forward and backward trajectories for top hotspots...")
    top_hotspots = hotspots_data.get('hotspots', [])[:5]
    trajectories = []
    
    for h in top_hotspots:
        h_lat = h['lat']
        h_lon = h['lon']
        h_loc = h.get('location', f"Lat {h_lat:.3f}, Lon {h_lon:.3f}")
        
        # Grid index
        lat_idx = np.argmin(np.abs(lats - h_lat))
        lon_idx = np.argmin(np.abs(lons - h_lon))
        
        # Wind components at hotspot on target date
        u = float(ds['u_wind'].values[t_idx, lat_idx, lon_idx])
        v = float(ds['v_wind'].values[t_idx, lat_idx, lon_idx])
        
        # Calculate advection displacements in degrees over 24h
        # 1 degree lat = 111,000 meters
        # 1 degree lon = 111,000 * cos(lat) meters
        dy = v * 86400.0 / 111000.0
        dx = u * 86400.0 / (111000.0 * np.cos(np.radians(h_lat)))
        
        fwd_lat = float(h_lat + dy)
        fwd_lon = float(h_lon + dx)
        bwd_lat = float(h_lat - dy)
        bwd_lon = float(h_lon - dx)
        
        trajectories.append({
            'location': h_loc,
            'lat': float(h_lat),
            'lon': float(h_lon),
            'u': u,
            'v': v,
            'fwd_lat': fwd_lat,
            'fwd_lon': fwd_lon,
            'bwd_lat': bwd_lat,
            'bwd_lon': bwd_lon
        })
        
    output_data = {
        'region': region_name,
        'data_date': data_date_str,
        'wind_rose': sectors_summary,
        'hotspot_trajectories': trajectories
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Successfully saved wind transport analysis to {output_json_path}")

if __name__ == "__main__":
    main()
