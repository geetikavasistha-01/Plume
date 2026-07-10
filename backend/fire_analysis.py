import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
from scipy.stats import pearsonr

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    parser = argparse.ArgumentParser(description="Analyze fire counts and HCHO correlation")
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
    
    fire_csv_path = paths['fire_cache']
    processed_grid_path = paths['processed_grid']
    output_json_path = paths['fire_correlation']
    
    if not os.path.exists(fire_csv_path):
        print(f"Error: Fire cache file not found at {fire_csv_path}. Run fire_ingest.py first.")
        sys.exit(1)
        
    if not os.path.exists(processed_grid_path):
        print(f"Error: Processed grid file not found at {processed_grid_path}. Run preprocess.py first.")
        sys.exit(1)
        
    print(f"Loading processed grid dataset from {processed_grid_path}...")
    with xr.open_dataset(processed_grid_path) as ds_open:
        ds = ds_open.load()
    
    print(f"Loading fire detections from {fire_csv_path}...")
    df_fires = pd.read_csv(fire_csv_path)
    
    dates = pd.to_datetime(ds.time.values)
    lats = ds.lat.values
    lons = ds.lon.values
    
    n_time = len(dates)
    n_lat = len(lats)
    n_lon = len(lons)
    
    # 1. Aggregate fire counts onto the same 0.05° grid
    print("Aggregating fire point detections onto grid...")
    fire_grid = np.zeros((n_time, n_lat, n_lon), dtype=float)
    
    # Parse dates in fire DataFrame
    df_fires['acq_date'] = pd.to_datetime(df_fires['acq_date'])
    
    # Group fires by date and grid cells
    for idx, row in df_fires.iterrows():
        f_date = row['acq_date']
        f_lat = row['latitude']
        f_lon = row['longitude']
        
        # Find time index
        t_indices = np.where(dates == f_date)[0]
        if len(t_indices) == 0:
            continue
        t_idx = t_indices[0]
        
        # Find closest lat/lon index
        lat_idx = np.argmin(np.abs(lats - f_lat))
        lon_idx = np.argmin(np.abs(lons - f_lon))
        
        # Check if coordinates are in bounding box
        dist = np.sqrt((lats[lat_idx] - f_lat)**2 + (lons[lon_idx] - f_lon)**2) * 111.0
        if dist <= 10.0:  # Within reasonable grid cell match
            fire_grid[t_idx, lat_idx, lon_idx] += 1
            
    # Add fire count variable to dataset and save
    ds['fire_count'] = (['time', 'lat', 'lon'], fire_grid)
    # Save back updated dataset
    ds.to_netcdf(processed_grid_path)
    print(f"Added fire_count variable to processed grid at {processed_grid_path}")
    
    # 2. Compute correlation
    hcho_vals = ds['hcho'].values
    
    # Calculate daily time series
    daily_fires = np.sum(fire_grid, axis=(1, 2))
    # Daily spatial mean of HCHO (ignoring NaNs if any)
    daily_mean_hcho = np.nanmean(hcho_vals, axis=(1, 2))
    
    # Compute daily time-series Pearson correlation
    if np.std(daily_fires) > 0 and np.std(daily_mean_hcho) > 0:
        daily_corr, _ = pearsonr(daily_fires, daily_mean_hcho)
    else:
        daily_corr = 0.0
        
    print(f"Daily Time-Series Correlation (Fires vs Mean HCHO): {daily_corr:.4f}")
    
    # Compute global spatial-temporal correlation
    flat_fires = fire_grid.flatten()
    flat_hcho = hcho_vals.flatten()
    valid_mask = np.isfinite(flat_fires) & np.isfinite(flat_hcho)
    
    if np.std(flat_fires[valid_mask]) > 0 and np.std(flat_hcho[valid_mask]) > 0:
        global_corr, _ = pearsonr(flat_fires[valid_mask], flat_hcho[valid_mask])
    else:
        global_corr = 0.0
    print(f"Global Spatial-Temporal Correlation (Fires vs HCHO): {global_corr:.4f}")
    
    # 3. Identify Biomass Burning Periods: Threshold = mean + 2 * std
    mean_daily = np.mean(daily_fires)
    std_daily = np.std(daily_fires)
    threshold = mean_daily + 2.0 * std_daily
    print(f"Biomass Burning Statistical Threshold: {threshold:.2f} fires/day (Mean: {mean_daily:.2f}, Std: {std_daily:.2f})")
    
    burning_dates = []
    for t in range(n_time):
        if daily_fires[t] > threshold:
            burning_dates.append(dates[t])
            
    # Group contiguous dates into periods
    burning_periods = []
    if burning_dates:
        burning_dates.sort()
        start = burning_dates[0]
        prev = burning_dates[0]
        
        for d in burning_dates[1:]:
            if d - prev <= timedelta(days=2):  # Group days with a small gap as part of the same period
                prev = d
            else:
                # Close period
                period_fires = df_fires[(df_fires['acq_date'] >= start) & (df_fires['acq_date'] <= prev)]
                burning_periods.append({
                    'start_date': start.strftime('%Y-%m-%d'),
                    'end_date': prev.strftime('%Y-%m-%d'),
                    'total_fires': int(len(period_fires)),
                    'mean_hcho': float(np.nanmean(hcho_vals[(dates >= start) & (dates <= prev)]))
                })
                start = d
                prev = d
        # Add last period
        period_fires = df_fires[(df_fires['acq_date'] >= start) & (df_fires['acq_date'] <= prev)]
        burning_periods.append({
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': prev.strftime('%Y-%m-%d'),
            'total_fires': int(len(period_fires)),
            'mean_hcho': float(np.nanmean(hcho_vals[(dates >= start) & (dates <= prev)]))
        })
        
    print(f"Identified {len(burning_periods)} distinct Biomass Burning Periods.")
    
    # 4. Compute Grid Cell Correlations (only for cells with sufficient fires)
    grid_correlations = []
    for i in range(n_lat):
        for j in range(n_lon):
            cell_fires = fire_grid[:, i, j]
            cell_hcho = hcho_vals[:, i, j]
            if np.sum(cell_fires) > 5 and np.std(cell_fires) > 0 and np.std(cell_hcho) > 0:
                corr, _ = pearsonr(cell_fires, cell_hcho)
                if not np.isnan(corr):
                    grid_correlations.append({
                        'lat': float(lats[i]),
                        'lon': float(lons[j]),
                        'correlation': float(corr)
                    })
                    
    # Format time series data for JSON
    time_series_data = []
    for t in range(n_time):
        time_series_data.append({
            'date': dates[t].strftime('%Y-%m-%d'),
            'fire_count': int(daily_fires[t]),
            'mean_hcho': float(daily_mean_hcho[t])
        })
        
    # Export results
    output_data = {
        'region': region_name,
        'global_correlation': float(global_corr),
        'daily_correlation': float(daily_corr),
        'statistical_threshold': float(threshold),
        'mean_daily_fires': float(mean_daily),
        'std_daily_fires': float(std_daily),
        'burning_periods': burning_periods,
        'time_series': time_series_data,
        'grid_correlations': grid_correlations
    }
    
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
        
    print(f"Saved correlation analysis to {output_json_path}")

if __name__ == "__main__":
    main()
