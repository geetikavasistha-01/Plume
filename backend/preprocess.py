import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import joblib

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def calculate_proxy_aqi(ds):
    """Computes a proxy Air Quality Index (AQI) based on pollutants."""
    i_no2 = np.clip(ds['no2'].values / 0.0003, 0, 1.0) * 150.0
    i_hcho = np.clip(ds['hcho'].values / 0.0004, 0, 1.0) * 100.0
    i_aod = np.clip((ds['aod'].values + 0.5) / 3.5, 0, 1.0) * 200.0
    
    proxy_aqi = np.maximum(i_no2, i_aod) + 0.15 * i_hcho
    proxy_aqi = np.clip(proxy_aqi, 10.0, 500.0)
    
    return proxy_aqi

def build_sequences(data, target, is_ground, lookback):
    """Creates sliding window sequences from gridded data."""
    n_time, n_lat, n_lon, n_feats = data.shape
    X_list = []
    y_list = []
    is_ground_list = []
    
    for i in range(n_lat):
        for j in range(n_lon):
            cell_data = data[:, i, j, :]
            cell_target = target[:, i, j]
            cell_is_ground = is_ground[:, i, j]
            
            for t in range(lookback, n_time):
                X_list.append(cell_data[t-lookback:t])
                y_list.append(cell_target[t])
                is_ground_list.append(cell_is_ground[t])
                
    return np.array(X_list), np.array(y_list), np.array(is_ground_list)

def main():
    parser = argparse.ArgumentParser(description="Preprocess grid data")
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
    cache_file = paths['raw_grid']
    
    if not os.path.exists(cache_file):
        print(f"Error: Cache file not found at {cache_file} for region '{region_name}'. Ingest data first.")
        sys.exit(1)
        
    print(f"Loading raw grid cache for region '{region_name}' from {cache_file}...")
    ds = xr.open_dataset(cache_file)
    
    # 1. Compute Proxy AQI
    print("Calculating proxy AQI target...")
    proxy_aqi = calculate_proxy_aqi(ds)
    ds['proxy_aqi'] = (['time', 'lat', 'lon'], proxy_aqi)
    # Maintain backwards compatibility for other code referencing 'aqi'
    ds['aqi'] = (['time', 'lat', 'lon'], proxy_aqi)
    
    # 2. Integrate CPCB ground stations if cache exists
    cpcb_cache_path = os.path.join(config.CACHE_DIR, f"cpcb_{region_slug}.csv")
    ground_aqi = np.full(ds['aqi'].shape, np.nan)
    is_ground_grid = np.zeros(ds['aqi'].shape, dtype=bool)
    
    if os.path.exists(cpcb_cache_path):
        print(f"Loading CPCB station data from {cpcb_cache_path}...")
        df_cpcb = pd.read_csv(cpcb_cache_path)
        
        # Spatial matching: find stations within 10 km (~0.09 degrees)
        stations = df_cpcb[['station_name', 'latitude', 'longitude']].drop_duplicates().to_dict(orient='records')
        station_matches = []
        lats = ds.lat.values
        lons = ds.lon.values
        
        for s in stations:
            lat_idx = np.argmin(np.abs(lats - s['latitude']))
            lon_idx = np.argmin(np.abs(lons - s['longitude']))
            # Approx distance in km: 1 degree latitude = 111 km
            # Simple Euclidean distance in degrees scaled to km is fine for 10km threshold
            dist = np.sqrt((lats[lat_idx] - s['latitude'])**2 + (lons[lon_idx] - s['longitude'])**2) * 111.0
            
            if dist <= 10.0:
                station_matches.append({
                    'name': s['station_name'],
                    'lat_idx': lat_idx,
                    'lon_idx': lon_idx
                })
                print(f"CPCB Station Mapping: '{s['station_name']}' mapped to grid cell idx ({lat_idx}, {lon_idx}) at distance {dist:.2f} km")
            else:
                print(f"CPCB Station Mapping: '{s['station_name']}' is outside 10 km radius (distance {dist:.2f} km). Skipping.")
                
        # Populate ground_aqi array
        for match in station_matches:
            name = match['name']
            lat_idx = match['lat_idx']
            lon_idx = match['lon_idx']
            
            df_station = df_cpcb[df_cpcb['station_name'] == name].copy()
            df_station['date'] = pd.to_datetime(df_station['date'])
            df_station = df_station.set_index('date')
            
            for t, time_val in enumerate(pd.to_datetime(ds.time.values)):
                if time_val in df_station.index:
                    val = df_station.loc[time_val, 'ground_aqi']
                    if not np.isnan(val):
                        if np.isnan(ground_aqi[t, lat_idx, lon_idx]):
                            ground_aqi[t, lat_idx, lon_idx] = val
                        else:
                            # Average readings if multiple stations map to same grid cell
                            ground_aqi[t, lat_idx, lon_idx] = 0.5 * (ground_aqi[t, lat_idx, lon_idx] + val)
                        is_ground_grid[t, lat_idx, lon_idx] = True
                        
        print(f"Mapped CPCB stations to {is_ground_grid.sum()} grid-day combinations.")
    else:
        print(f"Warning: CPCB ground truth cache not found at {cpcb_cache_path}. Proceeding with proxy-only ground data.")
        
    ds['ground_aqi'] = (['time', 'lat', 'lon'], ground_aqi)
    ds['is_ground'] = (['time', 'lat', 'lon'], is_ground_grid)
    
    # 3. Construct Consolidated Training Target
    target_aqi = np.where(is_ground_grid, ground_aqi, proxy_aqi)
    
    # Define features (with proxy_aqi included as a feature)
    feature_names = ['hcho', 'no2', 'aod', 'temp', 'u_wind', 'v_wind', 'dewpoint', 'precip', 'proxy_aqi']
    
    # Extract features as numpy stack
    feat_arrays = []
    for f in feature_names:
        feat_arrays.append(ds[f].values)
    features_stack = np.stack(feat_arrays, axis=-1)  # (time, lat, lon, 9)
    
    # Normalize features
    flat_features = features_stack.reshape(-1, len(feature_names))
    
    # Paths for scaling
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    scaler_path_delhi = os.path.join(config.PROCESSED_DIR, "scaler_delhi_ncr.joblib")
    scaler_path_legacy = os.path.join(config.PROCESSED_DIR, "scaler.joblib")
    
    # Check if we should fit or load the Delhi-NCR scaling
    if region_slug == "delhi_ncr":
        print("Training on Delhi-NCR: Fitting new scaler parameters...")
        means = np.nanmean(flat_features, axis=0)
        stds = np.nanstd(flat_features, axis=0)
        stds[stds == 0.0] = 1.0
        
        scalers = {
            'feature_names': feature_names,
            'means': means.tolist(),
            'stds': stds.tolist()
        }
        # Save to both file names for compatibility
        joblib.dump(scalers, scaler_path_delhi)
        joblib.dump(scalers, scaler_path_legacy)
        print(f"Saved Delhi-NCR scaler to {scaler_path_delhi}")
    else:
        # Load Delhi-NCR scaler parameters if they exist
        if os.path.exists(scaler_path_delhi):
            print(f"Applying pre-trained Delhi-NCR scaling factors to region '{region_name}'...")
            scalers = joblib.load(scaler_path_delhi)
            # Ensure loaded scaler matches our feature length (incase of legacy runs)
            if len(scalers['means']) == len(feature_names):
                means = np.array(scalers['means'])
                stds = np.array(scalers['stds'])
            else:
                print("Warning: Pre-trained scaler features length mismatch. Fitting local parameters...")
                means = np.nanmean(flat_features, axis=0)
                stds = np.nanstd(flat_features, axis=0)
                stds[stds == 0.0] = 1.0
        else:
            print(f"Warning: Delhi-NCR scaler not found at {scaler_path_delhi}. Fitting local parameters on the fly...")
            means = np.nanmean(flat_features, axis=0)
            stds = np.nanstd(flat_features, axis=0)
            stds[stds == 0.0] = 1.0
            
    # Normalize stack
    norm_flat_features = (flat_features - means) / stds
    norm_features_stack = norm_flat_features.reshape(features_stack.shape)
    
    # 4. Build training/inference sequences
    print(f"Building sequences with lookback of {config.LOOKBACK_DAYS} days...")
    X, y, is_ground_seq = build_sequences(norm_features_stack, target_aqi, is_ground_grid, config.LOOKBACK_DAYS)
    print(f"Sequences shape: X={X.shape}, y={y.shape}, is_ground={is_ground_seq.shape}")
    
    # Save processed tensors
    np.save(paths['X'], X)
    np.save(paths['y'], y)
    np.save(paths['is_ground'], is_ground_seq)
    print(f"Saved processed sequence arrays to {paths['X']}, {paths['y']} and {paths['is_ground']}")
    
    # For Delhi-NCR (default models/training), save to direct files
    if region_slug == "delhi_ncr":
        X_path_direct = os.path.join(config.PROCESSED_DIR, "X.npy")
        y_path_direct = os.path.join(config.PROCESSED_DIR, "y.npy")
        is_ground_path_direct = os.path.join(config.PROCESSED_DIR, "is_ground.npy")
        np.save(X_path_direct, X)
        np.save(y_path_direct, y)
        np.save(is_ground_path_direct, is_ground_seq)
        print(f"Saved default Delhi-NCR sequence arrays for training.")
        
    # Save processed xarray grid dataset
    ds.to_netcdf(paths['processed_grid'])
    print(f"Saved processed dataset grid to {paths['processed_grid']}")
    print(f"Preprocessing completed successfully for region '{region_name}'.")


if __name__ == "__main__":
    main()
