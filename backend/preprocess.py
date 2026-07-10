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

def build_sequences(data, target, lookback):
    """Creates sliding window sequences from gridded data."""
    n_time, n_lat, n_lon, n_feats = data.shape
    X_list = []
    y_list = []
    
    for i in range(n_lat):
        for j in range(n_lon):
            cell_data = data[:, i, j, :]
            cell_target = target[:, i, j]
            
            for t in range(lookback, n_time):
                X_list.append(cell_data[t-lookback:t])
                y_list.append(cell_target[t])
                
    return np.array(X_list), np.array(y_list)

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
    aqi = calculate_proxy_aqi(ds)
    ds['aqi'] = (['time', 'lat', 'lon'], aqi)
    
    # Define features
    feature_names = ['hcho', 'no2', 'aod', 'temp', 'u_wind', 'v_wind', 'dewpoint', 'precip']
    
    # 2. Extract features as numpy stack
    feat_arrays = []
    for f in feature_names:
        feat_arrays.append(ds[f].values)
    features_stack = np.stack(feat_arrays, axis=-1)  # (time, lat, lon, 8)
    
    # 3. Normalize features
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
            means = np.array(scalers['means'])
            stds = np.array(scalers['stds'])
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
    X, y = build_sequences(norm_features_stack, aqi, config.LOOKBACK_DAYS)
    print(f"Sequences shape: X={X.shape}, y={y.shape}")
    
    # Save processed tensors
    np.save(paths['X'], X)
    np.save(paths['y'], y)
    print(f"Saved processed sequence arrays to {paths['X']} and {paths['y']}")
    
    # Save processed xarray grid dataset
    ds.to_netcdf(paths['processed_grid'])
    print(f"Saved processed dataset grid to {paths['processed_grid']}")
    print(f"Preprocessing completed successfully for region '{region_name}'.")

if __name__ == "__main__":
    main()
