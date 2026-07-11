import os
import xarray as xr
import pandas as pd
import sqlite3

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nc_path = os.path.join(base_dir, "data", "processed", "processed_grid_delhi_ncr.nc")
    db_dir = os.path.join(base_dir, "database")
    db_path = os.path.join(db_dir, "plume_training_data.db")
    csv_path = os.path.join(db_dir, "plume_training_data.csv")

    if not os.path.exists(nc_path):
        print(f"Error: Delhi-NCR processed grid not found at {nc_path}")
        return

    print("Loading Delhi-NCR processed grid...")
    ds = xr.open_dataset(nc_path)
    
    print("Converting to pandas DataFrame...")
    df = ds.to_dataframe()
    
    # Reset index to make coordinates (lat, lon, time) normal columns
    df = df.reset_index()
    
    # Convert datetime columns to string for SQLite compatibility if needed
    if 'time' in df.columns:
        df['time'] = df['time'].dt.strftime('%Y-%m-%d')
    
    # Ensure database folder exists
    os.makedirs(db_dir, exist_ok=True)
    
    print(f"Saving to SQLite database table 'training_data' at {db_path}...")
    conn = sqlite3.connect(db_path)
    df.to_sql("training_data", conn, if_exists="replace", index=False)
    conn.close()
    
    print(f"Saving to CSV file at {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    print("Database and CSV creation complete!")

if __name__ == "__main__":
    main()
