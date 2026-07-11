import os
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend import config

PRECOMPUTE_REGIONS = [
    # Presets
    "Delhi-NCR", "Mumbai", "Bengaluru", "Kolkata", "Chennai", "Punjab (state, wide)",
    # States
    "Maharashtra", "Karnataka", "Tamil Nadu", "Uttar Pradesh", "Gujarat", 
    "West Bengal", "Rajasthan", "Madhya Pradesh", "Andhra Pradesh", "Bihar", 
    "Haryana", "Kerala", "Assam", "Odisha",
    # Cities
    "Nagpur", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Ahmedabad", "Patna"
]

LOG_FILE = os.path.join(config.CACHE_DIR, "precompute_log.json")

def load_log():
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_run": None, "regions": {}}

def save_log(log_data):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'w') as f:
        json.dump(log_data, f, indent=4)

def check_cache_freshness(region_name, max_age_hours=24):
    region_slug = config.get_region_slug(region_name)
    paths = config.get_paths(region_slug)
    
    # Check key processed files
    critical_files = [paths['processed_grid'], paths['hotspots'], paths['predictions']]
    for f in critical_files:
        if not os.path.exists(f):
            return False
        # Check file age
        file_age_seconds = time.time() - os.path.getmtime(f)
        if file_age_seconds > max_age_hours * 3600:
            return False
    return True

def run_pipeline_for_region(region_name):
    print(f"\n==================================================")
    print(f"Starting pipeline precomputation for '{region_name}'")
    print(f"==================================================")
    
    python_path = sys.executable
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Resolve bounds if it is not a preset
    extra_args = []
    if region_name not in config.PRESET_REGIONS:
        from backend.region_resolver import resolve_region_bounds
        try:
            bbox = resolve_region_bounds(region_name)
            extra_args = [
                "--lat-min", str(bbox['min_lat']),
                "--lat-max", str(bbox['max_lat']),
                "--lon-min", str(bbox['min_lon']),
                "--lon-max", str(bbox['max_lon'])
            ]
        except Exception as e:
            print(f"Failed to resolve bounds for '{region_name}': {e}")
            return False, f"Geocoding error: {e}"
            
    # Commands list
    commands = [
        [python_path, os.path.join(backend_dir, "data_ingest.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "cpcb_ingest.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "preprocess.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "inference.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "hotspot.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "fire_ingest.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "fire_analysis.py"), "--region", region_name] + extra_args,
        [python_path, os.path.join(backend_dir, "transport_analysis.py"), "--region", region_name] + extra_args,
    ]
    
    for idx, cmd in enumerate(commands, 1):
        print(f"\n[Step {idx}/8] Running: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"Step {idx} failed!")
            print(f"Stdout:\n{res.stdout}")
            print(f"Stderr:\n{res.stderr}")
            return False, f"Step {idx} failed: {res.stderr.strip()}"
            
    print(f"Successfully processed region '{region_name}'.")
    return True, None

def run_precompute(force=False, max_age_hours=24):
    log_data = load_log()
    log_data["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    for region in PRECOMPUTE_REGIONS:
        # Check freshness
        if not force and check_cache_freshness(region, max_age_hours):
            print(f"Skipping '{region}': cache is fresh (less than {max_age_hours}h old)")
            log_data["regions"][region] = {
                "status": "skipped",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "reason": f"Cache is fresh (less than {max_age_hours}h old)"
            }
            save_log(log_data)
            continue
            
        start_time = time.time()
        success, err = run_pipeline_for_region(region)
        duration = time.time() - start_time
        
        if success:
            log_data["regions"][region] = {
                "status": "success",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 2),
                "error": None
            }
        else:
            log_data["regions"][region] = {
                "status": "failed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": round(duration, 2),
                "error": err
            }
        save_log(log_data)

def main():
    parser = argparse.ArgumentParser(description="ISRO AQI HCHO Precomputation Cache Scheduler")
    parser.add_argument("--force", action="store_true", help="Force precomputation even if cache is fresh")
    parser.add_argument("--age", type=int, default=24, help="Max cache age in hours (default: 24)")
    parser.add_argument("--loop", action="store_true", help="Run as a daemon loop")
    parser.add_argument("--interval", type=int, default=86400, help="Daemon loop interval in seconds (default: 86400 / 24h)")
    args = parser.parse_args()
    
    if args.loop:
        print(f"Starting precompute scheduler in daemon loop mode (interval: {args.interval}s)...")
        while True:
            try:
                run_precompute(args.force, args.age)
            except Exception as e:
                print(f"Error during precompute run loop: {e}")
            print(f"Sleeping for {args.interval} seconds...")
            time.sleep(args.interval)
    else:
        run_precompute(args.force, args.age)

if __name__ == "__main__":
    main()
