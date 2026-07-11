import os
from datetime import datetime, timedelta

# Project directories
BASE_DIR = "/Users/geetikavasistha/isro_aqi_hcho"
DATA_DIR = os.path.join(BASE_DIR, "data")
CACHE_DIR = os.path.join(DATA_DIR, "cache")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Bounding box for Delhi-NCR (Default)
BBOX = {
    'min_lat': 27.0,
    'max_lat': 29.5,
    'min_lon': 76.5,
    'max_lon': 78.5
}

# Date range: past 60 days to today
END_DATE = datetime(2026, 7, 10)  # Using local current date 2026-07-10
START_DATE = END_DATE - timedelta(days=60)

START_DATE_STR = START_DATE.strftime('%Y-%m-%d')
END_DATE_STR = END_DATE.strftime('%Y-%m-%d')

# Google Earth Engine Project ID
GEE_PROJECT_ID = "project-3cb1a433-8a2a-42c4-9bc"

# Gridding resolution (degrees)
RESOLUTION = 0.05  # ~5 km resolution (default for cities)
STATE_RESOLUTION = 0.1  # ~10 km resolution (for states)
RESOLUTION_THRESHOLD = 2.0  # size span threshold (degrees) to classify as state

def get_resolution(bbox):
    """Returns dynamic resolution based on bounding box size (span)."""
    lat_diff = bbox['max_lat'] - bbox['min_lat']
    lon_diff = bbox['max_lon'] - bbox['min_lon']
    if lat_diff > RESOLUTION_THRESHOLD or lon_diff > RESOLUTION_THRESHOLD:
        return STATE_RESOLUTION
    return RESOLUTION

# ML Model training parameters
LOOKBACK_DAYS = 7
EPOCHS = 15
BATCH_SIZE = 32
VAL_SPLIT = 0.2

# Preset Regions
PRESET_REGIONS = {
    "Delhi-NCR": {"lat_min": 27.0, "lat_max": 29.5, "lon_min": 76.5, "lon_max": 78.5},
    "Mumbai": {"lat_min": 18.8, "lat_max": 19.3, "lon_min": 72.7, "lon_max": 73.1},
    "Bengaluru": {"lat_min": 12.7, "lat_max": 13.2, "lon_min": 77.4, "lon_max": 77.8},
    "Kolkata": {"lat_min": 22.4, "lat_max": 22.7, "lon_min": 88.2, "lon_max": 88.5},
    "Chennai": {"lat_min": 12.8, "lat_max": 13.2, "lon_min": 80.1, "lon_max": 80.3},
    "Punjab (state, wide)": {"lat_min": 29.5, "lat_max": 32.5, "lon_min": 73.9, "lon_max": 76.9},
}

def get_region_slug(region_name):
    """Generates a clean slug for a region name."""
    return region_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("-", "_")

def get_paths(region_slug):
    """Returns region-specific cache, processed, and outputs paths."""
    return {
        'raw_grid': os.path.join(CACHE_DIR, f"raw_grid_{region_slug}.nc"),
        'processed_grid': os.path.join(PROCESSED_DIR, f"processed_grid_{region_slug}.nc"),
        'X': os.path.join(PROCESSED_DIR, f"X_{region_slug}.npy"),
        'y': os.path.join(PROCESSED_DIR, f"y_{region_slug}.npy"),
        'is_ground': os.path.join(PROCESSED_DIR, f"is_ground_{region_slug}.npy"),
        'hotspots': os.path.join(PROCESSED_DIR, f"hotspots_{region_slug}.json"),
        'fire_cache': os.path.join(CACHE_DIR, f"fire_{region_slug}.csv"),
        'fire_correlation': os.path.join(PROCESSED_DIR, f"fire_hcho_correlation_{region_slug}.json"),
        'wind_rose': os.path.join(PROCESSED_DIR, f"wind_rose_{region_slug}.json"),
        'predictions': os.path.join(PROCESSED_DIR, f"predictions_{region_slug}.npy")
    }
