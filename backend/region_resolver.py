import sys
import time
import argparse
from geopy.geocoders import Nominatim

def resolve_region_bounds(place_name: str) -> dict:
    """
    Resolves the bounding box for any Indian city, district, or state.
    Returns:
        dict: {
            'min_lat': float,
            'max_lat': float,
            'min_lon': float,
            'max_lon': float
        }
    """
    if not place_name:
        raise ValueError("Place name cannot be empty.")
        
    query = place_name.strip()
    if "india" not in query.lower():
        query = f"{query}, India"
        
    print(f"Resolving location: '{query}' via Nominatim...")
    
    # Initialize geolocator with a descriptive user agent
    geolocator = Nominatim(user_agent="isro_aqi_hcho_region_resolver")
    
    location = None
    last_error = None
    
    for attempt in range(3):
        try:
            # Query Nominatim restricted to country code 'in' (India)
            location = geolocator.geocode(query, country_codes='in', timeout=10)
            if location:
                break
        except Exception as e:
            last_error = e
            print(f"Geocoding attempt {attempt + 1} failed: {e}. Retrying in 1s...")
            time.sleep(1)
            
    if not location:
        error_msg = f"Could not resolve location '{place_name}' in India."
        if last_error:
            error_msg += f" (Last error: {last_error})"
        raise ValueError(error_msg)
        
    raw = location.raw
    bbox_raw = raw.get('boundingbox')
    
    if not bbox_raw or len(bbox_raw) != 4:
        raise ValueError(f"No bounding box coordinates returned for '{place_name}'.")
        
    # Nominatim returns boundingbox as [south_lat, north_lat, west_lon, east_lon] (strings)
    lat_min = float(bbox_raw[0])
    lat_max = float(bbox_raw[1])
    lon_min = float(bbox_raw[2])
    lon_max = float(bbox_raw[3])
    
    lat_diff = lat_max - lat_min
    lon_diff = lon_max - lon_min
    
    # 50km buffer is approximately 0.45 degrees total (0.225 degrees in each direction)
    BUFFER = 0.225
    
    # If the returned bounding box is a point or extremely small (e.g. < 0.05 degrees),
    # buffer it to form a standard city study grid of ~50km x 50km
    if lat_diff < 0.05 or lon_diff < 0.05:
        center_lat = (lat_min + lat_max) / 2.0
        center_lon = (lon_min + lon_max) / 2.0
        lat_min = center_lat - BUFFER
        lat_max = center_lat + BUFFER
        lon_min = center_lon - BUFFER
        lon_max = center_lon + BUFFER
        print(f"Location resolved as a point/small area. Applied 50km buffer around centroid ({center_lat:.4f}, {center_lon:.4f}).")
    else:
        print(f"Location resolved as administrative region boundaries.")
        
    # Ensure coordinates are within standard bounds
    lat_min = max(-90.0, min(90.0, lat_min))
    lat_max = max(-90.0, min(90.0, lat_max))
    lon_min = max(-180.0, min(180.0, lon_min))
    lon_max = max(-180.0, min(180.0, lon_max))
    
    resolved_bbox = {
        'min_lat': round(lat_min, 4),
        'max_lat': round(lat_max, 4),
        'min_lon': round(lon_min, 4),
        'max_lon': round(lon_max, 4)
    }
    
    print(f"Resolved bounds for '{place_name}':")
    print(f"  Lat: {resolved_bbox['min_lat']} to {resolved_bbox['max_lat']}")
    print(f"  Lon: {resolved_bbox['min_lon']} to {resolved_bbox['max_lon']}")
    
    return resolved_bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test query Indian region boundary resolution")
    parser.add_argument("place", type=str, help="Name of the city, district, or state in India")
    args = parser.parse_args()
    
    try:
        resolve_region_bounds(args.place)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
