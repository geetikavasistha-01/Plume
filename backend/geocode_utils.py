import reverse_geocoder as rg

def geocode_coordinates(coords):
    """
    Reverse-geocodes a list of (latitude, longitude) coordinate pairs.
    Returns a list of location strings (e.g. 'Panipat, Haryana (IN)').
    """
    if not coords:
        return []
    try:
        # rg.search expects a list of (lat, lon) tuples
        results = rg.search(coords, verbose=False)
        labels = []
        for r in results:
            name = r.get('name', '')
            district = r.get('admin2', '')
            state = r.get('admin1', '')
            
            if district and district != name:
                label = f"{name}, {district} ({state})"
            else:
                label = f"{name}, {state}"
            labels.append(label)
        return labels
    except Exception as e:
        # Fallback to coordinate string representation on exception
        return [f"Lat {lat:.3f}, Lon {lon:.3f}" for lat, lon in coords]
