import os
import sys
import argparse
import numpy as np
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def main():
    parser = argparse.ArgumentParser(description="Run model predictions for a region")
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
    
    X_path = paths['X']
    model_path = os.path.join(config.MODELS_DIR, "cnn_lstm_aqi.keras")
    pred_path = paths['predictions']
    
    if not os.path.exists(X_path):
        print(f"Error: X feature array not found at {X_path}. Run preprocess.py first.")
        sys.exit(1)
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Train the model first.")
        sys.exit(1)
        
    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print(f"Loading X features from {X_path}...")
    X = np.load(X_path)
    
    print(f"Running inference on {len(X)} samples...")
    # Predict in batches to optimize performance
    preds = model.predict(X, batch_size=4096, verbose=1).flatten()
    
    # Save predictions
    os.makedirs(os.path.dirname(pred_path), exist_ok=True)
    np.save(pred_path, preds)
    print(f"Successfully saved model predictions to {pred_path}")

if __name__ == "__main__":
    main()
