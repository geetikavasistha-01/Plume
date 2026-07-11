import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.model_selection import train_test_split

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def build_model(input_shape):
    """Creates a CNN-1D + LSTM model."""
    model = models.Sequential([
        # 1D Convolutional Layer for temporal feature extraction
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', 
                      padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Dropout(0.1),
        
        # LSTM Layer to capture sequential/temporal dependencies
        layers.LSTM(units=32, return_sequences=False),
        layers.Dropout(0.1),
        
        # Dense Layer
        layers.Dense(units=16, activation='relu'),
        # Output Layer predicting AQI next day
        layers.Dense(units=1)
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model

def main():
    import argparse
    import glob
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=str, default="all", help="Comma-separated list of regions to train on, or 'all'")
    args = parser.parse_args()
    
    processed_dir = config.PROCESSED_DIR
    
    if args.regions.lower() == "all":
        # Load all regional X_*.npy files (X.npy is excluded because it does not have the underscore)
        x_files = sorted(glob.glob(os.path.join(processed_dir, "X_*.npy")))
    else:
        # Load specific comma-separated regions
        regions_list = [r.strip() for r in args.regions.split(",")]
        x_files = []
        for r in regions_list:
            slug = config.get_region_slug(r)
            f_path = os.path.join(processed_dir, f"X_{slug}.npy")
            if os.path.exists(f_path):
                x_files.append(f_path)
            else:
                print(f"Warning: Dataset for region '{r}' (slug: {slug}) not found at {f_path}. Skipping.")
                
    if not x_files:
        print("Error: No regional training datasets found!")
        sys.exit(1)
        
    print(f"Found {len(x_files)} regional datasets to combine:")
    X_list = []
    y_list = []
    is_ground_list = []
    
    for f in x_files:
        basename = os.path.basename(f)
        region_slug = basename.replace("X_", "").replace(".npy", "")
        y_file = os.path.join(processed_dir, f"y_{region_slug}.npy")
        is_ground_file = os.path.join(processed_dir, f"is_ground_{region_slug}.npy")
        
        if not os.path.exists(y_file):
            print(f"Warning: Target file {y_file} missing for {region_slug}. Skipping.")
            continue
            
        print(f" - Loading {region_slug}...")
        X_reg = np.load(f)
        y_reg = np.load(y_file)
        
        if os.path.exists(is_ground_file):
            is_ground_reg = np.load(is_ground_file)
        else:
            is_ground_reg = np.zeros_like(y_reg, dtype=bool)
            
        X_list.append(X_reg)
        y_list.append(y_reg)
        is_ground_list.append(is_ground_reg)
        
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    is_ground = np.concatenate(is_ground_list, axis=0)
    
    # Save the consolidated training files as the default dataset
    X_default_path = os.path.join(processed_dir, "X.npy")
    y_default_path = os.path.join(processed_dir, "y.npy")
    is_ground_default_path = os.path.join(processed_dir, "is_ground.npy")
    np.save(X_default_path, X)
    np.save(y_default_path, y)
    np.save(is_ground_default_path, is_ground)
    
    print(f"\nConsolidated dataset shape: X={X.shape}, y={y.shape}, is_ground={is_ground.shape}")
    
    # Split into Train and Validation
    X_train, X_val, y_train, y_val, is_ground_train, is_ground_val = train_test_split(
        X, y, is_ground, test_size=config.VAL_SPLIT, random_state=42
    )
    
    print(f"Training on {X_train.shape[0]} samples, validating on {X_val.shape[0]} samples.")
    
    # Build Model
    input_shape = (X.shape[1], X.shape[2])  # (lookback, features)
    model = build_model(input_shape)
    model.summary()
    
    # Callbacks
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(config.MODELS_DIR, "cnn_lstm_aqi.keras")
    
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train
    print(f"Starting model training for {config.EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate
    print("Evaluating model...")
    # First, evaluate on the full validation set (mainly proxy targets)
    all_val_loss, all_val_mae, all_val_rmse = model.evaluate(X_val, y_val, verbose=0)
    all_predictions = model.predict(X_val, batch_size=1024, verbose=0).flatten()
    all_correlation = float(np.corrcoef(all_predictions, y_val)[0, 1])
    print(f"All Validation (Proxy-dominated) Metrics -> MAE: {all_val_mae:.4f}, RMSE: {all_val_rmse:.4f}, R: {all_correlation:.4f}")
    
    # Now, evaluate specifically on ground-truth CPCB monitor samples
    has_ground = (is_ground_val == True)
    if np.any(has_ground):
        print(f"Found {np.sum(has_ground)} ground-truth CPCB validation samples. Computing final metrics against them...")
        X_val_ground = X_val[has_ground]
        y_val_ground = y_val[has_ground]
        val_loss, val_mae, val_rmse = model.evaluate(X_val_ground, y_val_ground, verbose=0)
        predictions = model.predict(X_val_ground, batch_size=1024, verbose=0).flatten()
        correlation = float(np.corrcoef(predictions, y_val_ground)[0, 1])
        print(f"CPCB Ground Validation Metrics -> MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R: {correlation:.4f}")
    else:
        print("Warning: No CPCB ground-truth validation samples found. Falling back to proxy metrics.")
        val_loss, val_mae, val_rmse = all_val_loss, all_val_mae, all_val_rmse
        correlation = all_correlation
        
    # Save Model
    model.save(model_path)
    print(f"Saved model file to {model_path}")
    
    # Prepare metrics log
    metrics = {
        'val_loss': float(val_loss),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'correlation': correlation,
        'proxy_val_loss': float(all_val_loss),
        'proxy_val_mae': float(all_val_mae),
        'proxy_val_rmse': float(all_val_rmse),
        'proxy_correlation': all_correlation,
        'history': {
            'loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']],
            'rmse': [float(x) for x in history.history['rmse']],
            'val_rmse': [float(x) for x in history.history['val_rmse']],
        }
    }
    
    metrics_path = os.path.join(config.MODELS_DIR, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics report to {metrics_path}")
    
if __name__ == "__main__":
    main()
