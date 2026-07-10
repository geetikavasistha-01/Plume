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
    X_path = os.path.join(config.PROCESSED_DIR, "X.npy")
    y_path = os.path.join(config.PROCESSED_DIR, "y.npy")
    
    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        print(f"Error: Processed data not found at {config.PROCESSED_DIR}. Run preprocess.py first.")
        sys.exit(1)
        
    print("Loading preprocessed tensors...")
    X = np.load(X_path)
    y = np.load(y_path)
    
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    
    # Split into Train and Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.VAL_SPLIT, random_state=42
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
    val_loss, val_mae, val_rmse = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation Metrics -> Loss: {val_loss:.4f}, MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
    
    # Generate predictions to compute correlation and metrics
    predictions = model.predict(X_val, batch_size=1024, verbose=0).flatten()
    correlation = float(np.corrcoef(predictions, y_val)[0, 1])
    print(f"Predicted vs Actual Correlation: {correlation:.4f}")
    
    # Save Model
    model.save(model_path)
    print(f"Saved model file to {model_path}")
    
    # Prepare metrics log
    metrics = {
        'val_loss': float(val_loss),
        'val_mae': float(val_mae),
        'val_rmse': float(val_rmse),
        'correlation': correlation,
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
