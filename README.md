#  Air Quality Index (AQI) & Formaldehyde (HCHO) Prediction Pipeline

A full-stack machine learning project built for predicting surface Air Quality Index (AQI) over India and detecting Formaldehyde (HCHO) hotspots using Sentinel-5P satellite observations and ERA5-Land reanalysis products. Serving a premium, editorial-dark Streamlit dashboard.

---

## 🗺️ Project Architecture

```
isro_aqi_hcho/
├── backend/
│   ├── config.py           # Bounding box, date range, file paths, and GEE project ID
│   ├── data_ingest.py      # Earth Engine data downloader (Sentinel-5P + ERA5 Land)
│   ├── preprocess.py       # Gridding, Proxy AQI calculation, and Sequence builder
│   ├── model.py            # CNN-1D + LSTM next-day AQI forecast network (Keras)
│   └── hotspot.py          # Isolation Forest spatial HCHO hotspot detector (scikit-learn)
├── data/
│   ├── cache/              # Raw ingested NetCDF dataset
│   └── processed/          # Normalized sequences, scale factors, and hotspots JSON
├── models/
│   ├── cnn_lstm_aqi.keras  # Trained convolutional-recurrent model
│   └── metrics.json        # Training history & validation MAE/RMSE metrics
├── dashboard/
│   ├── app.py              # Main app entry point (Overview/Home)
│   └── pages/
│       ├── 1_AQI_Forecast_Map.py   # Spatiotemporal predicted/actual AQI mapping
│       ├── 2_HCHO_Hotspots.py       # Isolation Forest hotspot visualization
│       └── 3_Methodology.py         # Technical implementation and math formulation
├── requirements.txt        # Python dependency manifest
└── README.md               # Setup & Execution guidelines
```

---

## 🛠️ Installation & Setup

All required packages are pre-installed in the virtual environment `/Users/geetikavasistha/isro_env`.

### 1. Active virtual environment
Verify your shell is referencing the environment:
```bash
source /Users/geetikavasistha/isro_env/bin/activate
```

### 2. Verify GEE configuration
The environment has persistent credentials configured at `~/.config/earthengine/credentials` and has successfully initialized with project ID `project-3cb1a433-8a2a-42c4-9bc`.

---

## 🚀 Execution Guide

Run the pipeline steps sequentially:

### Step 1: Ingest Data
Downloads daily satellite observations (HCHO, NO2, UV Aerosol Index) and ERA5 land variables (temp, winds, dewpoint, precipitation) for the Delhi-NCR bounding box:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/data_ingest.py
```

### Step 2: Preprocess Features
Standardizes all variables, computes the custom **Proxy AQI** target, and builds sliding sequence lookbacks:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/preprocess.py
```

### Step 3: Train Forecast Model
Trains the Conv1D-LSTM model for 15 epochs on 133,560 sequence samples:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/model.py
```

### Step 4: Detect Hotspots
Fits an Isolation Forest outlier model on spatial grids to isolate volatile organic chemical hotspots:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/hotspot.py
```

### Step 5: Launch Dashboard
Launch the Streamlit web dashboard in the browser:
```bash
/Users/geetikavasistha/isro_env/bin/streamlit run dashboard/app.py
```

---

## 📡 Live Google Earth Engine Dataset References

| Variable | Earth Engine Dataset | Band Name |
|---|---|---|
| **HCHO** (Formaldehyde) | `COPERNICUS/S5P/OFFL/L3_HCHO` | `tropospheric_HCHO_column_number_density` |
| **NO2** (Nitrogen Dioxide) | `COPERNICUS/S5P/OFFL/L3_NO2` | `tropospheric_NO2_column_number_density` |
| **AOD Proxy** (Aerosol Index) | `COPERNICUS/S5P/OFFL/L3_AER_AI` | `absorbing_aerosol_index` |
| **Temp 2m** | `ECMWF/ERA5_LAND/HOURLY` | `temperature_2m` |
| **Wind U/V 10m** | `ECMWF/ERA5_LAND/HOURLY` | `u_component_of_wind_10m`, `v_component_of_wind_10m` |
| **Dewpoint 2m** | `ECMWF/ERA5_LAND/HOURLY` | `dewpoint_temperature_2m` |
| **Precipitation** | `ECMWF/ERA5_LAND/HOURLY` | `total_precipitation` |

---

## 🎯 Model Architecture & Metrics

- **Lookback window:** 7 days
- **Train Validation Split:** 80/20
- **Validation Mean Absolute Error (MAE):** 6.44 AQI Points
- **Validation Root Mean Squared Error (RMSE):** 9.60
- **Prediction-Target Correlation (R):** 0.9746
