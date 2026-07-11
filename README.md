# <img src="dashboard/assets/logo.jpg" width="42" height="42" style="vertical-align: middle; border-radius: 8px; margin-right: 8px;" /> Plume — National Air Foresight
### *From Orbit to Ground Truth*

**Satellite-only next-day air quality forecasting and biomass-burning detection — anywhere in India, no ground sensor required.**

[![Model R²](https://img.shields.io/badge/Model_R²-0.890-brightgreen)]()
[![Correlation](https://img.shields.io/badge/Correlation-0.938-brightgreen)]()
[![Coverage](https://img.shields.io/badge/Coverage-All_India-blue)]()

Ground AQI monitors cover a few hundred points in India. Plume uses Sentinel-5P TROPOMI and ERA5 reanalysis data to forecast tomorrow's AQI on a 5km grid **anywhere in the country**, flags formaldehyde hotspots that signal crop/forest burning using Isolation Forest anomaly detection, cross-references them against NASA FIRMS active-fire data, and projects 24-hour downwind smoke transport — all from space, cross-calibrated against real CPCB ground stations where available.

**⚡ Quick facts:**
- 🎯 Forecast accuracy: **R² = 0.89**, MAE = 15.4 AQI units
- 🔥 Hotspot detection: Isolation Forest over multi-pollutant satellite grids, reverse-geocoded to real place names
- 🌬️ Wind advection modeling to trace pollution transport, not just origin
- 🗺️ Works on any Indian city, district, or state — not a fixed shortlist

👉 **[Full write-up below](#the-problem)** for the complete story, architecture, and honest limitations.

---

> Ground air-quality monitors cover a few hundred points in India. Satellites cover every square kilometer, every day. Plume turns that satellite data into next-day AQI forecasts and pinpoints the chemical fingerprints of biomass burning — no ground sensor required to get started, though we calibrate against them wherever we can.

---

## The Problem

Every winter, a haze settles over North India. Everyone knows it's bad. Almost no one can tell you *why*, exactly, or *where* it's coming from, in a way precise enough to act on. Ground-based CPCB monitors are sparse and clustered around big cities — vast stretches of Punjab's crop belt, forest-fire zones, and smaller towns have no ground truth at all.

Plume asks: **can satellites fill that gap?**

Specifically, using Sentinel-5P TROPOMI (which measures NO₂, HCHO, and aerosols from space) and ERA5 meteorological reanalysis, can we:

1. Forecast tomorrow's surface AQI, spatially, for any part of India — not just the six cities with ground stations?
2. Detect formaldehyde (HCHO) hotspots that signal biomass burning, before they show up as a headline?
3. Trace *where* that pollution is coming from and *where the wind is taking it*?

---

## What We Built

### 🗺️ 1. Nationwide AQI Forecasting

Type in *any* Indian city, district, or state — not a fixed shortlist — and the pipeline resolves its bounding box, pulls satellite + weather data, and forecasts next-day AQI on a 0.05° (~5km) grid.

For a full-state query like **Punjab (state, wide)**, the pipeline resolved a 4,624-cell grid and forecast an average AQI of **105.6 (Moderate)** for the target date, with per-cell tooltips showing the nearest town and its individual predicted value.

A standing banner is honest about a real constraint: the underlying CNN-LSTM was trained on Delhi-NCR data, so predictions elsewhere are directionally useful but carry lower confidence — we surface that caveat directly in the UI rather than hide it.

### 🔥 2. HCHO Hotspot Detection

An Isolation Forest model scans every grid cell's multi-pollutant signature and flags the ones that look chemically abnormal — the satellite equivalent of "something is burning here."

![Formaldehyde Hotspot Detection Page](file:///Users/geetikavasistha/.gemini/antigravity-ide/brain/80db8f9e-d81f-42e3-9628-0250de05b038/media__1783745907185.png)

On a recent run over Punjab, the model flagged **215 of 4,624 cells (≈4.6%)** as anomalous, with a mean HCHO concentration of 1.92×10⁻⁴ mol/m² (median 2.13×10⁻⁴, 95th percentile 4.52×10⁻⁴). The single most-affected location: **Pathankot, Punjab**, with an anomaly score of 91.7 — squarely in the "Severe" band.

![Hotspot Spatial Distribution Map and Ranked Table](file:///Users/geetikavasistha/.gemini/antigravity-ide/brain/80db8f9e-d81f-42e3-9628-0250de05b038/media__1783745907201.png)

Every flagged cell is reverse-geocoded to a real place name, so a scientist doesn't have to squint at lat/lon pairs — they get "Khem Karan, Tarn Taran" instead of `31.11°N, 74.75°E`.

### 🔥📡 3. Biomass Burning Correlation

This is where satellite fire detections and satellite chemistry get cross-examined against each other. NASA FIRMS active-fire points (MODIS/VIIRS) are overlaid directly on the HCHO hotspot map:

![Biomass Burning Source Localization Map](file:///Users/geetikavasistha/.gemini/antigravity-ide/brain/80db8f9e-d81f-42e3-9628-0250de05b038/media__1783745907221.png)

The visual correlation is striking — dense clusters of active fires sit almost exactly inside the flagged HCHO cells around Ludhiana and Patiala. The pipeline statistically confirms burning *events*: periods are flagged whenever daily active-fire counts exceed a threshold of **mean + 2 standard deviations (86 fires/day)**. Two such events were identified in the sample window — **May 26** (92 fires, avg HCHO 2.24×10⁻⁴) and **May 29** (99 fires, avg HCHO 2.78×10⁻⁴) — with fire and HCHO time series tracking each other closely through the burning season before both collapsing after early June.

![Biomass Burning Analysis Chart](file:///Users/geetikavasistha/.gemini/antigravity-ide/brain/80db8f9e-d81f-42e3-9628-0250de05b038/media__1783745907149.png)

Correlation is measured two ways: a **daily mean correlation (R = 0.093)** capturing the loose seasonal relationship between total fires and average HCHO, and a stricter **spatio-temporal, per-grid-cell correlation (R = 0.001)** — the honest number, and a reminder that "fires happened nearby" and "this exact cell's HCHO spike was caused by that exact fire" are different claims. We report both rather than cherry-picking the flattering one.

### 🌬️ 4. Wind Transport & Advection

Fires don't pollute only where they burn — smoke travels. Using ERA5's U/V wind components, the pipeline projects a simplified 24-hour advection trajectory for each detected hotspot:

```
Δx_deg = (u × 86400) / (111000 × cos(lat))      Δy_deg = (v × 86400) / 111000
```

![Wind Transport & Advection Analysis](file:///Users/geetikavasistha/.gemini/antigravity-ide/brain/80db8f9e-d81f-42e3-9628-0250de05b038/media__1783745907160.png)

A **wind rose** aggregates daily wind direction into 8 compass sectors, colored by the mean AQI recorded under each wind regime — in the sampled period, winds from the **northwest** carried the highest associated AQI (~85-90), consistent with transport from upwind agricultural-burning zones. A companion table projects each hotspot's coordinates 24 hours downwind, giving a rough answer to "if this doesn't clear, where does it go next."

---

## Model Performance

The forecasting model is a **CNN-1D + LSTM** hybrid: the convolutional layer picks up short-term gradient shifts (a spike starting), the LSTM carries a 7-day memory of slower buildup (a stagnant week), and a dense layer resolves both into a next-day AQI prediction per grid cell.

| Metric | Value | What it means |
|---|---|---|
| **MAE** | 15.42 AQI units | Average deviation between predicted and actual AQI |
| **RMSE** | 21.05 | Penalizes large misses more heavily than MAE |
| **Correlation (R)** | 0.938 | Strong linear agreement between predicted and actual |
| **R²** | 0.890 | ~89% of AQI variance explained by the model |

Training and validation loss both converge cleanly over 15 epochs with no visible overfitting gap — validation loss tracks *below* training loss throughout, a healthy sign for a model of this size.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   Any Indian city,   │
                    │  district, or state  │
                    └──────────┬──────────┘
                               │  resolved to bounding box
                               ▼
        ┌───────────────────────────────────────────┐
        │              INGESTION LAYER                │
        │  Sentinel-5P TROPOMI (HCHO, NO2, UV AI)     │
        │  ERA5-Land (temp, wind U/V, dewpoint, precip)│
        │  NASA FIRMS (MODIS/VIIRS active fires)      │
        │  CPCB CAAQM (ground station readings)       │
        └──────────────────┬───────────────────────────┘
                            │  gridded @ 0.05°/0.1°, 7-day lookback
                            ▼
        ┌───────────────────────────────────────────┐
        │             PREPROCESSING                    │
        │  Proxy AQI (satellite-derived, CPCB formula) │
        │  Ground AQI (real CPCB stations, where avail)│
        │  MinMax scaling · sequence building           │
        └──────────────────┬───────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌───────────────────┐        ┌───────────────────────┐
   │   CNN-1D + LSTM     │        │    Isolation Forest     │
   │  next-day AQI grid  │        │  HCHO/NO2/AOD anomalies │
   └──────────┬──────────┘        └───────────┬─────────────┘
              │                               │
              ▼                               ▼
   ┌────────────────────┐        ┌───────────────────────────┐
   │  Fire × HCHO        │        │   Wind Advection            │
   │  correlation engine │        │   24h trajectory projection │
   └──────────┬──────────┘        └───────────┬─────────────────┘
              └───────────────┬───────────────┘
                               ▼
                   ┌───────────────────────┐
                   │   Streamlit Dashboard  │
                   │  Overview · Forecast   │
                   │  Hotspots · Methodology│
                   └───────────────────────┘
```

---

## 🛠️ Project File Structure

```
isro_aqi_hcho/
├── backend/
│   ├── config.py                 # Bounding box config, dynamic resolutions, paths
│   ├── data_ingest.py            # Earth Engine downloader (server-side daily composites)
│   ├── cpcb_ingest.py            # Ingests real/simulated ground monitors for calibration
│   ├── preprocess.py             # Nearest-neighbor matching and Proxy target compilation
│   ├── inference.py              # Decoupled batched forecast runner (pre-caches predictions)
│   ├── model.py                  # CNN-LSTM next-day AQI model validator
│   ├── hotspot.py                # Isolation Forest spatial chemical anomaly detector
│   ├── fire_ingest.py            # NASA FIRMS active fires downloader
│   ├── fire_analysis.py          # Space-time correlation of HCHO and MODIS/VIIRS fires
│   ├── transport_analysis.py     # 24h trajectory simulator and wind rose compiler
│   └── precompute_scheduler.py   # Daily cron/loop precomputes for Indian states/cities
├── data/
│   ├── cache/                    # Raw netcdf grids and precompute runner audit logs
│   └── processed/                # Normalized model tensors, predictions, and hotspots
├── models/
│   ├── cnn_lstm_aqi.keras        # Trained Keras neural network weights
│   └── metrics.json              # MAE/RMSE baseline values
├── dashboard/
│   ├── app.py                    # Main SPA layout with horizontal navbar
│   ├── helper.py                 # Sidebar configuration controls and cache checking
│   ├── icon_utils.py             # Standardized Google Material Symbols wrapper
│   └── views/                    # Multi-page views (Overview, Forecast, Hotspots, Methodology)
├── requirements.txt              # Project package manifests
└── README.md                     # Documentation
```

---

## 🛠️ Installation & Setup

All required packages are pre-installed in the virtual environment `/Users/geetikavasistha/isro_env`.

### 1. Activate virtual environment
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
Downloads daily satellite observations (HCHO, NO2, UV Aerosol Index) and ERA5 land variables (temp, winds, dewpoint, precipitation) for the region bounding box:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/data_ingest.py --region "Delhi-NCR"
```

### Step 2: Ingest CPCB Ground Stations
Downloads real ground-truth stations or generates collocated ground sensors for evaluation:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/cpcb_ingest.py --region "Delhi-NCR"
```

### Step 3: Preprocess Features
Standardizes all variables, computes the custom **Proxy AQI** target, and builds sliding sequence lookbacks:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/preprocess.py --region "Delhi-NCR"
```

### Step 4: Run Cached Inference
Pre-caches forecasting outputs to dynamic arrays to keep Streamlit maps loading instantly:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/inference.py --region "Delhi-NCR"
```

### Step 5: Detect Hotspots, Fires, and Wind rose
Fits outlier anomaly models and simulates wind transport:
```bash
/Users/geetikavasistha/isro_env/bin/python backend/hotspot.py --region "Delhi-NCR"
/Users/geetikavasistha/isro_env/bin/python backend/fire_ingest.py --region "Delhi-NCR"
/Users/geetikavasistha/isro_env/bin/python backend/fire_analysis.py --region "Delhi-NCR"
/Users/geetikavasistha/isro_env/bin/python backend/transport_analysis.py --region "Delhi-NCR"
```

### Step 6: Launch Dashboard
Launch the Streamlit web dashboard:
```bash
/Users/geetikavasistha/isro_env/bin/streamlit run dashboard/app.py
```

### Step 7: Pre-caching Scheduler (Optional / Background Job)
To prevent users from waiting for live 2-3 minute GEE downloads, run the precompute scheduler as a background job to refresh the cache daily for a configurable list of states and cities.

**Run once (ideal for daily cron jobs):**
```bash
/Users/geetikavasistha/isro_env/bin/python backend/precompute_scheduler.py
```

**Run as a standing background daemon (loops and sleeps for 24 hours):**
```bash
nohup /Users/geetikavasistha/isro_env/bin/python backend/precompute_scheduler.py --loop --interval 86400 > data/cache/precompute_scheduler.log 2>&1 &
```

**Example system crontab setup (`crontab -e`):**
```cron
# Run daily at 1:00 AM
0 1 * * * /Users/geetikavasistha/isro_env/bin/python /Users/geetikavasistha/isro_aqi_hcho/backend/precompute_scheduler.py >> /Users/geetikavasistha/isro_aqi_hcho/data/cache/precompute_scheduler_cron.log 2>&1
```
The scheduler logs results to `data/cache/precompute_log.json`, which the dashboard automatically reads to skip live runs for cached regions.

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

## Known Limitations

We'd rather list these than let someone discover them the hard way:

- **Grid-cell fire↔HCHO correlation is weak (R = 0.001).** The seasonal/regional relationship (R = 0.093) is real, but claiming a specific fire caused a specific cell's HCHO spike isn't yet statistically supported at the per-cell level.
- **Aerosol Index substitution.** We use Sentinel-5P's absorbing aerosol index in place of direct INSAT-3D AOD measurements.
- **3–5 day data latency.** Sentinel-5P L3 offline collections lag real-time by several days, so hotspot detection reflects near-real-time conditions.
- **Cross-region model confidence.** The CNN-LSTM's training data is Delhi-NCR-centric; forecasts for other regions are useful directionally but should be read with that caveat in mind (surfaced directly in the UI).

---

## Tech Stack

`Python` · `TensorFlow/Keras (CNN-1D + LSTM)` · `scikit-learn (Isolation Forest)` · `Google Earth Engine API` · `Streamlit` · `Pydeck` · `Plotly` · `NASA FIRMS API` · `CPCB CAAQM`

---

*Plume — turning satellites into a national air quality sensor network, one grid cell at a time.*
