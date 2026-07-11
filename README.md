# 🛰️ Plume — National Air Foresight
### *From Orbit to Ground Truth*

**Satellite-only next-day air quality forecasting and biomass-burning detection — anywhere in India, no ground sensor required.**

<img width="1377" height="911" alt="Screenshot 2026-07-11 at 2 38 28 AM" src="https://github.com/user-attachments/assets/d73e8ff1-ef19-4d36-8a86-20af3622fd5f" />
<img width="1341" height="682" alt="Screenshot 2026-07-11 at 2 39 11 AM" src="https://github.com/user-attachments/assets/befc2738-c870-4aa4-998e-45b627154399" />


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

## What I Built

### 🗺️ 1. Nationwide AQI Forecasting

Type in *any* Indian city, district, or state — not a fixed shortlist — and the pipeline resolves its bounding box, pulls satellite + weather data, and forecasts next-day AQI on a 0.05° (~5km) grid.

<p align="center"><em>[Insert screenshot: AQI Spatial Forecasting map for Punjab, showing the grid heatmap and forecast summary card]</em></p>

For a full-state query like **Punjab (state, wide)**, the pipeline resolved a 4,624-cell grid and forecast an average AQI of **105.6 (Moderate)** for the target date, with per-cell tooltips showing the nearest town and its individual predicted value.

A standing banner is honest about a real constraint: the underlying CNN-LSTM was trained on Delhi-NCR data, so predictions elsewhere are directionally useful but carry lower confidence — we surface that caveat directly in the UI rather than hide it.

### 🔥 2. HCHO Hotspot Detection

An Isolation Forest model scans every grid cell's multi-pollutant signature and flags the ones that look chemically abnormal — the satellite equivalent of "something is burning here."

<p align="center"><em>[Insert screenshot: Formaldehyde Hotspot Detection page, showing 215/4624 flagged cells and the ranked hotspot table]</em></p>

On a recent run over Punjab, the model flagged **215 of 4,624 cells (≈4.6%)** as anomalous, with a mean HCHO concentration of 1.92×10⁻⁴ mol/m² (median 2.13×10⁻⁴, 95th percentile 4.52×10⁻⁴). The single most-affected location: **Pathankot, Punjab**, with an anomaly score of 91.7 — squarely in the "Severe" band.

Every flagged cell is reverse-geocoded to a real place name, so a scientist doesn't have to squint at lat/lon pairs — they get "Khem Karan, Tarn Taran" instead of `31.11°N, 74.75°E`.

### 🔥📡 3. Biomass Burning Correlation

This is where satellite fire detections and satellite chemistry get cross-examined against each other. NASA FIRMS active-fire points (MODIS/VIIRS) are overlaid directly on the HCHO hotspot map:

<p align="center"><em>[Insert screenshot: Biomass Burning Source Localization map — orange fire points clustering inside magenta HCHO hotspot cells over Punjab's crop belt]</em></p>

The visual correlation is striking — dense clusters of active fires sit almost exactly inside the flagged HCHO cells around Ludhiana and Patiala. The pipeline statistically confirms burning *events*: periods are flagged whenever daily active-fire counts exceed a threshold of **mean + 2 standard deviations (86 fires/day)**. Two such events were identified in the sample window — **May 26** (92 fires, avg HCHO 2.24×10⁻⁴) and **May 29** (99 fires, avg HCHO 2.78×10⁻⁴) — with fire and HCHO time series tracking each other closely through the burning season before both collapsing after early June.

Correlation is measured two ways: a **daily mean correlation (R = 0.093)** capturing the loose seasonal relationship between total fires and average HCHO, and a stricter **spatio-temporal, per-grid-cell correlation (R = 0.001)** — the honest number, and a reminder that "fires happened nearby" and "this exact cell's HCHO spike was caused by that exact fire" are different claims. We report both rather than cherry-picking the flattering one.

### 💨 4. Wind Transport & Advection

Fires don't pollute only where they burn — smoke travels. Using ERA5's U/V wind components, the pipeline projects a simplified 24-hour advection trajectory for each detected hotspot:

```
Δx_deg = (u × 86400) / (111000 × cos(lat))      Δy_deg = (v × 86400) / 111000
```

<p align="center"><em>[Insert screenshot: Wind Rose diagram showing dominant NW-to-E wind sectors colored by mean AQI, alongside the forward-trajectory table]</em></p>

A **wind rose** aggregates daily wind direction into 8 compass sectors, colored by the mean AQI recorded under each wind regime — in the sampled period, winds from the **northwest** carried the highest associated AQI (~85-90), consistent with transport from upwind agricultural-burning zones. A companion table projects each hotspot's coordinates 24 hours downwind, giving a rough answer to "if this doesn't clear, where does it go next."

---

## Model Performance

The forecasting model is a **CNN-1D + LSTM** hybrid: the convolutional layer picks up short-term gradient shifts (a spike starting), the LSTM carries a 7-day memory of slower buildup (a stagnant week), and a dense layer resolves both into a next-day AQI prediction per grid cell.

<p align="center"><em>[Insert screenshot: Model Performance & Validation panel — predicted vs. actual scatter plot and training/validation loss curves]</em></p>

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
                            │  gridded @ 0.05°, 7-day lookback
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

<p align="center"><em>[Insert screenshot: Dashboard overview page — top navbar with logo + Overview/AQI Forecast Map/HCHO Hotspots/Methodology tabs, sidebar with region config, live GEE mode banner]</em></p>

---

## Data Sources

| Source | Provides | Used For |
|---|---|---|
| **Sentinel-5P TROPOMI** | HCHO, NO₂, UV Aerosol Index | Proxy AQI, hotspot detection |
| **ERA5-Land** | Temperature, wind U/V, dewpoint, precipitation | Model features, wind advection |
| **CPCB CAAQM** | Ground-station PM2.5/PM10/AQI | Ground-truth calibration |
| **NASA FIRMS (MODIS/VIIRS)** | Active fire detections | Biomass burning correlation |

---

## Getting Started

```bash
git clone https://github.com/geetikavasistha-01/Plume.git
cd Plume
python -m venv isro_env
source isro_env/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

You'll need:
- A Google Earth Engine project ID (for TROPOMI/ERA5 ingestion)
- A free NASA FIRMS API key (for active fire data) — set as `FIRMS_API_KEY`

---

## Known Limitations

We'd rather list these than let someone discover them the hard way:

- **Grid-cell fire↔HCHO correlation is weak (R = 0.001).** The seasonal/regional relationship (R = 0.093) is real, but claiming a specific fire caused a specific cell's HCHO spike isn't yet statistically supported at the per-cell level.
- **Aerosol Index substitution.** We use Sentinel-5P's absorbing aerosol index in place of direct INSAT-3D AOD measurements, which weren't yet integrated.
- **3–5 day data latency.** Sentinel-5P L3 offline collections lag real-time by several days, so hotspot detection reflects near-real-time conditions, not this exact moment.
- **Cross-region model confidence.** The CNN-LSTM's training data is Delhi-NCR-centric; forecasts for other regions are useful directionally but should be read with that caveat in mind (surfaced directly in the UI).

## Roadmap

- [ ] Deeper CPCB station alignment — ingest real-time PM2.5/PM10 via data.gov.in APIs to keep calibrating GEE-derived proxies
- [ ] INSAT-3D AOD integration once hourly data access is available
- [ ] National-scale gridding beyond current state/city bounding boxes
- [ ] Decoupled FastAPI serving layer for external consumption of predictions

---

## Tech Stack

`Python` · `TensorFlow/Keras (CNN-1D + LSTM)` · `scikit-learn (Isolation Forest)` · `Google Earth Engine API` · `Streamlit` · `Pydeck` · `Plotly` · `NASA FIRMS API` · `CPCB CAAQM`

---

*Plume — turning satellites into a national air quality sensor network, one grid cell at a time.*
