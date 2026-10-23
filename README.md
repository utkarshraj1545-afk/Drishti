## Interactive Vision-Language Assistant for Remote Sensing Analysis

## Problem Statement

Remote sensing data contains valuable information about environmental and land-cover changes, but extracting insights typically requires GIS expertise and specialized software.

SatQuery AI aims to bridge this gap by allowing users to ask simple natural-language questions about satellite imagery and receive accurate, data-backed answers.

Example:

> "Has vegetation decreased in this area over the last five years?"

Instead of manually processing satellite imagery, the system automatically analyzes Sentinel-2 data and explains the results in plain language.

---

## Core Design Principle

### AI Never Calculates. AI Only Explains.

A major challenge with Vision-Language Models (VLMs) is that they can describe images but cannot reliably perform precise geospatial calculations.

SatQuery AI solves this by separating responsibilities:

### Deterministic Layer (Ground Truth)

Google Earth Engine performs:

- Satellite image retrieval
- Cloud filtering
- Image compositing
- NDVI calculation
- Change detection
- Statistical computation

### AI Layer

The VLM receives:

- Precomputed metrics
- Satellite imagery
- Analysis results

and generates:

- Natural-language explanations
- Human-readable summaries
- Environmental interpretations

The AI never estimates numerical values.

---

## Current Demo Scope

The project focuses on:

### Vegetation Change Detection

Using:

- Sentinel-2 Surface Reflectance Harmonized imagery
- NDVI (Normalized Difference Vegetation Index)

The system compares two time periods and identifies:

- Vegetation gain
- Vegetation loss
- No significant change

---

## Technology Stack

### Backend

- Python
- FastAPI

### Satellite Data

- Google Earth Engine
- Sentinel-2 SR Harmonized

### AI Layer

- GPT / Claude (planned)

### Frontend

- Leaflet (planned)

---

## Project Structure

```text
satquery_backend/
│
├── app/
│   ├── routes/
│   ├── schemas/
│   └── services/
│       └── ndvi_pipeline.py
│
├── secrets/
│   └── key.json
│
├── test_ndvi_pipeline.py
├── requirements.txt
└── README.md
```

---

## NDVI Processing Pipeline

### 1. Earth Engine Initialization

Authenticate using a Google Cloud service account.

### 2. Area of Interest (AOI)

Convert user-provided coordinates into an Earth Engine geometry.

### 3. Scene Retrieval

Fetch Sentinel-2 imagery for:

- Date range
- Area of interest
- Cloud cover threshold

### 4. Composite Creation

Generate a median composite image to reduce cloud and shadow artifacts.

### 5. NDVI Calculation

NDVI formula:

```text
NDVI = (NIR - Red) / (NIR + Red)
```

Sentinel-2 bands:

```text
B8 = NIR
B4 = Red
```

### 6. Statistical Reduction

Calculate mean NDVI over the selected area.

### 7. Change Detection

Compare:

```text
Before NDVI
vs
After NDVI
```

and determine:

- Vegetation gain
- Vegetation loss
- No significant change

---

## Validated Test Case

### Mohand / Shivalik Forest Corridor, Uttarakhand

This region experienced documented forest clearing associated with the Delhi–Dehradun Expressway project.

### AOI

```text
West  = 77.87
South = 30.15
East  = 77.93
North = 30.21
```

### Results

```text
Before (2018):
Mean NDVI = 0.4925

After (2024):
Mean NDVI = 0.4432

Change:
-10.0%

Classification:
Vegetation Loss
```

The result aligns with independently documented vegetation clearing in the region.

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone <repository-url>
cd satquery_backend
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate Environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Add Earth Engine Credentials

Place your service-account key file:

```text
secrets/key.json
```

---

## Run Test Script

```bash
python test_ndvi_pipeline.py
```

Expected output:

```text
Before NDVI: 0.4925
After NDVI: 0.4432
Change: -10.0%
Vegetation Loss
```

---

## Future Work

### Phase 1

- FastAPI API endpoint
- Swagger documentation

### Phase 2

- Satellite image thumbnails
- Before/after visualization

### Phase 3

- Vision-Language explanation layer

### Phase 4

- Interactive map interface
- AOI drawing tools

### Phase 5

- Photo upload and EXIF geolocation support

---

## Team Drishti

SatQuery AI is being developed as a Smart India Hackathon solution to make satellite-data interpretation accessible to non-GIS users through grounded AI and remote-sensing analytics.
