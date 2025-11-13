# File Analysis - Quick Reference Guide

## What is "File Analysis"?
Processing FlySight GPS CSV files to detect skydiving swoop maneuvers and calculate performance metrics (rotation, speed, altitude, timing, etc.).

## Quick Facts

| Aspect | Details |
|--------|---------|
| **Input** | FlySight or standard CSV with GPS data |
| **Primary Engine** | `flights/flight_manager.py` (FlightManager class) |
| **Processing Phases** | 7 main phases (parse → detect → calculate → store) |
| **Key Detection** | Landing point, flare start, peak speeds, rollout phase |
| **Main Metrics** | Rotation angle, vertical speed, ground speed, altitudes |
| **ML Component** | XGBoost models for rotation/metric enhancement |
| **Database** | PostgreSQL with PostGIS, stores Flight + GPSPoint records |
| **Logging** | Django logging framework + middleware error tracking |

## File Locations

```
Core Analysis:
├── flights/flight_manager.py          Main processing engine (1,998 lines)
├── flights/models.py                  Database models (Flight, GPSPoint, etc.)
├── flights/utils/gate_parser.py       Competition gate parsing
└── flights/units.py                   Unit conversions

Entry Points:
├── users/views.py:upload_flight_view()         Web upload handler
├── flights/management/commands/import_flysight_data.py    Batch import
├── flights/management/commands/reanalyze_*.py            Reanalysis

Configuration:
├── Swoopr/settings.py                 Logging configuration
└── Swoopr/middleware.py               Error logging middleware
```

## Main Classes & Methods

### FlightManager (flights/flight_manager.py)
```python
manager = FlightManager()

# Entry point
manager.process_file(filepath, pilot, canopy)

# Parsing
manager.read_flysight_file(filepath)
manager._read_standard_csv(filepath)
manager._read_flysight_format(filepath)

# Detection
manager.analyze_swoop(flight, dataframe)
manager.get_landing(df)
manager.find_flare(df, landing_idx)
manager.find_max_speeds(df, flare_idx, landing_idx)
manager.get_roll_out(df, max_vspeed_idx, max_gspeed_idx, landing_idx)

# Calculations
manager.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)
manager.get_rotation_with_ml_enhancement(df, flare_idx, max_gspeed_idx)
manager.extract_ml_features(df, flare_idx, max_gspeed_idx)

# Utilities
manager.calculate_agl(df)
manager._check_for_duplicates(flight, df, pilot)
manager._update_flight_naming(flight)
```

## Processing Flow Summary

```
CSV File Upload (Web or CLI)
    ↓
Read & Parse CSV (auto-detect format)
    ↓
Create Flight record in database
    ↓
Create GPSPoint records (5Hz samples)
    ↓
Check for duplicate flights
    ↓
Calculate Above Ground Level (AGL)
    ↓
analyze_swoop() - Main analysis
    ├─ Detect landing point
    ├─ Detect flare start
    ├─ Find peak speeds
    ├─ Calculate rotation (traditional + ML)
    ├─ Detect rollout phase
    └─ Extract all metrics
    ↓
Store results in Flight record
    ↓
Update flight naming (chronological)
    ↓
Return to user / Update UI
```

## Key Operations in Detail

### 1. Landing Detection
- Finds when aircraft reaches ground
- Criteria: Sustained low speed (<5 m/s) + low altitude (<10m AGL)
- Lookback window: 120 seconds
- Forward confirmation: 15 seconds

### 2. Flare Detection
**Method A (Primary)**: Peak vertical speed
- Finds maximum downward velocity point
- Verifies it's within flare characteristics

**Method B (Fallback)**: Turn-based detection
- Uses heading rate to detect rotation onset
- Applied when Method A fails

### 3. Rotation Metrics (Multiple Approaches)
- **Full Swoop**: Heading change from flare to near-landing
- **Turn Segment**: Alternative calculation within altitude band (gswoop-style)
- **ML-Enhanced**: XGBoost prediction with confidence scoring
- **Dual Rotation**: Stores both approaches for comparison

### 4. ML Feature Extraction
Extracts ~1800+ features from flight data:
- Turn rate characteristics
- Flare event properties
- Speed/altitude profiles
- Duration and timing metrics
- G-force profiles
- GPS accuracy statistics

## Database Models

### Flight (Main Model)
```python
# Identifiers
pilot (ForeignKey to User)
device_id, session_id, firmware_version

# Analysis Status
is_swoop (Boolean)
landing_detected (Boolean)
analysis_successful (Boolean)
analysis_error (TextField)

# Key Metrics
turn_rotation (degrees)
turn_rotation_confidence
turn_rotation_method
intended_turn (Boolean)
max_vertical_speed_ms, max_ground_speed_ms
entry_gate_speed_mps

# Indices (flare, landing, max speeds, rollout)
landing_idx, flare_idx, max_vspeed_idx, max_gspeed_idx
rollout_start_idx, rollout_end_idx

# Altitudes
exit_altitude_agl, flare_altitude_agl, landing_altitude_agl

# ML Metrics
ml_rotation, ml_rotation_confidence, ml_intended_turn
(Plus many multi-metric ML fields)
```

### GPSPoint (GPS Data)
```python
flight (ForeignKey)
time, latitude, longitude
altitude_msl, altitude_agl
velocity_north, velocity_east, velocity_down
horizontal_accuracy, vertical_accuracy, speed_accuracy
num_satellites
```

## Entry Points for File Analysis

### 1. Web UI
- Route: `POST /users/flight/upload/`
- Handler: `users/views.py:upload_flight_view()`
- Single/multiple file upload with automatic processing

### 2. Direct Function Call
```python
from flights.flight_manager import process_flysight_file
flight = process_flysight_file('/path/to/file.csv', pilot=user_obj, canopy=canopy_obj)
```

### 3. Batch Import (CLI)
```bash
python manage.py import_flysight_data /path/to/csv/files --username=username
python manage.py import_flysight_data /path --skip-existing --dry-run
```

### 4. Reanalysis (CLI)
```bash
python manage.py reanalyze_all              # Reanalyze all flights
python manage.py reanalyze_failed_flights   # Reanalyze failed only
python manage.py reanalyze_simple           # Simple reanalysis
```

### 5. Programmatic Directory Processing
```python
from flights.flight_manager import process_directory
results = process_directory('/path/to/dir', pilot=user_obj, canopy=canopy_obj)
```

## Logging Configuration

### Log Files
- `/logs/errors.log` - All ERROR level (rotating: 10MB, 5 backups)
- `/logs/server_errors.log` - Server/request errors (rotating: 10MB, 10 backups)

### Loggers
- `flights` - Flight processing operations
- `django.request` - HTTP request errors
- `django.server` - Server errors
- `django.security` - Security events
- `swoopr` - Generic application logs

### Log Levels
- **Development**: INFO to console
- **Production**: WARNING to file only

### Middleware Monitoring
- `ErrorLoggingMiddleware`: Logs all 500 errors with context
  - Request metadata (method, path, user, IP)
  - Query parameters (filters sensitive data)
  - POST data size and files uploaded count
  - Full exception traceback

## Performance Considerations

### Currently Monitored
- Database query indexes (Flight.pilot_id, created_at, is_swoop, analysis_successful)
- Django logging (errors to rotating files)
- Middleware error tracking

### Not Currently Monitored (Optimization Opportunities)
- Operation-level timing (parse time, detection time, calculation time)
- Memory usage during processing
- ML model inference time
- Database query execution time
- File size impact on processing time
- Concurrent processing metrics
- Success/failure rate statistics

## Configuration Files

### Django Settings
- **File**: `Swoopr/settings.py`
- **Key Settings for Analysis**:
  - `INSTALLED_APPS`: flights, users, analysis, visualization, logbook, api
  - `LOGGING`: Complete logging configuration
  - `FILE_UPLOAD_MAX_MEMORY_SIZE`: 5MB
  - `DATA_UPLOAD_MAX_MEMORY_SIZE`: 5MB

### Environment Variables (from .env)
```
DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
SECRET_KEY
DEBUG (True/False)
ALLOWED_HOSTS
SECURE_SSL_REDIRECT, etc.
```

## Useful Django Commands for Analysis

```bash
# View recent error logs
tail -f /home/smiley/PycharmProjects/Swoopr/logs/errors.log

# Check flight count by user
python manage.py shell
>>> from flights.models import Flight
>>> Flight.objects.filter(pilot__username='smiley').count()

# Reindex database
python manage.py sqlsequencereset flights | python manage.py dbshell

# Check for data integrity
python manage.py check

# Database migrations
python manage.py showmigrations flights
python manage.py migrate flights
```

## Common Issues & Debugging

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| CSV parsing fails | Wrong format or missing columns | Check format detection logic in `_read_standard_csv` vs `_read_flysight_format` |
| Landing not detected | Flaky GPS data at end | Review `get_landing()` thresholds |
| Flare detection fails | No clear peak vertical speed | Fallback to turn detection method |
| ML model not loaded | Wrong path or corrupted file | Check `/multi_metric_ml_model.joblib` exists |
| Rotation calculation off | Heading unwrapping issue | Review `unwrap_deg()` function |
| Duplicate detection false positive | Similar flight signatures | Check `_check_for_duplicates()` logic |

---

**Document Generated**: Comprehensive analysis of Swoopr file analysis system  
**Last Updated**: 2025-11-13  
**Primary Source**: `/flights/flight_manager.py`, Django models, views, and settings
