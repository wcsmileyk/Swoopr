# Swoopr Codebase Analysis: File Analysis System

## Executive Summary

**Swoopr** is a Django-based web application for analyzing skydiving swoop flights using FlySight GPS data. The application processes uploaded GPS CSV files to detect swoop maneuvers, calculate rotation metrics, and provide performance analytics for skydivers. The core file analysis operations involve GPS data processing, flight event detection, and machine learning-enhanced rotation calculations.

---

## 1. Overall Project Structure

### Application Type
- **Framework**: Django 5.2 with PostgreSQL/PostGIS database
- **Primary Language**: Python
- **Purpose**: FlySight GPS data analysis platform with swoop flight detection and metrics calculation
- **Key Features**: 
  - User authentication and profiles
  - Flight data import and analysis
  - Swoop detection and rotation measurement
  - Competition gate system
  - ML-enhanced metrics prediction
  - Public flight sharing/statistics

### Main Applications (Django Apps)
```
Swoopr/
├── flights/          # Core flight processing and analysis
├── users/            # User management and authentication
├── analysis/         # Analysis utilities (minimal)
├── visualization/    # Visualization tools (minimal)
├── logbook/          # Logbook features (minimal)
└── api/              # API endpoints (minimal)
```

### Database Architecture
- **Type**: PostgreSQL with PostGIS extension
- **Purpose**: Geographic data support for GPS coordinates
- **Key Models**:
  - `User`: Django authentication
  - `Flight`: Flight records with analysis results
  - `GPSPoint`: Individual GPS samples from flights
  - `CompetitionGate`: Competition gate definitions
  - `Canopy`: Parachute equipment profiles

---

## 2. What "Analyzing Files" Means

### Definition
"Analyzing files" refers to the complete processing pipeline of uploaded FlySight GPS CSV files to extract skydiving metrics and detect swoop maneuvers.

### Specific Operations

#### Phase 1: File Parsing
- **Input**: FlySight CSV format or standard CSV with GPS data
- **Operations**:
  - Parse CSV headers and metadata
  - Extract GPS coordinates (lat, lon, altitude)
  - Extract velocity components (velN, velE, velD)
  - Extract GPS accuracy metrics (hAcc, vAcc, sAcc)
  - Handle multiple CSV format variants

#### Phase 2: Data Processing
- **Altitude Calculation**:
  - Calculate Above Ground Level (AGL) using vertical accuracy (vAcc) as ground level estimator
  - Uses median absolute deviation (MAD) approach for robust ground level detection
  - Examines last 90 seconds of data to find stable ground reference

- **Derived Metrics**:
  - Ground speed calculation: sqrt(velN² + velE²)
  - Heading calculation: arctan2(velE, velN)
  - Vertical speed: velD (downward velocity)
  - Time series alignment and resampling

#### Phase 3: Flight Event Detection
Detects key events in the swoop sequence:

1. **Landing Detection**:
   - Identifies when aircraft reaches ground
   - Checks for sustained low speed/altitude combinations
   - Confirms with backward-looking window of 120 seconds
   - Forward confirmation window of 15 seconds

2. **Flare Detection** (multiple methods):
   - **Traditional Method**: Peak vertical speed analysis
     - Finds maximum downward velocity
     - Verifies flare characteristics (vspeed threshold of 5 m/s)
     - Examines 30-second window around flare
   - **Fallback Method**: Turn detection approach
     - Detects rotation onset as flare proxy
     - Used when traditional method fails

3. **Max Speed Points**:
   - Maximum vertical speed point
   - Maximum ground speed point
   - Timing relative to flare detection

4. **Roll-Out Detection**:
   - Start of pullout phase
   - End of pullout phase
   - Based on vertical speed recovery patterns

#### Phase 4: Rotation Metrics Calculation
Calculates turn/rotation metrics using multiple approaches:

1. **Full Swoop Rotation**:
   - Measures heading change from flare to near-landing
   - Applies rotation continuity detection
   - Filters noise (0.1°/sample threshold)
   - Detects full 360° rotations
   - Provides confidence scoring

2. **Turn Segment Rotation** (gswoop-style):
   - Alternative rotation measurement within specific altitude band
   - Start: above 5000m AGL OR 20 seconds after flare
   - End: below 500m AGL OR max ground speed
   - Separate metrics for comparison

3. **ML-Enhanced Rotation**:
   - XGBoost-based model predictions
   - Feature extraction from flight data
   - Confidence scoring based on model agreement

4. **Multi-Metric ML**:
   - Separate ML models for various performance metrics
   - Predictions stored in database for analysis

#### Phase 5: Feature Extraction for ML
- **Features Extracted**:
  - Turn start characteristics (heading rate, angle accumulation)
  - Flare event properties
  - Speed and altitude at key points
  - Turn duration and average rates
  - G-force profiles during turn
  - Accuracy metrics from GPS

#### Phase 6: Duplicate Detection
- Checks for previously analyzed flights
- Compares with recent flights by user
- Uses signature matching on GPS data

#### Phase 7: Flight Naming
- Chronological flight numbering per user
- Auto-resequencing when gaps detected
- User-friendly naming format

---

## 3. Where File Analysis Code Is Located

### Primary Analysis Engine
**File**: `/home/smiley/PycharmProjects/Swoopr/flights/flight_manager.py` (1,998 lines)

**Key Classes and Methods**:
```python
class FlightManager:
    # Main entry point
    def process_file(filepath, pilot, canopy)
        # Steps: Read → Parse → Create DB objects → Analyze → Update flight

    # File parsing methods
    def read_flysight_file(filepath)
    def _read_standard_csv(filepath)
    def _read_flysight_format(filepath)
    
    # Flight setup
    def create_or_update_flight(filepath, metadata, pilot, canopy)
    def create_gps_points(flight, dataframe)
    
    # Core analysis methods
    def analyze_swoop(flight, dataframe)          # Main analysis orchestrator
    def get_landing(dataframe)                     # Landing detection
    def find_flare(dataframe, landing_idx)        # Flare detection
    def find_max_speeds(dataframe, ...)           # Speed point detection
    def get_roll_out(dataframe, ...)              # Roll-out detection
    
    # Rotation calculations
    def calculate_dual_rotation_metrics(...)       # Full swoop + turn segment
    def get_rotation_with_metadata(...)            # Traditional rotation
    def get_rotation_with_ml_enhancement(...)      # ML-enhanced rotation
    
    # ML integration
    def extract_ml_features(dataframe, ...)        # Feature extraction
    def load_ml_model()                            # Load pre-trained models
    def _add_multi_metric_ml_predictions(...)      # Multi-metric ML
    
    # Utility methods
    def calculate_agl(dataframe)                   # AGL calculation
    def compute_heading(dataframe)                 # Heading computation
    def _check_for_duplicates(flight, ...)         # Duplicate detection
    def _update_flight_naming(flight)              # Flight naming
```

### Supporting Modules
- `/flights/models.py`: Database models (Flight, GPSPoint, CompetitionGate)
- `/flights/utils/gate_parser.py`: Gate file parsing
- `/flights/utils/course_builder.py`: Competition course building
- `/flights/units.py`: Unit conversion utilities

### Entry Points

#### 1. Web Upload (User Interface)
**File**: `/home/smiley/PycharmProjects/Swoopr/users/views.py`
```python
def upload_flight_view(request)
    └─> _process_single_file(request, file, canopy)
        └─> process_flysight_file(filepath, pilot, canopy)
            └─> FlightManager.process_file(filepath, pilot, canopy)
```

**Flow**:
1. User uploads CSV file via web form
2. File temporarily saved to disk
3. FlightManager processes file
4. Results stored in database
5. User redirected to flight detail or swoop selection page

#### 2. Management Command (Batch Processing)
**File**: `/flights/management/commands/import_flysight_data.py`
```python
def handle(directory, username, skip_existing, dry_run)
    └─> FlightManager.process_file(csv_file, ...)
```

**Flow**:
1. CLI command: `python manage.py import_flysight_data /path/to/files`
2. Recursively finds all CSV files
3. Processes each file in sequence
4. Tracks success/failure statistics

#### 3. Reanalysis Commands
**Files**: 
- `/flights/management/commands/reanalyze_all.py`
- `/flights/management/commands/reanalyze_failed_flights.py`
- `/flights/management/commands/reanalyze_simple.py`

**Purpose**: Re-run analysis on existing flights with updated algorithms

---

## 4. Main Entry Points for File Analysis

### 1. Web UI Upload Endpoint
```
Route: POST /users/flight/upload/
Handler: upload_flight_view()
Processing: _process_single_file() or _process_multiple_files()
```

### 2. Programmatic Entry Point
```python
# Direct function call
from flights.flight_manager import process_flysight_file
flight = process_flysight_file(filepath, pilot=user, canopy=canopy_obj)
```

### 3. Batch Processing
```
Command: python manage.py import_flysight_data /path/to/csv/files
Handler: flights.management.commands.import_flysight_data.Command
```

### 4. Reanalysis
```
Commands:
  - python manage.py reanalyze_all
  - python manage.py reanalyze_failed_flights
  - python manage.py reanalyze_simple
```

### 5. Programmatic Directory Processing
```python
from flights.flight_manager import process_directory
results = process_directory(dir_path, pilot=user, canopy=canopy_obj)
```

---

## 5. Performance Monitoring and Logging

### Logging Infrastructure

#### Configuration
**File**: `/Swoopr/settings.py` (lines 230-337)

**Log Files**:
- `/logs/errors.log`: All ERROR level logs (rotating, 10MB max, 5 backups)
- `/logs/server_errors.log`: Server/request errors (rotating, 10MB max, 10 backups)

**Loggers Defined**:
```python
'flights':        # Flight processing logs
'users':          # User operations logs
'django':         # Django framework logs
'django.request': # HTTP request errors
'django.server':  # Server errors
'django.security': # Security events
'swoopr':         # Generic application logs
```

#### Log Levels
- **Development**: INFO level, prints to console
- **Production**: WARNING level, files only
- **Error Logging**: ERROR level captured to file + optional email alerts

### Middleware Monitoring

**File**: `/Swoopr/middleware.py`

#### ErrorLoggingMiddleware
- Logs all 500 errors with:
  - Request method/path
  - Authenticated user
  - Query parameters (sensitive data filtered)
  - POST data size
  - Files uploaded count
  - Full exception traceback
- Filters sensitive parameters: password, token, key, secret

**Usage**: Active in all environments

#### RequestLoggingMiddleware (Optional)
- Can be enabled for verbose request logging
- Logs HTTP status codes: 4xx, 5xx
- Excludes 404/403 by default

### Print Statements (Development)

The flight_manager.py uses strategic print statements for development/debugging:

```python
# Performance indicators
print(f"✅ ML rotation model loaded (improvement: {model_data.get('improvement', 0):+.1f}%)")
print(f"⚠️  ML model not found: {model_path}")

# Warnings
print(f"Warning: Failed to predict {metric_name}: {e}")
print(f"Warning: Multi-metric ML prediction failed: {e}")

# Informational
print(f"Updated flight naming: {flight.flight_name}")
print(f"Loaded multi-metric ML models for {len(self.multi_ml_models)} metrics")
print(f"Potential duplicate detected: {duplicate_info['message']}")
```

### Database Performance

**Indexed Fields** (for query optimization):
- `Flight.pilot_id`: User's flights lookup
- `Flight.created_at`: Recent flights sorting
- `Flight.is_swoop`: Swoop filtering
- `Flight.analysis_successful`: Successful analysis filtering
- `CompetitionGate.gate_type`: Gate type filtering
- `CompetitionGate.created_by`: User's gates
- `CompetitionGate.is_parsed`: Parsed status

### Missing Performance Monitoring

**Not Currently Implemented**:
- No elapsed time measurements for analysis phases
- No memory profiling
- No ML model inference time tracking
- No database query performance monitoring
- No operation success/failure rate metrics
- No file size impact analysis
- No concurrent processing metrics

---

## 6. File Analysis Workflow Diagram

```
User Upload / Management Command
        |
        v
process_flysight_file()
        |
        v
FlightManager.process_file()
        |
        +-----> Read & Parse CSV
        |       - Detect format (FlySight vs standard)
        |       - Parse GPS coordinates
        |       - Parse velocity components
        |
        +-----> Create/Update Flight DB Record
        |
        +-----> Create GPS Points in Database
        |
        +-----> Check for Duplicates
        |
        +-----> Update Flight Naming
        |
        +-----> analyze_swoop()
        |       |
        |       +----> get_landing()
        |       |       - Detect sustained low altitude/speed
        |       |
        |       +----> find_flare()
        |       |       - Traditional: max vspeed detection
        |       |       - Fallback: turn detection
        |       |
        |       +----> find_max_speeds()
        |       |       - Max vertical speed point
        |       |       - Max ground speed point
        |       |
        |       +----> calculate_dual_rotation_metrics()
        |       |       - Full swoop rotation
        |       |       - Turn segment rotation
        |       |
        |       +----> get_rotation_with_ml_enhancement()
        |       |       - Extract features
        |       |       - Load ML model
        |       |       - Predict rotation
        |       |
        |       +----> get_roll_out()
        |       |       - Detect pullout phase
        |       |
        |       +----> Calculate Additional Metrics
        |       |       - Altitudes at key points
        |       |       - Timings and durations
        |       |       - GPS accuracy averages
        |       |
        |       +----> _add_multi_metric_ml_predictions()
        |               - ML predictions for various metrics
        |
        +-----> Store Results in Database
        |
        v
Return Flight Object
```

---

## 7. Key Technical Details

### Data Structures Used
- **Pandas DataFrame**: Main data structure for GPS processing
  - Columns: time, lat, lon, hMSL, velN, velE, velD, hAcc, vAcc, sAcc, heading, AGL, gspeed
  - 5Hz sampling rate typical

- **NumPy Arrays**: Fast numerical operations

- **JSON Fields**: Database storage of:
  - Gate positions
  - Course configurations
  - ML model metadata
  - Predictions

### Machine Learning Integration
- **Models**: Pre-trained XGBoost models
- **Model Path**: `/multi_metric_ml_model.joblib`
- **Feature Count**: ~1800+ features extracted
- **Confidence Scoring**: Model agreement-based

### Units System
- **Storage**: Metric (meters, m/s, degrees)
- **Display**: User preference (imperial/metric)
- **Conversion**: Built-in conversion utilities in `flights/units.py`

---

## 8. Summary

Swoopr is a sophisticated Django application that:

1. **Accepts FlySight GPS files** through web upload or batch import
2. **Parses CSV data** in multiple formats
3. **Detects flight events** (landing, flare, speed peaks, rollout)
4. **Calculates rotation metrics** using traditional and ML-enhanced approaches
5. **Stores results** in PostgreSQL with proper indexing
6. **Provides logging** for errors and monitoring
7. **Uses machine learning** for metric enhancement and prediction

The file analysis operations are well-encapsulated in the FlightManager class with clear separation of concerns across parsing, detection, calculation, and storage phases. Performance monitoring exists at the middleware and database levels but could be enhanced with operation-level timing metrics.

