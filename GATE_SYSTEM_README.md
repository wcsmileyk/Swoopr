# Competition Gate System - Implementation Summary

## Overview
Successfully implemented a comprehensive system for uploading, parsing, and visualizing competition gate files for swoop courses. The system automatically identifies gate positions from GPS tracks and builds complete course geometries based on USPA competition rules.

## Features Implemented

### 1. **Gate File Parsing** (`flights/utils/gate_parser.py`)
- Parses .gsw/.CSV gate files containing GPS tracks
- Identifies stationary GPS clusters (45-second stands) using DBSCAN clustering
- Calculates cross pattern centroid to determine inside entry gate (Gate 1)
- Automatically places outside entry gate (Gate 2) 10 meters to the left
- For speed courses, identifies 5th cluster as exit gate center
- Supports both FlySight v2 and legacy CSV formats

### 2. **Course Builder** (`flights/utils/course_builder.py`)
- Builds complete USPA competition courses from gate positions
- **Standard Courses** (Distance/Zone Accuracy):
  - 10m wide entry gates (Gates 1 & 2)
  - 120m course length
  - 5 zones for Zone Accuracy, each 2m wide
- **Speed Courses**:
  - 10m wide entry gates
  - 70m course length
  - 4m wide exit gates (Gates 4 & 5, 2m each side of center)
- Generates complete geometry including boundaries, zones, and reference lines

### 3. **Database Model** (`flights/models.py`)
- `CompetitionGate` model stores:
  - Gate file upload
  - Gate type (standard vs speed)
  - Parsed gate positions (JSON)
  - Course configuration (JSON)
  - Parse status and error tracking

### 4. **Admin Interface** (`flights/admin.py`)
- Upload gate files directly in Django admin
- Automatic parsing on upload
- Parse/Reparse buttons for manual control
- View parsed gate positions and course config
- "View Map" button links to visualization

### 5. **Map Visualization** (`flights/templates/flights/gate_map.html`)
- Interactive Leaflet map showing complete course
- Entry gates rendered with color coding:
  - Green: Gate 1 (inside)
  - Red: Gate 2 (outside)
  - Orange: Gates 4 & 5 (speed course exit gates)
- Course boundary polygon
- Zone Accuracy zones with color gradients
- Course centerline reference
- Responsive legend

### 6. **Gate Calculator** (`flights/utils/gate_calculator.py`)
- Calculates entry gate crossing metrics for flights
- Detects when flight crosses gate line
- Returns:
  - Speed at gate (m/s and mph)
  - Altitude at gate (meters and feet)
  - Distance from each gate
  - Whether flight passed between gates

## Testing Results

Tested with sample gate files:
- ✅ **13-43-22-AZgates.CSV** (Standard gates) - Successfully parsed
- ✅ **00-20-38-AZSpeedgates.CSV** (Speed gates) - Successfully parsed

Both files correctly identified:
- Cross pattern centroids
- Gate 1 (inside) position
- Gate 2 (outside) position 10m offset
- Gate 5 (exit center) for speed course
- Complete course geometry

## Usage

### Uploading Gate Files

1. Go to Django Admin → Flights → Competition Gates
2. Click "Add Competition Gate"
3. Enter name (e.g., "Arizona Swoop Pond")
4. Select gate type: Standard or Speed
5. Upload .gsw or .CSV gate file
6. Save - parsing happens automatically

### Viewing Course Map

1. In admin list, click "View Map" button for any parsed gate
2. Interactive map displays complete course with all gates and zones
3. Click on gates/zones for detailed information

### Using in Flight Analysis

```python
from flights.models import CompetitionGate, Flight
from flights.utils.gate_calculator import GateCalculator

# Get gate
gate = CompetitionGate.objects.get(name="Arizona Swoop Pond")

# Calculate metrics for a flight
flight = Flight.objects.get(id=123)
gps_data = flight.get_gps_data()

calculator = GateCalculator(gate.gate_positions)
metrics = calculator.calculate_entry_gate_metrics(gps_data)

if metrics:
    print(f"Entry Gate Speed: {metrics['speed_mph']:.1f} mph")
    print(f"Entry Gate Altitude: {metrics['altitude_ft']:.1f} ft AGL")
    print(f"Passed between gates: {metrics['passed_between_gates']}")
```

## File Structure

```
flights/
├── models.py                    # CompetitionGate model
├── admin.py                     # Admin interface with parse actions
├── views.py                     # Map visualization views
├── urls.py                      # URL patterns for gate views
├── utils/
│   ├── gate_parser.py          # GPS track parsing and clustering
│   ├── course_builder.py       # USPA course geometry builder
│   └── gate_calculator.py      # Flight gate crossing calculator
└── templates/
    └── flights/
        └── gate_map.html       # Interactive Leaflet map

test_gate_parser.py              # Test script for validation
```

## Next Steps

To integrate with flight analysis:

1. **Add gate selection to flights**:
   - Add `competition_gate` ForeignKey to Flight model
   - Allow users to select which gate file to use for each flight

2. **Automatic gate metrics calculation**:
   - Update flight analysis to use GateCalculator
   - Store entry gate speed/altitude in Flight model
   - Use for more accurate performance metrics

3. **Course-specific leaderboards**:
   - Filter personal bests by competition gate
   - Compare flights at the same venue
   - Track performance across different courses

4. **Advanced visualizations**:
   - Overlay flight paths on course map
   - Show multiple flights on same course
   - Heat maps of landing zones

## Dependencies

- **Django 5.2+** with PostGIS
- **NumPy** - Mathematical operations
- **scikit-learn** - DBSCAN clustering for stationary point detection
- **Leaflet.js** - Map visualization (loaded from CDN)

## Competition Rules Reference

Based on USPA Competition Manual Chapter 12:
- **Addendum B**: Basic course descriptions
- **Addendum C**: Speed course specifications (70m, 4m exit gates)
- **Addendum D**: Distance course specifications
- **Addendum E**: Zone Accuracy course specifications (5 zones × 2m)
- **Addendum G**: Visual course descriptions

Entry gates (Gates 1 & 2): 10 meters apart, 1.5 meters high
Speed exit gates (Gates 4 & 5): 4 meters total width, 1.5 meters high
