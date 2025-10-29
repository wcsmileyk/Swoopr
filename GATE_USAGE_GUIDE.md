# How to Use Competition Gates with Flights

## Overview
This guide explains how to associate competition gates with flights and automatically calculate entry gate speed and altitude metrics.

## Step 1: Upload Gate Files

### Via Django Admin
1. Go to `/admin/flights/competitiongate/`
2. Click "Add Competition Gate"
3. Fill in the form:
   - **Name**: e.g., "Arizona Swoop Pond" or "Nationals 2025"
   - **Gate Type**: Select "Standard" or "Speed"
   - **Gate File**: Upload your .gsw or .CSV file
4. Click "Save"
5. The gate file will be automatically parsed

### Verify Parsing
- After saving, check that "Is parsed" shows a green checkmark
- Click "View Map" to see the course visualization
- If parsing failed, check the "Parse error" field

## Step 2: Assign Gates to Flights

### Method 1: During Flight Upload (Future Enhancement)
Currently not implemented in the upload UI - use admin for now.

### Method 2: Via Django Admin
1. Go to `/admin/flights/flight/`
2. Find the flight you want to associate with a gate
3. Click to edit the flight
4. In the "Basic Info" section, select a **Competition Gate** from the dropdown
5. Click "Save"

### Method 3: Bulk Assignment (Python Script)
```python
from flights.models import Flight, CompetitionGate

# Get your gate
gate = CompetitionGate.objects.get(name="Arizona Swoop Pond")

# Assign to all flights at that location
flights = Flight.objects.filter(location_name="Arizona")
for flight in flights:
    flight.competition_gate = gate
    flight.save()

print(f"Assigned gate to {flights.count()} flights")
```

## Step 3: Calculate Gate Metrics

### Automatic Calculation
Gate metrics are automatically calculated when you assign a gate to a flight if you call the method manually:

```python
flight.competition_gate = gate
flight.save()
flight.calculate_gate_metrics()  # Call this manually
```

### Bulk Calculation via Management Command

Calculate for all flights with gates (only those missing metrics):
```bash
python manage.py calculate_gate_metrics
```

Calculate for specific gate:
```bash
python manage.py calculate_gate_metrics --gate-id=1
```

Calculate for specific flight:
```bash
python manage.py calculate_gate_metrics --flight-id=123
```

Recalculate all (including those with existing metrics):
```bash
python manage.py calculate_gate_metrics --recalculate
```

## Step 4: View Gate Metrics

### In Django Admin
1. Go to the flight's admin page
2. Expand the "Gate Metrics" section
3. You'll see:
   - **Gate crossing idx**: GPS point index where gate was crossed
   - **Gate speed mps**: Speed at gate crossing (m/s)
   - **Gate altitude agl**: Altitude at gate crossing (meters AGL)
   - **Passed between gates**: Whether flight cleanly passed between gates

### Convert to User Units
The metrics are stored in metric units. To display in imperial:

```python
# Speed in mph
gate_speed_mph = flight.gate_speed_mps * 2.23694 if flight.gate_speed_mps else None

# Altitude in feet
gate_alt_ft = flight.gate_altitude_agl * 3.28084 if flight.gate_altitude_agl else None
```

## Understanding Gate Metrics

### `gate_crossing_idx`
The GPS point index (at 5Hz = 0.2s intervals) where the flight crossed the entry gate line. This is when the jumper passed through the plane formed by gates 1 and 2.

### `gate_speed_mps`
The ground speed at the moment of gate crossing, in meters per second. This is the actual entry gate speed for competition purposes.

### `gate_altitude_agl`
The altitude above ground level at gate crossing, in meters. This helps verify the jumper was at the correct entry altitude.

### `passed_between_gates`
Boolean indicating whether the flight path crossed cleanly between gates 1 and 2 (within 5 meters of both gates). If `False`, the jumper may have been outside the gate width.

## Example: Complete Workflow

```python
from flights.models import Flight, CompetitionGate

# 1. Upload and parse a gate file (via admin or API)
gate = CompetitionGate.objects.get(name="Arizona Swoop Pond")
print(f"Gate parsed: {gate.is_parsed}")
print(f"Gate 1 position: {gate.get_entry_gate_inside()}")

# 2. Assign gate to flights
my_flights = Flight.objects.filter(pilot=request.user, location_name="Arizona")
for flight in my_flights:
    flight.competition_gate = gate
    flight.save()

# 3. Calculate gate metrics
from flights.management.commands.calculate_gate_metrics import Command
cmd = Command()
cmd.handle(gate_id=gate.id)

# 4. View results
for flight in my_flights:
    if flight.gate_speed_mps:
        print(f"Flight {flight.id}:")
        print(f"  Gate Speed: {flight.gate_speed_mps * 2.23694:.1f} mph")
        print(f"  Gate Altitude: {flight.gate_altitude_agl * 3.28084:.1f} ft AGL")
        print(f"  Passed Gates: {flight.passed_between_gates}")
```

## Troubleshooting

### Gate metrics are all None
- **Check**: Is `competition_gate` assigned to the flight?
- **Check**: Is the gate file parsed? (`gate.is_parsed == True`)
- **Check**: Does the flight have GPS data? (`flight.get_gps_data()` returns data)
- **Run**: `flight.calculate_gate_metrics()` manually
- **Check logs**: Look for error messages

### `passed_between_gates` is False
This means the flight path didn't cross cleanly between the two entry gates. Possible reasons:
- Flight approached from the side
- Flight was outside the 10m gate width
- Gate file parsing error (verify on map visualization)

### Gate speed seems wrong
- **Verify**: Check the gate map to ensure gates are in the right position
- **Check**: View the flight path on the map to see where it crossed
- **Compare**: The `gate_crossing_idx` should be near the `flare_idx` for a typical swoop

## Future Enhancements

Planned features:
- [ ] Gate selection dropdown in flight upload UI
- [ ] Automatic gate suggestion based on GPS location
- [ ] Display gate metrics in flight detail view
- [ ] Filter/sort flights by gate metrics
- [ ] Leaderboards per competition gate
- [ ] Overlay flight paths on gate map visualization

## API Reference

### Flight Model Methods

```python
# Calculate gate metrics for a flight
success = flight.calculate_gate_metrics()  # Returns True/False

# Get associated gate
gate = flight.competition_gate

# Access calculated metrics
speed_mph = flight.gate_speed_mps * 2.23694 if flight.gate_speed_mps else None
alt_ft = flight.gate_altitude_agl * 3.28084 if flight.gate_altitude_agl else None
clean_pass = flight.passed_between_gates
```

### CompetitionGate Model Methods

```python
# Get gate positions
gate_1 = gate.get_entry_gate_inside()  # {lat, lon, alt}
gate_2 = gate.get_entry_gate_outside()  # {lat, lon, alt}
gate_5 = gate.get_exit_gate_center()   # {lat, lon, alt} - speed only

# Get all positions and course config
positions = gate.gate_positions
course = gate.course_config
```

### GateCalculator Utility

```python
from flights.utils.gate_calculator import GateCalculator

calculator = GateCalculator(gate.gate_positions)
metrics = calculator.calculate_entry_gate_metrics(flight.get_gps_data())

# Returns:
# {
#     'crossing_idx': int,
#     'speed_mps': float,
#     'speed_mph': float,
#     'altitude_agl': float,
#     'altitude_ft': float,
#     'gate_1_distance': float,
#     'gate_2_distance': float,
#     'passed_between_gates': bool
# }
```
