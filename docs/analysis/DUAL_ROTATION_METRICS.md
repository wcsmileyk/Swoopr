# Dual Rotation Metrics System

## Overview

The Swoopr application now implements a **dual rotation metrics system** that provides two complementary analyses of swoop turn rotations:

1. **Full Swoop Analysis** - Comprehensive analysis from flare initiation to max ground speed
2. **Turn Segment Analysis** - gswoop-style focused analysis of the core rotational component

## Motivation

Different tools measure swoop rotations differently:
- **Our original algorithm**: Measures complete swoop maneuver (flare → max ground speed)
- **gswoop**: Measures focused turn segment (turn initiation → rollout end)

Both approaches have value:
- **Full analysis**: Complete performance evaluation including setup and rollout
- **Turn segment**: Pure rotational technique analysis, compatible with gswoop

## Database Fields

### Full Swoop Metrics (existing, enhanced)
- `turn_rotation` - Full swoop rotation in degrees
- `turn_rotation_confidence` - Confidence score (0.0-1.0)
- `turn_rotation_method` - Calculation method used
- `intended_turn` - Standard turn classification (90°, 270°, 450°, 630°, 810°, 990°)
- `turn_direction` - Turn direction (left/right)

### Turn Segment Metrics (new)
- `turn_segment_rotation` - Turn segment rotation in degrees
- `turn_segment_confidence` - Confidence score (0.0-1.0)
- `turn_segment_method` - Calculation method used
- `turn_segment_intended` - Standard turn classification for segment
- `turn_segment_start_alt` - Segment start altitude (ft AGL)
- `turn_segment_end_alt` - Segment end altitude (ft AGL)
- `turn_segment_duration` - Segment duration (seconds)
- `gswoop_difference` - Difference from gswoop reference (degrees)

## FlightManager Methods

### Core Method
```python
def calculate_dual_rotation_metrics(self, df, flare_idx, max_gspeed_idx, landing_idx):
    """Calculate both full swoop and turn segment rotation metrics"""
```

Returns dict with:
- `'full_swoop'` - Full swoop analysis results
- `'turn_segment'` - Turn segment analysis results (if available)

### Turn Segment Detection
The system uses intelligent boundary detection:

1. **Turn Start**: Altitude where turn rate increases significantly (400-800ft AGL)
2. **Rollout End**: Low altitude where heading stabilizes (10-50ft AGL)

### Rotation Calculation
Uses **intelligent path-following** algorithm that:
- Tracks actual flight path through intermediate headings
- Handles GPS noise with outlier rejection
- Determines dominant turn direction
- Calculates full rotations for multi-rotation turns

## Integration

### Flight Analysis
The `analyze_swoop()` method automatically:
1. Calculates dual metrics for all swoop flights
2. Stores both full swoop and turn segment data
3. Maintains backward compatibility with existing fields

### Usage Example
```python
manager = FlightManager()
dual_metrics = manager.calculate_dual_rotation_metrics(df, flare_idx, max_gspeed_idx, landing_idx)

# Access results
if 'full_swoop' in dual_metrics:
    fs = dual_metrics['full_swoop']
    print(f"Full swoop: {fs['rotation']:.1f}° → {fs['intended_turn']}°")

if 'turn_segment' in dual_metrics:
    ts = dual_metrics['turn_segment']
    print(f"Turn segment: {ts['rotation']:.1f}° → {ts['intended_turn']}°")
```

## ML Training Benefits

The dual metrics system provides clean training data for machine learning:

1. **Ground Truth**: gswoop comparisons for turn segment validation
2. **Feature Rich**: Multiple confidence scores and methods
3. **Comprehensive**: Both focused and complete rotation analyses
4. **User Feedback**: Corrections can improve algorithms over time

## Future Enhancements

### Planned Improvements
1. **gswoop Integration**: Direct comparison and validation
2. **User Corrections**: Interface for correcting direction/rotation
3. **ML Enhancement**: Train models on validated dual metrics
4. **Visualization**: Display both metrics in flight analysis UI

### Refinement Areas
1. **Direction Detection**: Improve left/right turn determination
2. **Boundary Detection**: Refine turn start/end point detection
3. **Confidence Scoring**: Enhance confidence calculation methods

## Benefits

✅ **Comprehensive Analysis**: Two complementary rotation measurements
✅ **gswoop Compatibility**: Turn segment analysis for cross-validation
✅ **ML Ready**: Clean training data for machine learning
✅ **Backward Compatible**: Existing analysis unchanged
✅ **User Validation**: Foundation for user correction system
✅ **Confidence Scoring**: Quality assessment for each measurement

## Technical Notes

### Confidence Calculation
Confidence scores (0.0-1.0) based on:
- Heading progression smoothness
- Distance from standard turn increments
- Turn duration reasonableness
- Method-specific factors

### Method Types
- `smoothed` - Complex number smoothing applied
- `raw` - Direct heading analysis
- `dominant` - Direction-consistent analysis
- `altitude_based_segment` - gswoop-style boundaries
- `legacy` - Fallback to original algorithm

### Standard Turn Classifications
Both metrics classify to standard increments:
- 90° - Quarter turn
- 270° - Three-quarter turn
- 450° - 1.25 turns
- 630° - 1.75 turns
- 810° - 2.25 turns
- 990° - 2.75 turns

This dual approach provides pilots with both comprehensive swoop analysis and focused turn technique evaluation, while laying the foundation for machine learning improvements and gswoop compatibility.