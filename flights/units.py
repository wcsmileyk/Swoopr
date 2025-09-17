"""
Unit conversion utilities for flight data.

All data is stored in metric units (meters, m/s) in the database
and converted for display based on user preferences.
"""

def meters_to_feet(meters):
    """Convert meters to feet"""
    if meters is None:
        return None
    return meters * 3.28084

def feet_to_meters(feet):
    """Convert feet to meters"""
    if feet is None:
        return None
    return feet / 3.28084

def mps_to_mph(mps):
    """Convert meters per second to miles per hour"""
    if mps is None:
        return None
    return mps * 2.23694

def mps_to_kmh(mps):
    """Convert meters per second to kilometers per hour"""
    if mps is None:
        return None
    return mps * 3.6

def mph_to_mps(mph):
    """Convert miles per hour to meters per second"""
    if mph is None:
        return None
    return mph / 2.23694

def kmh_to_mps(kmh):
    """Convert kilometers per hour to meters per second"""
    if kmh is None:
        return None
    return kmh / 3.6

def sqft_to_sqm(sqft):
    """Convert square feet to square meters"""
    if sqft is None:
        return None
    return sqft * 0.092903

def sqm_to_sqft(sqm):
    """Convert square meters to square feet"""
    if sqm is None:
        return None
    return sqm / 0.092903

def lbs_per_sqft_to_kg_per_sqm(lbs_per_sqft):
    """Convert lbs/sq ft to kg/sq m"""
    if lbs_per_sqft is None:
        return None
    return lbs_per_sqft * 4.88243

def kg_per_sqm_to_lbs_per_sqft(kg_per_sqm):
    """Convert kg/sq m to lbs/sq ft"""
    if kg_per_sqm is None:
        return None
    return kg_per_sqm / 4.88243

def format_distance(meters, units='imperial'):
    """Format distance with appropriate units"""
    if meters is None:
        return "-- "

    if units == 'metric':
        return f"{meters:.1f} m"
    else:  # imperial
        feet = meters_to_feet(meters)
        return f"{feet:.0f} ft"

def format_speed(mps, units='imperial'):
    """Format speed with appropriate units"""
    if mps is None:
        return "-- "

    if units == 'metric':
        kmh = mps_to_kmh(mps)
        return f"{kmh:.1f} km/h"
    else:  # imperial
        mph = mps_to_mph(mps)
        return f"{mph:.1f} mph"

def format_altitude(meters, units='imperial'):
    """Format altitude with appropriate units"""
    if meters is None:
        return "-- "

    if units == 'metric':
        return f"{meters:.1f} m"
    else:  # imperial
        feet = meters_to_feet(meters)
        return f"{feet:.0f} ft"

def get_distance_unit(units='imperial'):
    """Get distance unit string"""
    return "m" if units == 'metric' else "ft"

def get_speed_unit(units='imperial'):
    """Get speed unit string"""
    return "km/h" if units == 'metric' else "mph"

def get_altitude_unit(units='imperial'):
    """Get altitude unit string"""
    return "m" if units == 'metric' else "ft"

def format_area(sqft, units='imperial'):
    """Format area with appropriate units"""
    if sqft is None:
        return "-- "

    if units == 'metric':
        sqm = sqft_to_sqm(sqft)
        return f"{sqm:.1f} m²"
    else:  # imperial
        return f"{sqft:.0f} sq ft"

def format_wing_loading(lbs_per_sqft, units='imperial'):
    """Format wing loading with appropriate units"""
    if lbs_per_sqft is None:
        return "-- "

    if units == 'metric':
        kg_per_sqm = lbs_per_sqft_to_kg_per_sqm(lbs_per_sqft)
        return f"{kg_per_sqm:.1f} kg/m²"
    else:  # imperial
        return f"{lbs_per_sqft:.1f} lbs/sq ft"

def get_area_unit(units='imperial'):
    """Get area unit string"""
    return "m²" if units == 'metric' else "sq ft"

def get_wing_loading_unit(units='imperial'):
    """Get wing loading unit string"""
    return "kg/m²" if units == 'metric' else "lbs/sq ft"

# Conversion constants
METERS_TO_FEET = 3.28084
MPS_TO_MPH = 2.23694
MPS_TO_KMH = 3.6