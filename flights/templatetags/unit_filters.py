"""
Template filters for unit conversion and formatting.
"""

from django import template
from flights.units import *

register = template.Library()

@register.filter
def distance_display(meters, units='imperial'):
    """Format distance with appropriate units"""
    return format_distance(meters, units)

@register.filter
def speed_display(mps, units='imperial'):
    """Format speed with appropriate units"""
    return format_speed(mps, units)

@register.filter
def altitude_display(meters, units='imperial'):
    """Format altitude with appropriate units"""
    return format_altitude(meters, units)

@register.filter
def distance_unit(units='imperial'):
    """Get distance unit string"""
    return get_distance_unit(units)

@register.filter
def speed_unit(units='imperial'):
    """Get speed unit string"""
    return get_speed_unit(units)

@register.filter
def altitude_unit(units='imperial'):
    """Get altitude unit string"""
    return get_altitude_unit(units)

@register.filter
def area_display(sqft, units='imperial'):
    """Format area with appropriate units"""
    return format_area(sqft, units)

@register.filter
def wing_loading_display(lbs_per_sqft, units='imperial'):
    """Format wing loading with appropriate units"""
    return format_wing_loading(lbs_per_sqft, units)

@register.simple_tag
def flight_max_vertical_speed(flight, units='imperial'):
    """Get flight's max vertical speed in specified units"""
    return flight.get_max_vertical_speed(units)

@register.simple_tag
def flight_max_ground_speed(flight, units='imperial'):
    """Get flight's max ground speed in specified units"""
    return flight.get_max_ground_speed(units)

@register.simple_tag
def flight_entry_gate_speed(flight, units='imperial'):
    """Get flight's entry gate speed in specified units"""
    return flight.get_entry_gate_speed(units)

@register.simple_tag
def flight_swoop_distance(flight, units='imperial'):
    """Get flight's swoop distance in specified units"""
    return flight.get_swoop_distance(units)

@register.simple_tag
def flight_altitude(flight, altitude_field, units='imperial'):
    """Get flight's altitude field in specified units"""
    altitude_meters = getattr(flight, altitude_field, None)
    return flight.get_altitude_display(altitude_meters, units)