"""
Overhead view coordinate transformation for displaying flight paths and gates.

Converts lat/lon coordinates to overhead X/Y view where:
- Entry gate is at origin (0, 0)
- X-axis: forward/backward from gate (downrange is positive)
- Y-axis: left/right from gate (90° from forward direction)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class OverheadViewTransformer:
    """Transform GPS coordinates to overhead view coordinates relative to entry gate"""

    def __init__(self, entry_lat: float, entry_lon: float, entry_heading: float):
        """
        Initialize overhead view transformer.

        Args:
            entry_lat: Entry gate latitude
            entry_lon: Entry gate longitude
            entry_heading: Forward direction heading in degrees (0-360, 0=North)
        """
        self.entry_lat = entry_lat
        self.entry_lon = entry_lon
        self.entry_heading = entry_heading  # Direction pilot is going at entry

    def gps_to_overhead(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convert GPS coordinates to overhead view coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Tuple of (x_meters, y_meters) relative to entry gate
            x: positive = downrange, negative = uprange
            y: positive = right, negative = left
        """
        # Calculate distance and bearing from entry gate to this point
        distance_m = self._haversine_distance(
            self.entry_lat, self.entry_lon,
            lat, lon
        )
        bearing_deg = self._calculate_bearing(
            self.entry_lat, self.entry_lon,
            lat, lon
        )

        # Convert bearing to relative angle from entry heading
        relative_bearing = (bearing_deg - self.entry_heading) % 360

        # Convert to X/Y coordinates
        # X = forward/back (downrange positive), Y = left/right (right positive)
        x = distance_m * np.cos(np.radians(relative_bearing))
        y = distance_m * np.sin(np.radians(relative_bearing))

        return x, y

    def flight_path_to_overhead(self, gps_points: List[Dict]) -> Dict:
        """
        Convert entire flight path to overhead view.

        Args:
            gps_points: List of dicts with 'lat', 'lon' keys

        Returns:
            Dict with:
            - 'x': list of X coordinates in meters
            - 'y': list of Y coordinates in meters
            - 'x_ft': list of X coordinates in feet
            - 'y_ft': list of Y coordinates in feet
        """
        x_list, y_list = [], []

        for point in gps_points:
            x, y = self.gps_to_overhead(point['lat'], point['lon'])
            x_list.append(x)
            y_list.append(y)

        # Convert to feet for display
        x_ft = [x * 3.28084 for x in x_list]
        y_ft = [y * 3.28084 for y in y_list]

        return {
            'x': x_list,
            'y': y_list,
            'x_ft': x_ft,
            'y_ft': y_ft,
        }

    def gate_positions_to_overhead(self, gate_positions: Dict) -> Dict:
        """
        Convert gate positions to overhead view.

        Args:
            gate_positions: Dict from GateFileParser with keys like 'gate_1_inside'
                           Each contains {'lat': float, 'lon': float, 'alt': float}

        Returns:
            Dict with same keys but converted to {x_m, y_m, x_ft, y_ft, alt}
        """
        overhead_gates = {}

        for gate_name, pos in gate_positions.items():
            x_m, y_m = self.gps_to_overhead(pos['lat'], pos['lon'])
            x_ft = x_m * 3.28084
            y_ft = y_m * 3.28084

            overhead_gates[gate_name] = {
                'x_m': x_m,
                'y_m': y_m,
                'x_ft': x_ft,
                'y_ft': y_ft,
                'lat': pos['lat'],
                'lon': pos['lon'],
                'alt': pos['alt'],
            }

        return overhead_gates

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = (np.sin(dlat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c

    @staticmethod
    def _calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2 in degrees (0-360)"""
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2_rad)
        y = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(x, y))
        return (bearing + 360) % 360


def create_overhead_view_data(flight_df, entry_gate_lat: float, entry_gate_lon: float,
                               entry_heading: float, gate_positions: Optional[Dict] = None) -> Dict:
    """
    Create complete overhead view data for chart rendering.

    Args:
        flight_df: DataFrame with 'lat', 'lon' columns
        entry_gate_lat: Entry gate latitude
        entry_gate_lon: Entry gate longitude
        entry_heading: Heading at entry gate (0-360 degrees)
        gate_positions: Optional dict of gate positions from GateFileParser

    Returns:
        Dict with:
        - flight_path: {x_ft, y_ft} for plotting
        - gates: transformed gate positions (if provided)
        - bounds: {x_min, x_max, y_min, y_max} for axis limits
    """
    transformer = OverheadViewTransformer(entry_gate_lat, entry_gate_lon, entry_heading)

    # Convert flight path
    gps_points = [
        {'lat': row['lat'], 'lon': row['lon']}
        for _, row in flight_df.iterrows()
    ]
    flight_overhead = transformer.flight_path_to_overhead(gps_points)

    # Convert gates if provided
    gates_overhead = None
    if gate_positions:
        gates_overhead = transformer.gate_positions_to_overhead(gate_positions)

    # Calculate bounds with some padding
    x_ft = flight_overhead['x_ft']
    y_ft = flight_overhead['y_ft']

    if x_ft and y_ft:
        x_min, x_max = min(x_ft), max(x_ft)
        y_min, y_max = min(y_ft), max(y_ft)

        # Add 10% padding
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
    else:
        x_min, x_max = -500, 500
        y_min, y_max = -500, 500
        x_padding = y_padding = 50

    bounds = {
        'x_min': x_min - x_padding,
        'x_max': x_max + x_padding,
        'y_min': y_min - y_padding,
        'y_max': y_max + y_padding,
    }

    return {
        'flight_path': flight_overhead,
        'gates': gates_overhead,
        'bounds': bounds,
    }
