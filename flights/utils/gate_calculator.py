"""
Gate crossing calculator for flights.

Calculates entry gate speed and altitude when a flight crosses through competition gates.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class GateCalculator:
    """Calculate gate crossing metrics for flights"""

    def __init__(self, gate_positions: Dict):
        """
        Initialize calculator with gate positions.

        Args:
            gate_positions: Dict with parsed gate positions from CompetitionGate
        """
        self.gate_positions = gate_positions
        self.gate_1 = gate_positions.get('gate_1_inside')
        self.gate_2 = gate_positions.get('gate_2_outside')

    def calculate_entry_gate_metrics(self, gps_data: list) -> Optional[Dict]:
        """
        Calculate speed and altitude when flight crosses entry gates.

        Args:
            gps_data: List of GPS points from flight

        Returns:
            Dict with entry gate metrics:
            {
                'crossing_idx': int,  # Index where flight crossed gates
                'speed_mps': float,   # Speed at gate crossing in m/s
                'altitude_agl': float, # Altitude at gate crossing in meters AGL
                'gate_1_distance': float,  # Distance from gate 1 in meters
                'gate_2_distance': float,  # Distance from gate 2 in meters
            }
            Returns None if no crossing detected
        """
        if not self.gate_1 or not self.gate_2:
            return None

        # Define gate line
        gate_lat1 = self.gate_1['lat']
        gate_lon1 = self.gate_1['lon']
        gate_lat2 = self.gate_2['lat']
        gate_lon2 = self.gate_2['lon']

        # Find where flight crosses the gate line
        crossing_idx = None
        min_distance_to_line = float('inf')
        best_crossing_point = None

        for i in range(1, len(gps_data)):
            p1 = gps_data[i - 1]
            p2 = gps_data[i]

            # Check if flight crosses gate line between p1 and p2
            crossed, distance = self._check_line_crossing(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon'],
                gate_lat1, gate_lon1,
                gate_lat2, gate_lon2
            )

            if crossed and distance < min_distance_to_line:
                min_distance_to_line = distance
                crossing_idx = i
                best_crossing_point = p2

        if crossing_idx is None:
            # No crossing detected, find closest approach
            crossing_idx, best_crossing_point = self._find_closest_approach(
                gps_data, gate_lat1, gate_lon1, gate_lat2, gate_lon2
            )

        if crossing_idx is None or best_crossing_point is None:
            return None

        # Calculate metrics at crossing point
        speed_mps = best_crossing_point.get('ground_speed', 0)
        altitude_agl = best_crossing_point.get('altitude_agl', 0)

        # Calculate distances from each gate
        gate_1_dist = self._haversine_distance(
            best_crossing_point['lat'], best_crossing_point['lon'],
            gate_lat1, gate_lon1
        )
        gate_2_dist = self._haversine_distance(
            best_crossing_point['lat'], best_crossing_point['lon'],
            gate_lat2, gate_lon2
        )

        return {
            'crossing_idx': crossing_idx,
            'speed_mps': speed_mps,
            'speed_mph': speed_mps * 2.23694,  # Convert to mph
            'altitude_agl': altitude_agl,
            'altitude_ft': altitude_agl * 3.28084,  # Convert to feet
            'gate_1_distance': gate_1_dist,
            'gate_2_distance': gate_2_dist,
            'passed_between_gates': gate_1_dist < 5.0 and gate_2_dist < 5.0  # Within 5m of both gates
        }

    def _check_line_crossing(self, p1_lat: float, p1_lon: float,
                            p2_lat: float, p2_lon: float,
                            g1_lat: float, g1_lon: float,
                            g2_lat: float, g2_lon: float) -> Tuple[bool, float]:
        """
        Check if line segment p1-p2 crosses line segment g1-g2.

        Returns:
            Tuple of (crossed: bool, distance: float)
        """
        # Convert to simple 2D coordinates for crossing check
        # Using simple projection for small distances

        # Vector from g1 to g2 (gate line)
        gx = (g2_lon - g1_lon) * 111320 * np.cos(np.radians((g1_lat + g2_lat) / 2))
        gy = (g2_lat - g1_lat) * 110540

        # Vector from g1 to p1
        p1x = (p1_lon - g1_lon) * 111320 * np.cos(np.radians((g1_lat + p1_lat) / 2))
        p1y = (p1_lat - g1_lat) * 110540

        # Vector from g1 to p2
        p2x = (p2_lon - g1_lon) * 111320 * np.cos(np.radians((g1_lat + p2_lat) / 2))
        p2y = (p2_lat - g1_lat) * 110540

        # Cross products to determine which side of gate line points are on
        cross1 = gx * p1y - gy * p1x
        cross2 = gx * p2y - gy * p2x

        # If signs are different, the points are on opposite sides (crossing occurred)
        crossed = (cross1 * cross2) < 0

        # Calculate perpendicular distance from p2 to gate line
        gate_length = np.sqrt(gx**2 + gy**2)
        if gate_length > 0:
            distance = abs(cross2) / gate_length
        else:
            distance = float('inf')

        return crossed, distance

    def _find_closest_approach(self, gps_data: list,
                               gate_lat1: float, gate_lon1: float,
                               gate_lat2: float, gate_lon2: float) -> Tuple[Optional[int], Optional[Dict]]:
        """
        Find the point where flight comes closest to the gate line.

        Returns:
            Tuple of (index, gps_point)
        """
        min_distance = float('inf')
        closest_idx = None
        closest_point = None

        for i, point in enumerate(gps_data):
            # Calculate perpendicular distance to gate line
            distance = self._point_to_line_distance(
                point['lat'], point['lon'],
                gate_lat1, gate_lon1,
                gate_lat2, gate_lon2
            )

            if distance < min_distance:
                min_distance = distance
                closest_idx = i
                closest_point = point

        return closest_idx, closest_point

    def _point_to_line_distance(self, p_lat: float, p_lon: float,
                                l1_lat: float, l1_lon: float,
                                l2_lat: float, l2_lon: float) -> float:
        """Calculate perpendicular distance from point to line in meters"""
        # Convert to meters
        l1x = 0
        l1y = 0
        l2x = (l2_lon - l1_lon) * 111320 * np.cos(np.radians((l1_lat + l2_lat) / 2))
        l2y = (l2_lat - l1_lat) * 110540
        px = (p_lon - l1_lon) * 111320 * np.cos(np.radians((l1_lat + p_lat) / 2))
        py = (p_lat - l1_lat) * 110540

        # Line vector
        lx = l2x - l1x
        ly = l2y - l1y

        # Point vector
        vx = px - l1x
        vy = py - l1y

        # Calculate perpendicular distance
        line_length_sq = lx**2 + ly**2
        if line_length_sq == 0:
            return np.sqrt(vx**2 + vy**2)

        # Project point onto line
        t = max(0, min(1, (vx * lx + vy * ly) / line_length_sq))
        proj_x = l1x + t * lx
        proj_y = l1y + t * ly

        # Distance from point to projection
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)

    @staticmethod
    def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates in meters"""
        R = 6371000  # Earth radius in meters

        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return R * c
