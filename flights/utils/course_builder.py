"""
Course builder for USPA competition swoop courses.

Builds course geometry based on USPA Competition Manual Chapter 12:
- Distance Course (Addendum D)
- Speed Course (Addendum C)
- Zone Accuracy Course (Addendum E)

All courses use the same entry gates, but speed course has different exit gates.
"""

import numpy as np
from typing import Dict, List, Tuple


class CourseBuilder:
    """Build competition course geometry from gate positions"""

    # USPA Competition dimensions (in meters)
    ENTRY_GATE_WIDTH = 10.0  # Gates 1-2 are 10m apart
    ENTRY_GATE_HEIGHT = 1.5  # 1.5m high

    # Speed course
    SPEED_COURSE_LENGTH = 70.0  # 70m from entry to exit gates
    SPEED_EXIT_GATE_WIDTH = 4.0  # Gates 4-5 are 4m apart (2m each side of center)
    SPEED_EXIT_GATE_HEIGHT = 1.5  # 1.5m high

    # Distance course
    DISTANCE_COURSE_LENGTH = 120.0  # Typical distance course length (approximate)

    # Zone Accuracy - 5 zones
    ZONE_WIDTHS = [2.0, 2.0, 2.0, 2.0, 2.0]  # Each zone is 2m wide

    def __init__(self, gate_positions: Dict, gate_type: str):
        """
        Initialize course builder.

        Args:
            gate_positions: Dict with parsed gate positions
            gate_type: 'standard' or 'speed'
        """
        self.gate_positions = gate_positions
        self.gate_type = gate_type

    def build_course(self) -> Dict:
        """
        Build complete course geometry.

        Returns:
            Dict with course configuration including:
            - entry_gates: List of gate corner coordinates
            - exit_gates: List of gate corner coordinates (speed only)
            - course_boundary: Polygon coordinates
            - zones: Zone boundaries (zone accuracy)
            - reference_lines: Visual reference lines
        """
        gate_1 = self.gate_positions['gate_1_inside']
        gate_2 = self.gate_positions['gate_2_outside']

        # Calculate course bearing (direction from entry to exit)
        if self.gate_type == 'speed':
            gate_5 = self.gate_positions['gate_5_center']
            course_bearing = self._calculate_bearing(
                gate_1['lat'], gate_1['lon'],
                gate_5['lat'], gate_5['lon']
            )
            course_length = self.SPEED_COURSE_LENGTH
        else:
            # For standard courses, calculate bearing from gate 1-2 perpendicular
            gate_12_bearing = self._calculate_bearing(
                gate_1['lat'], gate_1['lon'],
                gate_2['lat'], gate_2['lon']
            )
            # Course runs perpendicular to gate line (90° clockwise)
            course_bearing = (gate_12_bearing + 90) % 360
            course_length = self.DISTANCE_COURSE_LENGTH

        # Build entry gates
        entry_gates = self._build_entry_gates(gate_1, gate_2, course_bearing)

        # Build course-specific components
        course_config = {
            'gate_type': self.gate_type,
            'course_bearing': course_bearing,
            'entry_gates': entry_gates,
        }

        if self.gate_type == 'speed':
            # Add speed course exit gates
            gate_5 = self.gate_positions['gate_5_center']
            exit_gates = self._build_speed_exit_gates(gate_5, course_bearing)
            course_config['exit_gates'] = exit_gates

            # Build speed course boundary
            course_config['course_boundary'] = self._build_speed_course_boundary(
                entry_gates, exit_gates
            )

        else:
            # Build distance/zone accuracy course
            course_config['course_boundary'] = self._build_standard_course_boundary(
                gate_1, gate_2, course_bearing, course_length
            )

            # Add zone accuracy zones
            course_config['zones'] = self._build_zone_accuracy_zones(
                gate_1, gate_2, course_bearing
            )

        # Add reference lines for visualization
        course_config['reference_lines'] = self._build_reference_lines(
            gate_1, course_bearing, course_length
        )

        return course_config

    def _build_entry_gates(self, gate_1: Dict, gate_2: Dict, course_bearing: float) -> List[Dict]:
        """
        Build entry gate geometry.

        Entry gates are 10m apart, 1.5m high, perpendicular to course bearing.

        Returns:
            List of gate definitions with corner coordinates
        """
        # Gate bearing is perpendicular to course bearing
        gate_bearing = (course_bearing - 90) % 360

        # Gate 1 (inside)
        gate_1_left = self._offset_position(gate_1['lat'], gate_1['lon'], gate_bearing, 0.5)
        gate_1_right = self._offset_position(gate_1['lat'], gate_1['lon'], gate_bearing + 180, 0.5)

        # Gate 2 (outside) - should be at gate_2 position
        gate_2_left = self._offset_position(gate_2['lat'], gate_2['lon'], gate_bearing, 0.5)
        gate_2_right = self._offset_position(gate_2['lat'], gate_2['lon'], gate_bearing + 180, 0.5)

        return [
            {
                'name': 'Gate 1 (Inside)',
                'center': {'lat': gate_1['lat'], 'lon': gate_1['lon'], 'alt': gate_1['alt']},
                'top_left': {'lat': gate_1_left[0], 'lon': gate_1_left[1], 'alt': gate_1['alt'] + self.ENTRY_GATE_HEIGHT},
                'top_right': {'lat': gate_1_right[0], 'lon': gate_1_right[1], 'alt': gate_1['alt'] + self.ENTRY_GATE_HEIGHT},
                'bottom_left': {'lat': gate_1_left[0], 'lon': gate_1_left[1], 'alt': gate_1['alt']},
                'bottom_right': {'lat': gate_1_right[0], 'lon': gate_1_right[1], 'alt': gate_1['alt']},
            },
            {
                'name': 'Gate 2 (Outside)',
                'center': {'lat': gate_2['lat'], 'lon': gate_2['lon'], 'alt': gate_2['alt']},
                'top_left': {'lat': gate_2_left[0], 'lon': gate_2_left[1], 'alt': gate_2['alt'] + self.ENTRY_GATE_HEIGHT},
                'top_right': {'lat': gate_2_right[0], 'lon': gate_2_right[1], 'alt': gate_2['alt'] + self.ENTRY_GATE_HEIGHT},
                'bottom_left': {'lat': gate_2_left[0], 'lon': gate_2_left[1], 'alt': gate_2['alt']},
                'bottom_right': {'lat': gate_2_right[0], 'lon': gate_2_right[1], 'alt': gate_2['alt']},
            }
        ]

    def _build_speed_exit_gates(self, gate_5: Dict, course_bearing: float) -> List[Dict]:
        """
        Build speed course exit gates.

        Exit gates are 4m apart (2m each side of center), 1.5m high.

        Returns:
            List of gate definitions
        """
        # Gate bearing is perpendicular to course bearing
        gate_bearing = (course_bearing - 90) % 360

        # Gate 4 (left) - 2m left of center
        gate_4_center = self._offset_position(gate_5['lat'], gate_5['lon'], gate_bearing, 2.0)
        gate_4_left = self._offset_position(gate_4_center[0], gate_4_center[1], gate_bearing, 0.5)
        gate_4_right = self._offset_position(gate_4_center[0], gate_4_center[1], gate_bearing + 180, 0.5)

        # Gate 5 (right) - 2m right of center
        gate_5_center_pos = self._offset_position(gate_5['lat'], gate_5['lon'], gate_bearing + 180, 2.0)
        gate_5_left = self._offset_position(gate_5_center_pos[0], gate_5_center_pos[1], gate_bearing, 0.5)
        gate_5_right = self._offset_position(gate_5_center_pos[0], gate_5_center_pos[1], gate_bearing + 180, 0.5)

        return [
            {
                'name': 'Gate 4 (Left)',
                'center': {'lat': gate_4_center[0], 'lon': gate_4_center[1], 'alt': gate_5['alt']},
                'top_left': {'lat': gate_4_left[0], 'lon': gate_4_left[1], 'alt': gate_5['alt'] + self.SPEED_EXIT_GATE_HEIGHT},
                'top_right': {'lat': gate_4_right[0], 'lon': gate_4_right[1], 'alt': gate_5['alt'] + self.SPEED_EXIT_GATE_HEIGHT},
                'bottom_left': {'lat': gate_4_left[0], 'lon': gate_4_left[1], 'alt': gate_5['alt']},
                'bottom_right': {'lat': gate_4_right[0], 'lon': gate_4_right[1], 'alt': gate_5['alt']},
            },
            {
                'name': 'Gate 5 (Right)',
                'center': {'lat': gate_5_center_pos[0], 'lon': gate_5_center_pos[1], 'alt': gate_5['alt']},
                'top_left': {'lat': gate_5_left[0], 'lon': gate_5_left[1], 'alt': gate_5['alt'] + self.SPEED_EXIT_GATE_HEIGHT},
                'top_right': {'lat': gate_5_right[0], 'lon': gate_5_right[1], 'alt': gate_5['alt'] + self.SPEED_EXIT_GATE_HEIGHT},
                'bottom_left': {'lat': gate_5_left[0], 'lon': gate_5_left[1], 'alt': gate_5['alt']},
                'bottom_right': {'lat': gate_5_right[0], 'lon': gate_5_right[1], 'alt': gate_5['alt']},
            }
        ]

    def _build_speed_course_boundary(self, entry_gates: List[Dict], exit_gates: List[Dict]) -> List[Dict]:
        """Build speed course boundary polygon"""
        # Course boundary is rectangle from entry gates to exit gates
        gate_1_left = entry_gates[0]['bottom_left']
        gate_2_right = entry_gates[1]['bottom_right']
        gate_4_left = exit_gates[0]['bottom_left']
        gate_5_right = exit_gates[1]['bottom_right']

        return [
            gate_1_left,
            gate_2_right,
            gate_5_right,
            gate_4_left,
        ]

    def _build_standard_course_boundary(self, gate_1: Dict, gate_2: Dict,
                                       course_bearing: float, course_length: float) -> List[Dict]:
        """Build distance/zone accuracy course boundary"""
        # Course boundary extends from entry gates downrange
        gate_bearing = (course_bearing - 90) % 360

        # Entry gate line
        gate_1_left = self._offset_position(gate_1['lat'], gate_1['lon'], gate_bearing, 0.5)
        gate_2_right = self._offset_position(gate_2['lat'], gate_2['lon'], gate_bearing + 180, 0.5)

        # End of course
        end_1 = self._offset_position(gate_1_left[0], gate_1_left[1], course_bearing, course_length)
        end_2 = self._offset_position(gate_2_right[0], gate_2_right[1], course_bearing, course_length)

        return [
            {'lat': gate_1_left[0], 'lon': gate_1_left[1]},
            {'lat': gate_2_right[0], 'lon': gate_2_right[1]},
            {'lat': end_2[0], 'lon': end_2[1]},
            {'lat': end_1[0], 'lon': end_1[1]},
        ]

    def _build_zone_accuracy_zones(self, gate_1: Dict, gate_2: Dict, course_bearing: float) -> List[Dict]:
        """
        Build zone accuracy zones.

        5 zones, each 2m wide, starting from entry gates.
        """
        gate_bearing = (course_bearing - 90) % 360
        zones = []

        # Start position at entry gates
        start_1 = self._offset_position(gate_1['lat'], gate_1['lon'], gate_bearing, 0.5)
        start_2 = self._offset_position(gate_2['lat'], gate_2['lon'], gate_bearing + 180, 0.5)

        cumulative_distance = 0
        for i, zone_width in enumerate(self.ZONE_WIDTHS, 1):
            # Zone boundaries
            zone_start_1 = self._offset_position(start_1[0], start_1[1], course_bearing, cumulative_distance)
            zone_start_2 = self._offset_position(start_2[0], start_2[1], course_bearing, cumulative_distance)

            cumulative_distance += zone_width

            zone_end_1 = self._offset_position(start_1[0], start_1[1], course_bearing, cumulative_distance)
            zone_end_2 = self._offset_position(start_2[0], start_2[1], course_bearing, cumulative_distance)

            zones.append({
                'zone_number': i,
                'width_m': zone_width,
                'boundary': [
                    {'lat': zone_start_1[0], 'lon': zone_start_1[1]},
                    {'lat': zone_start_2[0], 'lon': zone_start_2[1]},
                    {'lat': zone_end_2[0], 'lon': zone_end_2[1]},
                    {'lat': zone_end_1[0], 'lon': zone_end_1[1]},
                ]
            })

        return zones

    def _build_reference_lines(self, gate_1: Dict, course_bearing: float, course_length: float) -> List[Dict]:
        """Build reference lines for visualization"""
        # Center line down course
        centerline_start = {'lat': gate_1['lat'], 'lon': gate_1['lon']}
        centerline_end_pos = self._offset_position(gate_1['lat'], gate_1['lon'], course_bearing, course_length)
        centerline_end = {'lat': centerline_end_pos[0], 'lon': centerline_end_pos[1]}

        return [
            {
                'name': 'Course Centerline',
                'start': centerline_start,
                'end': centerline_end,
            }
        ]

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

    @staticmethod
    def _offset_position(lat: float, lon: float, bearing: float, distance: float) -> Tuple[float, float]:
        """
        Calculate new position given starting point, bearing, and distance.

        Args:
            lat: Starting latitude
            lon: Starting longitude
            bearing: Bearing in degrees
            distance: Distance in meters

        Returns:
            Tuple of (new_lat, new_lon)
        """
        R = 6371000  # Earth radius in meters

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        bearing_rad = np.radians(bearing)

        new_lat_rad = np.arcsin(
            np.sin(lat_rad) * np.cos(distance / R) +
            np.cos(lat_rad) * np.sin(distance / R) * np.cos(bearing_rad)
        )

        new_lon_rad = lon_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(distance / R) * np.cos(lat_rad),
            np.cos(distance / R) - np.sin(lat_rad) * np.sin(new_lat_rad)
        )

        return np.degrees(new_lat_rad), np.degrees(new_lon_rad)
