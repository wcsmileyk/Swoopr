
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
from django.contrib.postgres.indexes import GistIndex
from django.contrib.auth.models import User
import json
import gzip
import base64
from .units import *


class Flight(models.Model):
    # User association
    pilot = models.ForeignKey(User, on_delete=models.CASCADE, related_name='flights', null=True, blank=True)
    canopy = models.ForeignKey('users.Canopy', on_delete=models.SET_NULL, null=True, blank=True, related_name='flights')

    # Basic flight metadata
    device_id = models.CharField(max_length=50)
    session_id = models.CharField(max_length=50)
    firmware_version = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, blank=True)

    # Flight location and conditions (optional)
    location_name = models.CharField(max_length=100, blank=True, help_text="Drop zone or location name")
    weather_conditions = models.TextField(blank=True)
    notes = models.TextField(blank=True, help_text="Pilot notes about the jump")

    # Flight type and analysis status
    is_swoop = models.BooleanField(default=False)
    landing_detected = models.BooleanField(default=False)
    analysis_successful = models.BooleanField(default=False)
    analysis_error = models.TextField(blank=True)
    data_incorrect = models.BooleanField(default=False, help_text="Flag for flights with incorrect or problematic data")
    flare_detection_method = models.CharField(
        max_length=20,
        choices=[('traditional', 'Traditional Flare'), ('turn_detection', 'Turn Detection Fallback')],
        null=True, blank=True,
        help_text="Method used to detect the flare/turn initiation point"
    )

    # Privacy controls
    is_public = models.BooleanField(default=False, help_text="Allow others to view this flight")

    # Key indices (store as record numbers for referencing GPS points)
    landing_idx = models.IntegerField(null=True, blank=True)
    flare_idx = models.IntegerField(null=True, blank=True)
    max_vspeed_idx = models.IntegerField(null=True, blank=True)
    max_gspeed_idx = models.IntegerField(null=True, blank=True)
    rollout_start_idx = models.IntegerField(null=True, blank=True)
    rollout_end_idx = models.IntegerField(null=True, blank=True)

    # Calculated swoop metrics
    turn_rotation = models.FloatField(null=True, blank=True, help_text="Turn rotation in degrees")
    turn_direction = models.CharField(
        max_length=5,
        choices=[('left', 'Left'), ('right', 'Right')],
        null=True, blank=True
    )
    max_vertical_speed_mph = models.FloatField(null=True, blank=True)
    max_vertical_speed_ms = models.FloatField(null=True, blank=True)
    max_ground_speed_mph = models.FloatField(null=True, blank=True)
    max_ground_speed_ms = models.FloatField(null=True, blank=True)
    turn_time = models.FloatField(null=True, blank=True, help_text="Turn time in seconds")
    rollout_time = models.FloatField(null=True, blank=True, help_text="Rollout time in seconds")
    swoop_distance_ft = models.FloatField(null=True, blank=True, help_text="Swoop distance from rollout end to landing in feet")
    swoop_distance_m = models.FloatField(null=True, blank=True, help_text="Swoop distance from rollout end to landing in meters")

    # Altitudes at key points (in meters AGL)
    exit_altitude_agl = models.FloatField(null=True, blank=True)
    flare_altitude_agl = models.FloatField(null=True, blank=True)
    max_vspeed_altitude_agl = models.FloatField(null=True, blank=True)
    max_gspeed_altitude_agl = models.FloatField(null=True, blank=True)
    rollout_start_altitude_agl = models.FloatField(null=True, blank=True)
    rollout_end_altitude_agl = models.FloatField(null=True, blank=True)
    landing_altitude_agl = models.FloatField(null=True, blank=True)
    swoop_avg_altitude_agl = models.FloatField(null=True, blank=True, help_text="Average altitude during swoop (flare to landing) in meters AGL")

    # Speed metrics (stored in metric units)
    entry_gate_speed_mps = models.FloatField(null=True, blank=True, help_text="Speed at entry gate (flare initiation) in m/s")
    entry_gate_speed_mph = models.FloatField(null=True, blank=True, help_text="Speed at entry gate (flare initiation) in mph (legacy)")

    # Flight duration and timing
    total_flight_time = models.FloatField(null=True, blank=True, help_text="Total flight time in seconds")
    swoop_start_time = models.DateTimeField(null=True, blank=True)
    swoop_end_time = models.DateTimeField(null=True, blank=True)

    # Analysis metadata
    analysis_version = models.CharField(max_length=20, default='enhanced_v1')
    analyzed_at = models.DateTimeField(null=True, blank=True)

    # Accuracy metrics during swoop (in meters and m/s)
    swoop_avg_horizontal_accuracy = models.FloatField(null=True, blank=True, help_text="Average hAcc during swoop in meters")
    swoop_avg_vertical_accuracy = models.FloatField(null=True, blank=True, help_text="Average vAcc during swoop in meters")
    swoop_avg_speed_accuracy = models.FloatField(null=True, blank=True, help_text="Average sAcc during swoop in m/s")

    # GPS data storage (new efficient approach)
    gps_data_compressed = models.TextField(null=True, blank=True, help_text="Compressed JSON GPS data")
    gps_data_hash = models.CharField(max_length=64, null=True, blank=True, help_text="Hash of GPS data for integrity")

    class Meta:
        unique_together = ['pilot', 'device_id', 'session_id']
        indexes = [
            models.Index(fields=['pilot']),
            models.Index(fields=['pilot', 'created_at']),
            models.Index(fields=['pilot', 'is_swoop']),
            models.Index(fields=['pilot', 'is_swoop', 'analysis_successful']),  # Dashboard queries
            models.Index(fields=['device_id']),
            models.Index(fields=['session_id']),
            models.Index(fields=['created_at']),
            models.Index(fields=['is_swoop']),
            models.Index(fields=['turn_rotation']),
            models.Index(fields=['max_vertical_speed_mph']),
            models.Index(fields=['max_ground_speed_mph']),  # Dashboard personal bests
            models.Index(fields=['analysis_successful']),
            models.Index(fields=['canopy']),
            models.Index(fields=['data_incorrect']),
            models.Index(fields=['is_public']),  # Public flight queries
            models.Index(fields=['pilot', 'is_public']),  # Public user flight queries
            models.Index(fields=['rollout_end_idx']),  # Flight detail calculations
            models.Index(fields=['landing_idx']),  # Flight detail calculations
            models.Index(fields=['flare_idx']),  # Flight detail calculations
            models.Index(fields=['swoop_distance_ft']),  # Dashboard personal bests
            models.Index(fields=['swoop_avg_altitude_agl']),  # Distance PR filtering
            models.Index(fields=['gps_data_hash']),  # GPS data integrity checks
        ]
        ordering = ['-created_at']

    def __str__(self):
        swoop_info = f" - {self.turn_rotation:.0f}° {self.turn_direction}" if self.is_swoop else ""
        pilot_name = self.pilot.username if self.pilot else "Unknown"
        return f"{pilot_name}: {self.session_id[:8]}{swoop_info}"

    @property
    def rotation_magnitude(self):
        """Return absolute rotation value"""
        return abs(self.turn_rotation) if self.turn_rotation is not None else None

    @property
    def is_analyzed(self):
        """Check if flight has been analyzed"""
        return self.analyzed_at is not None

    @property
    def wing_loading(self):
        """Calculate wing loading for this flight"""
        if self.canopy and self.pilot.profile.exit_weight:
            return round(self.pilot.profile.exit_weight / self.canopy.size, 2)
        return None

    @property
    def display_location(self):
        """Return location name or coordinates"""
        if self.location_name:
            return self.location_name
        # Could return lat/lon from first GPS point if available
        first_point = self.gps_points.first()
        if first_point:
            return f"{first_point.location.y:.4f}, {first_point.location.x:.4f}"
        return "Unknown"

    @property
    def performance_grade(self):
        """Calculate performance grade based on personal bests for this specific canopy"""
        if not self.is_swoop or not self.pilot or not self.canopy:
            return None

        # Get pilot's personal bests for the same canopy
        pilot_swoops = Flight.objects.filter(
            pilot=self.pilot,
            canopy=self.canopy,
            is_swoop=True,
            analysis_successful=True
        ).exclude(id=self.id)  # Exclude current flight

        if not pilot_swoops.exists():
            return None  # No comparison data yet

        # Get personal bests
        max_vertical_speed = pilot_swoops.aggregate(
            models.Max('max_vertical_speed_mph')
        )['max_vertical_speed_mph__max']

        # Calculate max distance for personal best (only for swoops with avg altitude ≤ 5m AGL)
        max_distance = 0
        for swoop in pilot_swoops.filter(swoop_avg_altitude_agl__lte=5.0):
            # Use stored swoop distance if available, otherwise calculate
            if swoop.swoop_distance_ft:
                max_distance = max(max_distance, swoop.swoop_distance_ft)
            elif swoop.rollout_end_idx and swoop.landing_idx:
                # Calculate on demand if not stored
                distance = swoop.calculate_and_store_swoop_distance()
                if distance:
                    max_distance = max(max_distance, distance)

        # Calculate current flight distance
        current_distance = 0
        if self.swoop_distance_ft:
            current_distance = self.swoop_distance_ft
        elif self.rollout_end_idx and self.landing_idx:
            # Calculate on demand if not stored
            current_distance = self.calculate_and_store_swoop_distance() or 0

        # Calculate performance percentages
        speed_percentage = 0
        distance_percentage = 0

        if max_vertical_speed and self.max_vertical_speed_mph:
            speed_percentage = (self.max_vertical_speed_mph / max_vertical_speed) * 100

        if max_distance > 0 and current_distance > 0:
            distance_percentage = (current_distance / max_distance) * 100

        # Use the better of speed or distance performance
        best_percentage = max(speed_percentage, distance_percentage)

        # Grade based on percentage of personal best
        if best_percentage >= 99:  # Within 1%
            return "A"
        elif best_percentage >= 90:  # Within 10%
            return "B"
        elif best_percentage >= 70:  # Within 30%
            return "C"
        elif best_percentage >= 50:  # Within 50%
            return "D"
        else:
            return "F"

    def get_chart_data(self):
        """Get GPS data formatted for time series charts"""
        # Try new JSON storage first, fallback to legacy
        gps_data = self.get_gps_data()
        if not gps_data:
            return self._get_chart_data_legacy()

        if not gps_data:
            return None

        # Convert to lists for JSON serialization
        timestamps = []
        altitudes_agl = []
        altitudes_msl = []
        vertical_speeds = []
        ground_speeds = []
        headings = []

        # Convert to seconds from start for x-axis using index position and 5Hz sampling rate
        for i, point in enumerate(gps_data):
            # FlySight records at 5Hz, so each index represents 0.2 seconds
            time_offset = i * 0.2
            timestamps.append(time_offset)
            altitudes_agl.append(point.get('altitude_agl', 0))
            altitudes_msl.append(point.get('altitude_msl', 0))
            vertical_speeds.append(point.get('velocity_down', 0) * 2.23694)  # Convert to mph
            ground_speeds.append(point.get('ground_speed', 0) * 2.23694)  # Convert to mph
            headings.append(point.get('heading', 0))

        # Mark important indices as timestamps
        important_points = {}
        if self.flare_idx is not None and self.flare_idx < len(timestamps):
            important_points['flare'] = timestamps[self.flare_idx]
        if self.max_vspeed_idx is not None and self.max_vspeed_idx < len(timestamps):
            important_points['max_vspeed'] = timestamps[self.max_vspeed_idx]
        if self.max_gspeed_idx is not None and self.max_gspeed_idx < len(timestamps):
            important_points['max_gspeed'] = timestamps[self.max_gspeed_idx]
        if self.landing_idx is not None and self.landing_idx < len(timestamps):
            important_points['landing'] = timestamps[self.landing_idx]
        if self.rollout_start_idx is not None and self.rollout_start_idx < len(timestamps):
            important_points['rollout_start'] = timestamps[self.rollout_start_idx]

        return {
            'timestamps': timestamps,
            'altitude_agl': altitudes_agl,
            'altitude_msl': altitudes_msl,
            'vertical_speed': vertical_speeds,
            'ground_speed': ground_speeds,
            'heading': headings,
            'important_points': important_points
        }

    def _get_chart_data_legacy(self):
        """Legacy method using GPSPoint model - will be removed after migration"""
        import json

        # Get all GPS points for this flight, ordered by time
        points = self.gps_points.all().order_by('timestamp')

        if not points:
            return None

        # Convert to lists for JSON serialization
        timestamps = []
        altitudes_agl = []
        altitudes_msl = []
        vertical_speeds = []
        ground_speeds = []
        headings = []

        # Convert to seconds from start for x-axis using index position and 5Hz sampling rate
        for i, point in enumerate(points):
            # FlySight records at 5Hz, so each index represents 0.2 seconds
            time_offset = i * 0.2
            timestamps.append(time_offset)
            altitudes_agl.append(point.altitude_agl)
            altitudes_msl.append(point.altitude_msl)
            vertical_speeds.append(point.velocity_down * 2.23694)  # Convert to mph, positive down
            ground_speeds.append(point.ground_speed * 2.23694 if point.ground_speed else 0)  # Convert to mph
            headings.append(point.heading if point.heading else 0)

        # Mark important indices as timestamps
        important_points = {}
        if self.flare_idx is not None and self.flare_idx < len(timestamps):
            important_points['flare'] = timestamps[self.flare_idx]
        if self.max_vspeed_idx is not None and self.max_vspeed_idx < len(timestamps):
            important_points['max_vspeed'] = timestamps[self.max_vspeed_idx]
        if self.max_gspeed_idx is not None and self.max_gspeed_idx < len(timestamps):
            important_points['max_gspeed'] = timestamps[self.max_gspeed_idx]
        if self.landing_idx is not None and self.landing_idx < len(timestamps):
            important_points['landing'] = timestamps[self.landing_idx]
        if self.rollout_start_idx is not None and self.rollout_start_idx < len(timestamps):
            important_points['rollout_start'] = timestamps[self.rollout_start_idx]

        return {
            'timestamps': timestamps,
            'altitude_agl': altitudes_agl,
            'altitude_msl': altitudes_msl,
            'vertical_speed': vertical_speeds,
            'ground_speed': ground_speeds,
            'heading': headings,
            'important_points': important_points
        }

    def get_3d_visualization_data(self):
        """Get GPS data formatted for 3D visualization (side view, top view, map) - focused on swoop portion"""
        import math

        # Try JSON data first (new method), fallback to GPS points (legacy)
        gps_data = self.get_gps_data()

        if gps_data:
            # Use JSON data (preferred for performance)
            if self.is_swoop and self.flare_idx is not None and self.landing_idx is not None:
                # Get only the swoop portion (flare to landing)
                points_data = gps_data[self.flare_idx:self.landing_idx + 1]
                flare_offset = 0
                landing_offset = len(points_data) - 1
            else:
                # Use all points for non-swoop flights
                points_data = gps_data
                flare_offset = self.flare_idx
                landing_offset = self.landing_idx
        else:
            # Legacy fallback using GPS points
            points = self.gps_points.all().order_by('timestamp')
            if not points:
                return None

            if self.is_swoop and self.flare_idx is not None and self.landing_idx is not None:
                points = points[self.flare_idx:self.landing_idx + 1]
                flare_offset = 0
                landing_offset = len(points) - 1
            else:
                flare_offset = self.flare_idx
                landing_offset = self.landing_idx

            # Convert GPS points to data format
            points_data = []
            for point in points:
                points_data.append({
                    'lat': float(point.location.y),
                    'lon': float(point.location.x),
                    'altitude_agl': float(point.altitude_agl) if point.altitude_agl else 0
                })

        if not points_data:
            return None

        # Calculate cumulative distances and extract coordinates
        coordinates = []
        cumulative_distances = [0]  # Start at 0 distance
        altitudes_agl_ft = []

        prev_point = None
        total_distance = 0

        for point_data in points_data:
            lat = point_data['lat']
            lon = point_data['lon']
            alt_ft = point_data['altitude_agl'] * 3.28084  # Convert to feet

            coordinates.append([lat, lon])
            altitudes_agl_ft.append(alt_ft)

            if prev_point:
                # Calculate distance using Haversine formula
                lat1, lon1 = math.radians(prev_point['lat']), math.radians(prev_point['lon'])
                lat2, lon2 = math.radians(lat), math.radians(lon)

                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                distance_m = 6371000 * c  # Earth radius in meters
                distance_ft = distance_m * 3.28084  # Convert to feet

                total_distance += distance_ft
                cumulative_distances.append(total_distance)

            prev_point = point_data

        # Calculate center point for top view
        if coordinates:
            center_lat = sum(coord[0] for coord in coordinates) / len(coordinates)
            center_lon = sum(coord[1] for coord in coordinates) / len(coordinates)
            center_point = [center_lat, center_lon]

            # Calculate bounds
            min_lat = min(coord[0] for coord in coordinates)
            max_lat = max(coord[0] for coord in coordinates)
            min_lon = min(coord[1] for coord in coordinates)
            max_lon = max(coord[1] for coord in coordinates)
            bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        else:
            center_point = None
            bounds = None

        # Convert coordinates to relative XY for top view (centered at origin)
        top_view_coords = []
        if center_point:
            # Simple mercator-like projection for small areas
            for lat, lon in coordinates:
                # Convert to meters relative to center, then to feet
                x_m = (lon - center_point[1]) * 111320 * math.cos(math.radians(center_point[0]))
                y_m = (lat - center_point[0]) * 110540
                x_ft = x_m * 3.28084
                y_ft = y_m * 3.28084
                top_view_coords.append([x_ft, y_ft])

        # Mark important points with their indices (adjusted for swoop filtering)
        important_indices = {}
        if self.is_swoop and self.flare_idx is not None and self.landing_idx is not None:
            # For swoop data, adjust indices relative to the filtered range
            important_indices['flare'] = flare_offset
            important_indices['landing'] = landing_offset

            # Adjust other indices relative to the swoop start
            if self.max_vspeed_idx is not None and self.flare_idx <= self.max_vspeed_idx <= self.landing_idx:
                important_indices['max_vspeed'] = self.max_vspeed_idx - self.flare_idx
            if self.max_gspeed_idx is not None and self.flare_idx <= self.max_gspeed_idx <= self.landing_idx:
                important_indices['max_gspeed'] = self.max_gspeed_idx - self.flare_idx
            if self.rollout_start_idx is not None and self.flare_idx <= self.rollout_start_idx <= self.landing_idx:
                important_indices['rollout_start'] = self.rollout_start_idx - self.flare_idx
            if self.rollout_end_idx is not None and self.flare_idx <= self.rollout_end_idx <= self.landing_idx:
                important_indices['rollout_end'] = self.rollout_end_idx - self.flare_idx
        else:
            # For full flight data, use original indices
            if self.flare_idx is not None and self.flare_idx < len(coordinates):
                important_indices['flare'] = self.flare_idx
            if self.max_vspeed_idx is not None and self.max_vspeed_idx < len(coordinates):
                important_indices['max_vspeed'] = self.max_vspeed_idx
            if self.max_gspeed_idx is not None and self.max_gspeed_idx < len(coordinates):
                important_indices['max_gspeed'] = self.max_gspeed_idx
            if self.landing_idx is not None and self.landing_idx < len(coordinates):
                important_indices['landing'] = self.landing_idx
            if self.rollout_start_idx is not None and self.rollout_start_idx < len(coordinates):
                important_indices['rollout_start'] = self.rollout_start_idx
            if self.rollout_end_idx is not None and self.rollout_end_idx < len(coordinates):
                important_indices['rollout_end'] = self.rollout_end_idx

        return {
            'coordinates': coordinates,  # [[lat, lon], ...] for map view
            'top_view_coords': top_view_coords,  # [[x_ft, y_ft], ...] for top view
            'cumulative_distances': cumulative_distances,  # [0, d1, d2, ...] in feet
            'altitudes_agl_ft': altitudes_agl_ft,  # [alt1, alt2, ...] in feet
            'center_point': center_point,  # [lat, lon] for map centering
            'bounds': bounds,  # [[min_lat, min_lon], [max_lat, max_lon]]
            'important_indices': important_indices,  # {'flare': idx, ...}
            'total_distance_ft': total_distance  # Total flight path distance in feet
        }

    def calculate_swoop_accuracy_metrics(self):
        """Calculate average accuracy metrics during the swoop period"""
        if not self.is_swoop or not self.flare_idx or not self.landing_idx:
            return None, None, None

        # Try JSON data first (new method), fallback to GPS points (legacy)
        gps_data = self.get_gps_data()

        if gps_data and len(gps_data) > max(self.flare_idx, self.landing_idx):
            # JSON data method (preferred for performance)
            flare_to_landing = gps_data[self.flare_idx:self.landing_idx + 1]
            h_acc_values = [point.get('h_acc') for point in flare_to_landing if point.get('h_acc') is not None]
            v_acc_values = [point.get('v_acc') for point in flare_to_landing if point.get('v_acc') is not None]
            s_acc_values = [point.get('s_acc') for point in flare_to_landing if point.get('s_acc') is not None]
        else:
            # Legacy fallback using GPS points
            swoop_points = self.gps_points.all().order_by('timestamp')
            if not swoop_points.exists() or len(swoop_points) <= max(self.flare_idx, self.landing_idx):
                return None, None, None

            flare_to_landing = swoop_points[self.flare_idx:self.landing_idx + 1]
            h_acc_values = [point.horizontal_accuracy for point in flare_to_landing if point.horizontal_accuracy is not None]
            v_acc_values = [point.vertical_accuracy for point in flare_to_landing if point.vertical_accuracy is not None]
            s_acc_values = [point.speed_accuracy for point in flare_to_landing if point.speed_accuracy is not None]

        # Calculate averages
        avg_h_acc = sum(h_acc_values) / len(h_acc_values) if h_acc_values else None
        avg_v_acc = sum(v_acc_values) / len(v_acc_values) if v_acc_values else None
        avg_s_acc = sum(s_acc_values) / len(s_acc_values) if s_acc_values else None

        return avg_h_acc, avg_v_acc, avg_s_acc

    def calculate_and_store_swoop_distance(self):
        """Calculate and store swoop distance for efficient access"""
        if not self.is_swoop or not self.rollout_end_idx or not self.landing_idx:
            self.swoop_distance_ft = None
            return None

        try:
            # Try JSON data first (new method), fallback to GPS points (legacy)
            gps_data = self.get_gps_data()

            if gps_data and len(gps_data) > max(self.rollout_end_idx, self.landing_idx):
                # JSON data method (preferred for performance)
                rollout_end_point = gps_data[self.rollout_end_idx]
                landing_point = gps_data[self.landing_idx]

                # Calculate distance using Haversine formula
                import math
                lat1, lon1 = math.radians(rollout_end_point['lat']), math.radians(rollout_end_point['lon'])
                lat2, lon2 = math.radians(landing_point['lat']), math.radians(landing_point['lon'])

                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                distance_m = 6371000 * c  # Earth radius in meters
                distance_ft = distance_m * 3.28084  # Convert to feet
            else:
                # Legacy fallback using GPS points
                gps_points = list(self.gps_points.order_by('timestamp'))
                if not gps_points or len(gps_points) <= max(self.rollout_end_idx, self.landing_idx):
                    self.swoop_distance_ft = None
                    return None

                rollout_end_point = gps_points[self.rollout_end_idx]
                landing_point = gps_points[self.landing_idx]

                # Calculate distance between rollout end and landing
                distance_m = rollout_end_point.location.distance(landing_point.location) * 111000  # Convert degrees to meters
                distance_ft = distance_m * 3.28084  # Convert to feet

            self.swoop_distance_ft = distance_ft
            return distance_ft
        except (IndexError, TypeError, KeyError):
            self.swoop_distance_ft = None
            return None

    def store_gps_data(self, gps_data_list):
        """
        Store GPS data as compressed JSON
        gps_data_list: List of dicts with GPS point data
        """
        import hashlib

        # Convert to compact format
        compact_data = {
            'points': gps_data_list,
            'count': len(gps_data_list),
            'version': '1.0'
        }

        # Compress and encode
        json_str = json.dumps(compact_data, separators=(',', ':'))
        compressed = gzip.compress(json_str.encode('utf-8'))
        encoded = base64.b64encode(compressed).decode('ascii')

        # Store with hash for integrity
        self.gps_data_compressed = encoded
        self.gps_data_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def get_gps_data(self):
        """
        Retrieve and decompress GPS data
        Returns: List of GPS point dicts or None if no data
        """
        if not self.gps_data_compressed:
            return None

        try:
            # Decode and decompress
            compressed = base64.b64decode(self.gps_data_compressed.encode('ascii'))
            json_str = gzip.decompress(compressed).decode('utf-8')
            data = json.loads(json_str)

            # Verify integrity if hash exists
            if self.gps_data_hash:
                import hashlib
                computed_hash = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
                if computed_hash != self.gps_data_hash:
                    raise ValueError("GPS data integrity check failed")

            return data.get('points', [])
        except Exception as e:
            print(f"Error retrieving GPS data for flight {self.id}: {e}")
            return None

    def get_gps_points_legacy(self):
        """
        Fallback method to get GPS points from old GPSPoint model
        This will be removed after migration
        """
        return self.gps_points.all().order_by('timestamp')

    # Unit conversion properties
    def get_max_vertical_speed(self, units='imperial'):
        """Get max vertical speed in requested units"""
        if self.max_vertical_speed_ms is None:
            return None
        return format_speed(self.max_vertical_speed_ms, units)

    def get_max_ground_speed(self, units='imperial'):
        """Get max ground speed in requested units"""
        if self.max_ground_speed_ms is None:
            return None
        return format_speed(self.max_ground_speed_ms, units)

    def get_entry_gate_speed(self, units='imperial'):
        """Get entry gate speed in requested units"""
        if self.entry_gate_speed_mps is None:
            return None
        return format_speed(self.entry_gate_speed_mps, units)

    def get_swoop_distance(self, units='imperial'):
        """Get swoop distance in requested units"""
        if self.swoop_distance_m is None:
            return None
        return format_distance(self.swoop_distance_m, units)

    def get_altitude_display(self, altitude_meters, units='imperial'):
        """Get altitude in requested units"""
        return format_altitude(altitude_meters, units)

    def update_accuracy_metrics(self):
        """Update the stored accuracy metrics for this flight"""
        if self.is_swoop:
            h_acc, v_acc, s_acc = self.calculate_swoop_accuracy_metrics()
            self.swoop_avg_horizontal_accuracy = h_acc
            self.swoop_avg_vertical_accuracy = v_acc
            self.swoop_avg_speed_accuracy = s_acc
            self.save()


class GPSPoint(models.Model):
    flight = models.ForeignKey(Flight, on_delete=models.CASCADE, related_name='gps_points')
    timestamp = models.DateTimeField()
    location = models.PointField()  # PostGIS Point field (lat, lon)
    altitude_msl = models.FloatField()  # Height above mean sea level (hMSL)
    altitude_agl = models.FloatField(null=True, blank=True)  # Height above ground level (computed)

    # Velocity components (m/s)
    velocity_north = models.FloatField()
    velocity_east = models.FloatField()
    velocity_down = models.FloatField()

    # Computed fields
    ground_speed = models.FloatField(null=True, blank=True)  # 2D ground speed (m/s)
    heading = models.FloatField(null=True, blank=True)  # Heading in degrees (0-360)

    # Accuracy measurements (meters)
    horizontal_accuracy = models.FloatField(null=True, blank=True)
    vertical_accuracy = models.FloatField(null=True, blank=True)
    speed_accuracy = models.FloatField(null=True, blank=True)

    # GPS metadata
    num_satellites = models.IntegerField()

    class Meta:
        indexes = [
            models.Index(fields=['timestamp']),
            models.Index(fields=['flight', 'timestamp']),
            GistIndex(fields=['location']),  # Spatial index for location queries
        ]
        ordering = ['timestamp']

    def __str__(self):
        return f"GPS Point at {self.timestamp} - {self.location}"

    @property
    def total_speed(self):
        """Calculate 3D speed from all velocity components"""
        return (self.velocity_north ** 2 + self.velocity_east ** 2 + self.velocity_down ** 2) ** 0.5

    def calculate_ground_speed(self):
        """Calculate 2D ground speed from velocity components"""
        return (self.velocity_north ** 2 + self.velocity_east ** 2) ** 0.5

    def calculate_heading(self):
        """Calculate heading from velocity components"""
        import numpy as np
        return (np.degrees(np.arctan2(self.velocity_east, self.velocity_north)) + 360) % 360
