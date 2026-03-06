
from django.contrib.gis.db import models
from django.contrib.gis.geos import Point
from django.contrib.postgres.indexes import GistIndex
from django.contrib.auth.models import User
import json
import gzip
import base64
from .units import *


class CompetitionGate(models.Model):
    """Model to store competition gate files and parsed gate positions"""
    GATE_TYPE_CHOICES = [
        ('standard', 'Standard Gates (Distance/Zone Accuracy)'),
        ('speed', 'Speed Gates'),
    ]

    # Metadata
    name = models.CharField(max_length=200, help_text="Name of the competition location or event")
    gate_type = models.CharField(max_length=20, choices=GATE_TYPE_CHOICES, help_text="Type of gates")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='competition_gates')

    # File storage
    gate_file = models.FileField(upload_to='gate_files/', help_text="Upload .gsw or .CSV gate file")

    # Parsed gate positions (stored as JSON)
    # Format: {'gate_1_inside': {lat, lon, alt}, 'gate_2_outside': {lat, lon, alt}, 'gate_5_center': {lat, lon, alt}}
    gate_positions = models.JSONField(
        null=True, blank=True,
        help_text="Parsed gate positions with lat/lon/altitude"
    )

    # Course configuration (stored as JSON)
    course_config = models.JSONField(
        null=True, blank=True,
        help_text="Complete course geometry based on USPA rules"
    )

    # Location information
    center_lat = models.FloatField(null=True, blank=True, help_text="Center latitude for map display")
    center_lon = models.FloatField(null=True, blank=True, help_text="Center longitude for map display")

    # Status
    is_parsed = models.BooleanField(default=False, help_text="Whether the gate file has been successfully parsed")
    parse_error = models.TextField(blank=True, help_text="Error message if parsing failed")

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['gate_type']),
            models.Index(fields=['created_by']),
            models.Index(fields=['is_parsed']),
        ]

    def __str__(self):
        return f"{self.name} ({self.get_gate_type_display()})"

    def parse_gate_file(self):
        """Parse the uploaded gate file to identify gate positions from cross pattern"""
        from .utils.gate_parser import GateFileParser

        try:
            parser = GateFileParser(self.gate_file.path, self.gate_type)
            self.gate_positions = parser.extract_gate_positions()
            self.is_parsed = True
            self.parse_error = ""

            # Calculate center point for map
            if self.gate_positions and 'gate_1_inside' in self.gate_positions:
                gate_1 = self.gate_positions['gate_1_inside']
                self.center_lat = gate_1['lat']
                self.center_lon = gate_1['lon']

            # Generate course configuration
            self.generate_course_config()
            self.save()

            return True
        except Exception as e:
            self.is_parsed = False
            self.parse_error = str(e)
            self.save()
            return False

    def generate_course_config(self):
        """Generate course geometry based on USPA competition rules"""
        from .utils.course_builder import CourseBuilder

        if not self.gate_positions:
            return

        builder = CourseBuilder(self.gate_positions, self.gate_type)
        self.course_config = builder.build_course()
        self.save()

    def get_entry_gate_inside(self):
        """Get the inside entry gate (gate 1) position"""
        if self.gate_positions and 'gate_1_inside' in self.gate_positions:
            return self.gate_positions['gate_1_inside']
        return None

    def get_entry_gate_outside(self):
        """Get the outside entry gate (gate 2) position"""
        if self.gate_positions and 'gate_2_outside' in self.gate_positions:
            return self.gate_positions['gate_2_outside']
        return None

    def get_exit_gate_center(self):
        """Get the exit gate (gate 5) center position for speed course"""
        if self.gate_type == 'speed' and self.gate_positions and 'gate_5_center' in self.gate_positions:
            return self.gate_positions['gate_5_center']
        return None


class Flight(models.Model):
    # User association
    pilot = models.ForeignKey(User, on_delete=models.CASCADE, related_name='flights', null=True, blank=True)
    canopy = models.ForeignKey('users.Canopy', on_delete=models.SET_NULL, null=True, blank=True, related_name='flights')
    competition_gate = models.ForeignKey(CompetitionGate, on_delete=models.SET_NULL, null=True, blank=True, related_name='flights', help_text="Competition gate/course for this flight")

    # Basic flight metadata
    device_id = models.CharField(max_length=50)
    session_id = models.CharField(max_length=50)
    firmware_version = models.CharField(max_length=20, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    filename = models.CharField(max_length=255, blank=True)
    flight_name = models.CharField(max_length=255, blank=True, help_text="User-friendly flight name")

    # Flight location and conditions (optional)
    location_name = models.CharField(max_length=100, blank=True, help_text="Drop zone or location name")
    weather_conditions = models.TextField(blank=True)
    notes = models.TextField(blank=True, help_text="Pilot notes about the jump")

    # Competition gate metrics (calculated if competition_gate is set)
    gate_crossing_idx = models.IntegerField(null=True, blank=True, help_text="Index where flight crossed entry gates")
    gate_speed_mps = models.FloatField(null=True, blank=True, help_text="Speed at gate crossing in m/s")
    gate_altitude_agl = models.FloatField(null=True, blank=True, help_text="Altitude at gate crossing in meters AGL")
    passed_between_gates = models.BooleanField(null=True, blank=True, help_text="Whether flight passed cleanly between gates")

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
    turn_rotation_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score for rotation calculation (0.0-1.0)")
    turn_rotation_method = models.CharField(
        max_length=20,
        null=True, blank=True,
        help_text="Method used for rotation calculation (raw, smoothed, dominant, fallback)"
    )
    intended_turn = models.IntegerField(
        null=True, blank=True,
        choices=[(90, '90°'), (270, '270°'), (450, '450°'), (630, '630°'), (810, '810°'), (990, '990°')],
        help_text="Intended standard turn degree classification"
    )
    turn_direction = models.CharField(
        max_length=5,
        choices=[('left', 'Left'), ('right', 'Right')],
        null=True, blank=True
    )

    # gswoop-style turn segment metrics (turn initiation to rollout end)
    turn_segment_rotation = models.FloatField(null=True, blank=True, help_text="gswoop-style turn segment rotation in degrees")
    turn_segment_confidence = models.FloatField(null=True, blank=True, help_text="Confidence score for gswoop-style rotation (0.0-1.0)")
    turn_segment_method = models.CharField(
        max_length=30,
        null=True, blank=True,
        help_text="Method used for gswoop-style rotation calculation"
    )
    turn_segment_intended = models.IntegerField(
        null=True, blank=True,
        choices=[(90, '90°'), (270, '270°'), (450, '450°'), (630, '630°'), (810, '810°'), (990, '990°')],
        help_text="Intended turn classification for gswoop-style segment"
    )
    turn_segment_start_alt = models.FloatField(null=True, blank=True, help_text="Turn segment start altitude in ft AGL")
    turn_segment_end_alt = models.FloatField(null=True, blank=True, help_text="Turn segment end altitude in ft AGL")
    turn_segment_duration = models.FloatField(null=True, blank=True, help_text="Turn segment duration in seconds")
    gswoop_difference = models.FloatField(null=True, blank=True, help_text="Difference from gswoop published rotation in degrees")

    # ML-enhanced rotation metrics
    ml_rotation = models.FloatField(null=True, blank=True, help_text="ML-predicted rotation in degrees")
    ml_rotation_confidence = models.FloatField(null=True, blank=True, help_text="ML prediction confidence (0.0-1.0)")
    ml_rotation_method = models.CharField(
        max_length=20,
        null=True, blank=True,
        help_text="ML prediction method (ml_enhanced, fallback)"
    )
    ml_intended_turn = models.IntegerField(
        null=True, blank=True,
        choices=[(90, '90°'), (270, '270°'), (450, '450°'), (630, '630°'), (810, '810°'), (990, '990°')],
        help_text="ML-predicted intended turn classification"
    )

    # Multi-metric ML prediction fields
    # Timing predictions
    ml_turn_time = models.FloatField(null=True, blank=True, help_text="ML-predicted time to execute turn (seconds)")
    ml_turn_time_confidence = models.FloatField(null=True, blank=True, help_text="Turn time prediction confidence")
    ml_rollout_time = models.FloatField(null=True, blank=True, help_text="ML-predicted rollout duration (seconds)")
    ml_rollout_time_confidence = models.FloatField(null=True, blank=True, help_text="Rollout time prediction confidence")
    ml_swoop_time = models.FloatField(null=True, blank=True, help_text="ML-predicted time aloft during swoop (seconds)")
    ml_swoop_time_confidence = models.FloatField(null=True, blank=True, help_text="Swoop time prediction confidence")

    # Distance predictions
    ml_distance_to_stop = models.FloatField(null=True, blank=True, help_text="ML-predicted distance to stop (feet)")
    ml_distance_confidence = models.FloatField(null=True, blank=True, help_text="Distance prediction confidence")
    ml_touchdown_distance = models.FloatField(null=True, blank=True, help_text="ML-predicted touchdown distance (feet)")
    ml_touchdown_confidence = models.FloatField(null=True, blank=True, help_text="Touchdown distance prediction confidence")
    ml_touchdown_speed = models.FloatField(null=True, blank=True, help_text="ML-predicted touchdown speed (mph)")
    ml_touchdown_speed_confidence = models.FloatField(null=True, blank=True, help_text="Touchdown speed prediction confidence")

    # Gate and positioning predictions
    ml_entry_speed = models.FloatField(null=True, blank=True, help_text="ML-predicted entry gate speed (mph)")
    ml_entry_speed_confidence = models.FloatField(null=True, blank=True, help_text="Entry speed prediction confidence")
    ml_turn_init_back = models.FloatField(null=True, blank=True, help_text="ML-predicted turn initiation back position (feet)")
    ml_turn_init_offset = models.FloatField(null=True, blank=True, help_text="ML-predicted turn initiation offset position (feet)")

    # Metadata
    ml_predictions_count = models.IntegerField(null=True, blank=True, help_text="Number of ML predictions available")
    ml_predictions_updated_at = models.DateTimeField(null=True, blank=True, help_text="When ML predictions were last updated")

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
        # Use flight_name if available, otherwise fallback to old format
        if self.flight_name:
            pilot_name = self.pilot.username if self.pilot else "Unknown"
            return f"{pilot_name}: {self.flight_name}"

        swoop_info = f" - {self.turn_rotation:.0f}° {self.turn_direction}" if self.is_swoop else ""
        pilot_name = self.pilot.username if self.pilot else "Unknown"
        return f"{pilot_name}: {self.session_id[:8]}{swoop_info}"

    def get_flight_datetime(self):
        """Get the actual flight datetime from GPS data"""
        gps_data = self.get_gps_data()
        if gps_data and len(gps_data) > 0:
            first_point = gps_data[0]
            if 'timestamp' in first_point:
                import datetime
                return datetime.datetime.fromtimestamp(first_point['timestamp'])

        # Fallback to created_at if no GPS data
        return self.created_at

    def generate_chronological_name(self):
        """Generate chronological flight name based on pilot's flights that day"""
        if not self.pilot:
            return f"{self.filename}"  # Fallback for flights without pilot

        flight_datetime = self.get_flight_datetime()
        flight_date = flight_datetime.date()

        # Get all flights by this pilot on the same day, ordered chronologically
        same_day_flights = Flight.objects.filter(
            pilot=self.pilot
        ).exclude(
            id=self.id  # Exclude current flight from count
        )

        # Filter by flight date using GPS timestamps
        same_day_flights_list = []
        for flight in same_day_flights:
            other_flight_datetime = flight.get_flight_datetime()
            if other_flight_datetime.date() == flight_date:
                same_day_flights_list.append((flight, other_flight_datetime))

        # Sort by flight datetime
        same_day_flights_list.sort(key=lambda x: x[1])

        # Find position of current flight chronologically
        flight_number = 1
        for other_flight, other_datetime in same_day_flights_list:
            if other_datetime < flight_datetime:
                flight_number += 1
            else:
                break

        # Format: "Sep 21 2025 - Flight #1"
        formatted_date = flight_datetime.strftime("%b %-d %Y")
        return f"{formatted_date} - Flight #{flight_number}"

    def update_chronological_name(self, save=True):
        """Update this flight's name to chronological format"""
        self.flight_name = self.generate_chronological_name()
        if save:
            self.save(update_fields=['flight_name'])

    @classmethod
    def find_potential_duplicates(cls, pilot, gps_data, tolerance_seconds=30):
        """Find potential duplicate flights based on GPS data characteristics"""
        if not pilot or not gps_data or len(gps_data) == 0:
            return []

        first_point = gps_data[0]
        last_point = gps_data[-1]

        # Extract key characteristics for comparison
        start_time = first_point.get('timestamp')
        if not start_time:
            return []

        import datetime
        flight_datetime = datetime.datetime.fromtimestamp(start_time)
        flight_duration = last_point.get('timestamp', start_time) - start_time

        # Get approximate flight coordinates (first and last points)
        start_lat = first_point.get('lat')
        start_lon = first_point.get('lon')
        end_lat = last_point.get('lat')
        end_lon = last_point.get('lon')

        if None in [start_lat, start_lon, end_lat, end_lon]:
            return []

        # Find flights from the same pilot within a reasonable time window
        time_window_start = flight_datetime - datetime.timedelta(seconds=tolerance_seconds)
        time_window_end = flight_datetime + datetime.timedelta(seconds=tolerance_seconds)

        potential_duplicates = []
        existing_flights = cls.objects.filter(pilot=pilot)

        for flight in existing_flights:
            existing_gps_data = flight.get_gps_data()
            if not existing_gps_data or len(existing_gps_data) == 0:
                continue

            existing_first = existing_gps_data[0]
            existing_last = existing_gps_data[-1]

            existing_start_time = existing_first.get('timestamp')
            if not existing_start_time:
                continue

            existing_flight_datetime = datetime.datetime.fromtimestamp(existing_start_time)
            existing_duration = existing_last.get('timestamp', existing_start_time) - existing_start_time

            # Check if times are close
            if time_window_start <= existing_flight_datetime <= time_window_end:
                # Check if durations are similar (within 10% or 10 seconds)
                duration_diff = abs(flight_duration - existing_duration)
                duration_tolerance = max(10, min(flight_duration, existing_duration) * 0.1)

                if duration_diff <= duration_tolerance:
                    # Check if coordinates are close (within ~100 meters)
                    import math
                    start_distance = cls._calculate_distance(
                        start_lat, start_lon,
                        existing_first.get('lat', 0), existing_first.get('lon', 0)
                    )
                    end_distance = cls._calculate_distance(
                        end_lat, end_lon,
                        existing_last.get('lat', 0), existing_last.get('lon', 0)
                    )

                    if start_distance <= 100 and end_distance <= 100:  # 100 meters tolerance
                        potential_duplicates.append({
                            'flight': flight,
                            'confidence': cls._calculate_duplicate_confidence(
                                flight_duration, existing_duration,
                                start_distance, end_distance,
                                abs((flight_datetime - existing_flight_datetime).total_seconds())
                            )
                        })

        # Sort by confidence (highest first)
        potential_duplicates.sort(key=lambda x: x['confidence'], reverse=True)
        return potential_duplicates

    @staticmethod
    def _calculate_distance(lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates in meters"""
        import math
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        return r * c

    @staticmethod
    def _calculate_duplicate_confidence(duration1, duration2, start_distance, end_distance, time_diff):
        """Calculate confidence that two flights are duplicates (0-100%)"""
        # Start with 100% confidence
        confidence = 100.0

        # Reduce confidence based on duration difference
        duration_diff_ratio = abs(duration1 - duration2) / max(duration1, duration2)
        confidence -= duration_diff_ratio * 30  # Up to 30% penalty

        # Reduce confidence based on coordinate distance
        avg_distance = (start_distance + end_distance) / 2
        distance_penalty = min(30, avg_distance / 100 * 30)  # Up to 30% penalty
        confidence -= distance_penalty

        # Reduce confidence based on time difference
        time_penalty = min(20, time_diff / 30 * 20)  # Up to 20% penalty for 30+ second difference
        confidence -= time_penalty

        return max(0, confidence)

    @classmethod
    def resequence_pilot_flights(cls, pilot, target_date=None):
        """Resequence all flights for a pilot on a given date (or all dates if None)"""
        if not pilot:
            return

        # Get all flights for this pilot
        flights = cls.objects.filter(pilot=pilot)

        if target_date:
            # Filter by specific date using GPS timestamps
            flights_on_date = []
            for flight in flights:
                flight_datetime = flight.get_flight_datetime()
                if flight_datetime.date() == target_date:
                    flights_on_date.append((flight, flight_datetime))

            # Sort by flight datetime and update names
            flights_on_date.sort(key=lambda x: x[1])
            for i, (flight, _) in enumerate(flights_on_date, 1):
                formatted_date = target_date.strftime("%b %-d %Y")
                new_name = f"{formatted_date} - Flight #{i}"
                if flight.flight_name != new_name:
                    flight.flight_name = new_name
                    flight.save(update_fields=['flight_name'])
        else:
            # Resequence all flights by grouping by date
            flights_by_date = {}
            for flight in flights:
                flight_datetime = flight.get_flight_datetime()
                date_key = flight_datetime.date()
                if date_key not in flights_by_date:
                    flights_by_date[date_key] = []
                flights_by_date[date_key].append((flight, flight_datetime))

            # Resequence each date
            for date_key, date_flights in flights_by_date.items():
                date_flights.sort(key=lambda x: x[1])
                for i, (flight, _) in enumerate(date_flights, 1):
                    formatted_date = date_key.strftime("%b %-d %Y")
                    new_name = f"{formatted_date} - Flight #{i}"
                    if flight.flight_name != new_name:
                        flight.flight_name = new_name
                        flight.save(update_fields=['flight_name'])

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
        """Get GPS data formatted for FlySight-style time series viewer. All speeds in m/s, altitudes in m."""
        import math

        gps_data = self.get_gps_data()
        if not gps_data:
            return self._get_chart_data_legacy()
        if not gps_data:
            return None

        G = 9.81  # m/s²
        timestamps = []
        altitude_agl = []
        vertical_speed = []
        ground_speed = []
        total_speed = []
        dive_angle = []
        glide_ratio = []
        specific_energy = []

        for i, point in enumerate(gps_data):
            gs = float(point.get('ground_speed', 0) or 0)   # m/s horizontal
            vd = float(point.get('velocity_down', 0) or 0)  # m/s, positive = descending
            alt = float(point.get('altitude_agl', 0) or 0)  # meters AGL

            ts = math.sqrt(gs * gs + vd * vd)  # total 3D speed m/s

            if gs > 0.1:
                da = math.degrees(math.atan2(abs(vd), gs))
            elif abs(vd) > 0.1:
                da = 90.0
            else:
                da = 0.0

            if abs(vd) > 0.1:
                gr = min(30.0, max(-30.0, gs / abs(vd)))
            else:
                gr = 30.0  # effectively infinite when nearly level

            se = G * alt + 0.5 * ts * ts  # J/kg specific mechanical energy

            timestamps.append(round(i * 0.2, 1))
            altitude_agl.append(round(alt, 2))
            vertical_speed.append(round(vd, 3))
            ground_speed.append(round(gs, 3))
            total_speed.append(round(ts, 3))
            dive_angle.append(round(da, 2))
            glide_ratio.append(round(gr, 2))
            specific_energy.append(round(se, 1))

        n = len(timestamps)
        important_indices = {}
        for attr, key in [
            ('flare_idx', 'flare'),
            ('max_vspeed_idx', 'max_vspeed'),
            ('max_gspeed_idx', 'max_gspeed'),
            ('landing_idx', 'landing'),
            ('rollout_start_idx', 'rollout_start'),
            ('rollout_end_idx', 'rollout_end'),
        ]:
            val = getattr(self, attr)
            if val is not None and val < n:
                important_indices[key] = val

        return {
            'timestamps': timestamps,
            'altitude_agl': altitude_agl,
            'vertical_speed': vertical_speed,
            'ground_speed': ground_speed,
            'total_speed': total_speed,
            'dive_angle': dive_angle,
            'glide_ratio': glide_ratio,
            'specific_energy': specific_energy,
            'important_indices': important_indices,
        }

    def _get_chart_data_legacy(self):
        """Legacy fallback using GPSPoint model."""
        import math

        points = list(self.gps_points.all().order_by('timestamp'))
        if not points:
            return None

        G = 9.81
        timestamps = []
        altitude_agl = []
        vertical_speed = []
        ground_speed = []
        total_speed = []
        dive_angle = []
        glide_ratio = []
        specific_energy = []

        for i, point in enumerate(points):
            gs = float(point.ground_speed or 0)
            vd = float(point.velocity_down or 0)
            alt = float(point.altitude_agl or 0)
            ts = math.sqrt(gs * gs + vd * vd)

            if gs > 0.1:
                da = math.degrees(math.atan2(abs(vd), gs))
            elif abs(vd) > 0.1:
                da = 90.0
            else:
                da = 0.0

            gr = min(30.0, max(-30.0, gs / abs(vd))) if abs(vd) > 0.1 else 30.0
            se = G * alt + 0.5 * ts * ts

            timestamps.append(round(i * 0.2, 1))
            altitude_agl.append(round(alt, 2))
            vertical_speed.append(round(vd, 3))
            ground_speed.append(round(gs, 3))
            total_speed.append(round(ts, 3))
            dive_angle.append(round(da, 2))
            glide_ratio.append(round(gr, 2))
            specific_energy.append(round(se, 1))

        n = len(timestamps)
        important_indices = {}
        for attr, key in [
            ('flare_idx', 'flare'),
            ('max_vspeed_idx', 'max_vspeed'),
            ('max_gspeed_idx', 'max_gspeed'),
            ('landing_idx', 'landing'),
            ('rollout_start_idx', 'rollout_start'),
            ('rollout_end_idx', 'rollout_end'),
        ]:
            val = getattr(self, attr)
            if val is not None and val < n:
                important_indices[key] = val

        return {
            'timestamps': timestamps,
            'altitude_agl': altitude_agl,
            'vertical_speed': vertical_speed,
            'ground_speed': ground_speed,
            'total_speed': total_speed,
            'dive_angle': dive_angle,
            'glide_ratio': glide_ratio,
            'specific_energy': specific_energy,
            'important_indices': important_indices,
        }

    def get_3d_visualization_data(self):
        """Get full-flight GPS data for side view, top view, and map, with swoop window marked."""
        import math

        gps_data = self.get_gps_data()

        if gps_data:
            points_data = gps_data
        else:
            points = list(self.gps_points.all().order_by('timestamp'))
            if not points:
                return None
            points_data = [{
                'lat': float(p.location.y),
                'lon': float(p.location.x),
                'altitude_agl': float(p.altitude_agl) if p.altitude_agl else 0,
            } for p in points]

        if not points_data:
            return None

        coordinates = []
        cumulative_distances = []
        altitude_agl = []
        total_distance = 0
        prev_lat = prev_lon = None

        for point in points_data:
            lat = float(point.get('lat', 0))
            lon = float(point.get('lon', 0))
            alt = float(point.get('altitude_agl', 0))

            if prev_lat is not None:
                lat1r, lon1r = math.radians(prev_lat), math.radians(prev_lon)
                lat2r, lon2r = math.radians(lat), math.radians(lon)
                dlat, dlon = lat2r - lat1r, lon2r - lon1r
                a = math.sin(dlat/2)**2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon/2)**2
                total_distance += 6371000 * 2 * math.asin(math.sqrt(a))

            coordinates.append([lat, lon])
            cumulative_distances.append(round(total_distance, 1))
            altitude_agl.append(round(alt, 2))
            prev_lat, prev_lon = lat, lon

        if not coordinates:
            return None

        lats = [c[0] for c in coordinates]
        lons = [c[1] for c in coordinates]
        center_point = [sum(lats) / len(lats), sum(lons) / len(lons)]
        full_bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]

        n = len(coordinates)
        swoop_start_idx = None
        swoop_end_idx = None
        swoop_bounds = None
        if self.is_swoop and self.flare_idx is not None and self.landing_idx is not None:
            swoop_start_idx = min(self.flare_idx, n - 1)
            swoop_end_idx = min(self.landing_idx, n - 1)
            sc = coordinates[swoop_start_idx:swoop_end_idx + 1]
            if sc:
                swoop_bounds = [[min(c[0] for c in sc), min(c[1] for c in sc)],
                                [max(c[0] for c in sc), max(c[1] for c in sc)]]

        important_indices = {}
        for attr, key in [
            ('flare_idx', 'flare'),
            ('max_vspeed_idx', 'max_vspeed'),
            ('max_gspeed_idx', 'max_gspeed'),
            ('landing_idx', 'landing'),
            ('rollout_start_idx', 'rollout_start'),
            ('rollout_end_idx', 'rollout_end'),
        ]:
            val = getattr(self, attr)
            if val is not None and val < n:
                important_indices[key] = val

        return {
            'coordinates': coordinates,
            'cumulative_distances': cumulative_distances,  # meters
            'altitude_agl': altitude_agl,                 # meters
            'center_point': center_point,
            'full_bounds': full_bounds,
            'swoop_bounds': swoop_bounds,
            'swoop_start_idx': swoop_start_idx,
            'swoop_end_idx': swoop_end_idx,
            'important_indices': important_indices,
            'total_distance_m': round(total_distance, 1),
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
        self.save()

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

    def calculate_gate_metrics(self):
        """Calculate and store competition gate crossing metrics"""
        if not self.competition_gate or not self.competition_gate.is_parsed:
            return False

        from .utils.gate_calculator import GateCalculator

        try:
            gps_data = self.get_gps_data()
            if not gps_data:
                return False

            calculator = GateCalculator(self.competition_gate.gate_positions)
            metrics = calculator.calculate_entry_gate_metrics(gps_data)

            if metrics:
                self.gate_crossing_idx = metrics['crossing_idx']
                self.gate_speed_mps = metrics['speed_mps']
                self.gate_altitude_agl = metrics['altitude_agl']
                self.passed_between_gates = metrics['passed_between_gates']
                self.save(update_fields=[
                    'gate_crossing_idx', 'gate_speed_mps',
                    'gate_altitude_agl', 'passed_between_gates'
                ])
                return True

        except Exception as e:
            print(f"Error calculating gate metrics for flight {self.id}: {e}")

        return False


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
