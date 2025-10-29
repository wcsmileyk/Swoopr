"""
Gate file parser for competition swoop courses.

Parses .gsw/.CSV files containing GPS tracks where the person stood still
at cross pattern points to mark gate positions.
"""

import csv
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from sklearn.cluster import DBSCAN


class GateFileParser:
    """Parse gate files to extract gate positions from cross pattern"""

    def __init__(self, file_path: str, gate_type: str):
        """
        Initialize parser with gate file.

        Args:
            file_path: Path to .gsw or .CSV gate file
            gate_type: 'standard' or 'speed'
        """
        self.file_path = file_path
        self.gate_type = gate_type
        self.gps_points = []

    def extract_gate_positions(self) -> Dict:
        """
        Extract gate positions from the gate file.

        Returns:
            Dict with gate positions:
            {
                'gate_1_inside': {'lat': float, 'lon': float, 'alt': float},
                'gate_2_outside': {'lat': float, 'lon': float, 'alt': float},
                'gate_5_center': {'lat': float, 'lon': float, 'alt': float}  # speed only
            }
        """
        # Read GPS points from file
        self._read_gps_file()

        # Find stationary clusters (45 second stands)
        clusters = self._find_stationary_clusters()

        if self.gate_type == 'standard':
            if len(clusters) != 4:
                raise ValueError(f"Expected 4 stationary clusters for standard gates, found {len(clusters)}")
        elif self.gate_type == 'speed':
            if len(clusters) != 5:
                raise ValueError(f"Expected 5 stationary clusters for speed gates, found {len(clusters)}")

        # Calculate gate positions from cross pattern
        return self._calculate_gate_positions(clusters)

    def _read_gps_file(self):
        """Read GPS points from CSV file"""
        with open(self.file_path, 'r') as f:
            # Check if it's FlySight v2 format or legacy CSV
            first_line = f.readline().strip()
            f.seek(0)

            if first_line.startswith('$FLYS') or first_line.startswith('time,lat,lon'):
                # New FlySight v2 or legacy CSV format
                self._read_flysight_format(f)
            else:
                raise ValueError("Unknown gate file format")

    def _read_flysight_format(self, f):
        """Read FlySight CSV format"""
        reader = csv.reader(f)

        # Skip header lines
        for row in reader:
            if not row or row[0].startswith('$'):
                continue
            if row[0] == 'time' or row[0].startswith('('):
                continue

            try:
                # Parse GPS data
                if row[0].startswith('$GNSS'):
                    # FlySight v2 format: $GNSS,time,lat,lon,hMSL,...
                    timestamp_str = row[1]
                    lat = float(row[2])
                    lon = float(row[3])
                    alt = float(row[4])
                else:
                    # Legacy CSV format: time,lat,lon,hMSL,...
                    timestamp_str = row[0]
                    lat = float(row[1])
                    lon = float(row[2])
                    alt = float(row[3])

                # Parse timestamp
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

                self.gps_points.append({
                    'timestamp': timestamp,
                    'lat': lat,
                    'lon': lon,
                    'alt': alt
                })
            except (ValueError, IndexError) as e:
                # Skip malformed rows
                continue

    def _find_stationary_clusters(self) -> List[Dict]:
        """
        Find clusters of stationary GPS points (45 second stands).

        Returns:
            List of cluster centroids with their positions
        """
        if len(self.gps_points) < 100:
            raise ValueError("Insufficient GPS points in gate file")

        # Calculate velocities between consecutive points
        velocities = []
        for i in range(1, len(self.gps_points)):
            p1 = self.gps_points[i - 1]
            p2 = self.gps_points[i]

            # Time difference in seconds
            dt = (p2['timestamp'] - p1['timestamp']).total_seconds()
            if dt <= 0:
                continue

            # Distance in meters using Haversine
            dist = self._haversine_distance(
                p1['lat'], p1['lon'],
                p2['lat'], p2['lon']
            )

            # Velocity in m/s
            velocity = dist / dt if dt > 0 else 0
            velocities.append(velocity)

        # Identify stationary points (velocity < 0.5 m/s)
        stationary_indices = [i for i, v in enumerate(velocities) if v < 0.5]

        if len(stationary_indices) < 100:
            raise ValueError("Insufficient stationary GPS points found")

        # Extract coordinates of stationary points
        stationary_coords = np.array([
            [self.gps_points[i]['lat'], self.gps_points[i]['lon']]
            for i in stationary_indices
        ])

        # Cluster stationary points using DBSCAN
        # eps is in degrees, roughly 0.0001 degrees ≈ 11 meters
        clustering = DBSCAN(eps=0.0001, min_samples=50).fit(stationary_coords)
        labels = clustering.labels_

        # Get unique clusters (ignore noise with label -1)
        unique_labels = set(labels)
        unique_labels.discard(-1)

        if len(unique_labels) < 4:
            raise ValueError(f"Only found {len(unique_labels)} stationary clusters, expected 4-5")

        # Calculate centroid for each cluster
        clusters = []
        for label in unique_labels:
            cluster_mask = labels == label
            cluster_points_indices = [stationary_indices[i] for i, mask in enumerate(cluster_mask) if mask]

            # Calculate centroid
            lats = [self.gps_points[i]['lat'] for i in cluster_points_indices]
            lons = [self.gps_points[i]['lon'] for i in cluster_points_indices]
            alts = [self.gps_points[i]['alt'] for i in cluster_points_indices]

            clusters.append({
                'lat': np.mean(lats),
                'lon': np.mean(lons),
                'alt': np.mean(alts),
                'num_points': len(cluster_points_indices)
            })

        # Sort clusters by number of points (should all be similar, ~225 points)
        clusters.sort(key=lambda x: x['num_points'], reverse=True)

        return clusters

    def _calculate_gate_positions(self, clusters: List[Dict]) -> Dict:
        """
        Calculate gate positions from cross pattern clusters.

        For standard gates: 4 clusters form a cross
        For speed gates: 4 clusters form a cross + 1 for exit gate

        Args:
            clusters: List of cluster centroids

        Returns:
            Dict with gate positions
        """
        # Take first 4 clusters for the cross pattern
        cross_clusters = clusters[:4]

        # Calculate centroid of the cross = inside entry gate (gate 1)
        center_lat = np.mean([c['lat'] for c in cross_clusters])
        center_lon = np.mean([c['lon'] for c in cross_clusters])
        center_alt = np.mean([c['alt'] for c in cross_clusters])

        # Identify which cluster is A (back/uprange) and which is C/D
        # A is the cluster furthest from center in one direction
        # We need to identify the axis orientation

        # Calculate distances and bearings from center to each cluster
        cluster_info = []
        for cluster in cross_clusters:
            bearing = self._calculate_bearing(
                center_lat, center_lon,
                cluster['lat'], cluster['lon']
            )
            distance = self._haversine_distance(
                center_lat, center_lon,
                cluster['lat'], cluster['lon']
            )
            cluster_info.append({
                'cluster': cluster,
                'bearing': bearing,
                'distance': distance
            })

        # Sort by bearing to identify opposite pairs
        cluster_info.sort(key=lambda x: x['bearing'])

        # Identify A-B axis (should be roughly opposite bearings)
        # and C-D axis (perpendicular)
        # Pair clusters that are roughly opposite (180° apart)
        pair1 = [cluster_info[0], cluster_info[2]]  # Should be opposite
        pair2 = [cluster_info[1], cluster_info[3]]  # Should be opposite

        # Determine which pair is A-B (forward-back) vs C-D (left-right)
        # A-B should align with the course direction
        # For now, we'll assume the first pair is A-B based on bearing

        # Calculate perpendicular offset for outside entry gate (gate 2)
        # 10 meters toward C (left when facing downrange)
        # We need to identify which cluster is C (left)

        # The C-D axis is perpendicular to A-B
        # C is the "left" cluster (90° counterclockwise from downrange)
        # Calculate bearing from center to C
        c_bearing = cluster_info[1]['bearing']  # Approximate

        # Calculate position 10m in direction of C from center
        gate_2_lat, gate_2_lon = self._offset_position(
            center_lat, center_lon, c_bearing, 10.0
        )

        gate_positions = {
            'gate_1_inside': {
                'lat': center_lat,
                'lon': center_lon,
                'alt': center_alt
            },
            'gate_2_outside': {
                'lat': gate_2_lat,
                'lon': gate_2_lon,
                'alt': center_alt  # Same altitude as gate 1
            }
        }

        # For speed gates, add exit gate center (5th cluster)
        if self.gate_type == 'speed' and len(clusters) >= 5:
            exit_cluster = clusters[4]
            gate_positions['gate_5_center'] = {
                'lat': exit_cluster['lat'],
                'lon': exit_cluster['lon'],
                'alt': exit_cluster['alt']
            }

        return gate_positions

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
