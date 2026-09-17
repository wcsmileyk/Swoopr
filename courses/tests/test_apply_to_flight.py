"""
Integration tests for applying a course to a flight -- the piece that
turns the pure crossing engine into a stored, reviewable result.
"""

from django.contrib.auth import get_user_model
from django.test import TestCase

from courses.analysis.crossings import GateFrame
from courses.geometry import EntrySetup, generate_distance_geometry, generate_speed_geometry
from courses.models import Course, CourseRevision, CourseSet, FlightCourseAnalysis
from courses.services import ApplyCourseError, apply_course_to_flight
from flights.models import Flight

User = get_user_model()

EASTERN_G1 = (-105.165343056, 40.161709722)
EASTERN_HEADING = 129.6456265


def _entry_setup():
    return EntrySetup(g1_lon=EASTERN_G1[0], g1_lat=EASTERN_G1[1], entry_heading_deg_true=EASTERN_HEADING,
                       carve_direction='left', course_width_m=10.0)


def _point_relative_to_gate(gate, forward_m, tangential_m, t, **extra):
    frame = GateFrame.from_gate(gate)
    east = forward_m * frame.forward[0] + tangential_m * frame.tangent[0]
    north = forward_m * frame.forward[1] + tangential_m * frame.tangent[1]
    lon, lat = frame.projection.to_lonlat(east, north)
    point = {'timestamp': t, 'lon': lon, 'lat': lat, 'altitude_agl': 20.0, 'ground_speed': 30.0, 'heading': 130.0}
    point.update(extra)
    return point


def _make_course(owner, discipline, geometry, setup):
    course_set = CourseSet.objects.create(name='Test Course', owner=owner, visibility='private')
    course = Course.objects.create(course_set=course_set, discipline=discipline)
    revision = CourseRevision.objects.create(
        course=course, revision_number=1, geometry=geometry,
        g1_lon=setup.g1_lon, g1_lat=setup.g1_lat, entry_heading_deg_true=setup.entry_heading_deg_true,
        course_width_m=setup.course_width_m, carve_direction=setup.carve_direction,
        source='generated_from_g1', created_by=owner,
    )
    course.current_revision = revision
    course.save()
    return course


class ApplyDistanceCourseTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.setup = _entry_setup()
        self.geometry = generate_distance_geometry(self.setup)
        self.course = _make_course(self.user, 'distance', self.geometry, self.setup)
        self.flight = Flight.objects.create(pilot=self.user, device_id='d1', session_id='s1')

    def _build_track(self, g1_offset, g5_offset, dt=0.25):
        g1, g5 = self.geometry['gates']
        t0 = 1_757_000_000.0
        points = [
            _point_relative_to_gate(g1, -30.0 + g1_offset, 0.0, t0),
            _point_relative_to_gate(g1, -1.0 + g1_offset, 0.0, t0 + 10),
            _point_relative_to_gate(g1, 1.0 + g1_offset, 0.0, t0 + 10.25),
            _point_relative_to_gate(g5, -1.0 + g5_offset, 0.0, t0 + 11.5),
            _point_relative_to_gate(g5, 1.0 + g5_offset, 0.0, t0 + 11.75),
        ]
        return points

    def test_apply_with_clean_crossing_computes_elapsed_time(self):
        points = self._build_track(0, 0)
        self.flight.flare_idx = 1
        self.flight.landing_idx = len(points) - 1
        self.flight.store_gps_data(points)

        analysis = apply_course_to_flight(self.flight, self.course, actor=self.user)

        self.assertEqual(analysis.entry_status, 'crossed')
        self.assertTrue(analysis.is_primary)
        self.assertEqual(analysis.gate_results['distance:G1']['status'], 'crossed')
        self.assertEqual(analysis.gate_results['distance:G5']['status'], 'crossed')
        self.assertAlmostEqual(analysis.metrics['elapsed_g1_to_g5_s'], 1.5, delta=0.01)
        # Last point is 1m past G5 (station 50m), so ~51m down-course from G1.
        self.assertAlmostEqual(analysis.metrics['down_course_distance_m'], 51.0, delta=0.1)
        # altitude_agl=20.0 on every synthetic point (see _point_relative_to_gate default)
        self.assertAlmostEqual(analysis.gate_results['distance:G1']['altitude_agl'], 20.0, delta=0.01)

    def test_avg_vertical_accuracy_computed_over_window(self):
        g1, g5 = self.geometry['gates']
        t0 = 1_757_000_000.0
        points = [
            _point_relative_to_gate(g1, -1.0, 0.0, t0, v_acc=0.6),
            _point_relative_to_gate(g1, 1.0, 0.0, t0 + 0.25, v_acc=0.8),
            _point_relative_to_gate(g5, -1.0, 0.0, t0 + 1.75, v_acc=1.0),
            _point_relative_to_gate(g5, 1.0, 0.0, t0 + 2.0, v_acc=1.2),
        ]
        self.flight.flare_idx = 0
        self.flight.landing_idx = len(points) - 1
        self.flight.store_gps_data(points)

        analysis = apply_course_to_flight(self.flight, self.course, actor=self.user)
        self.assertAlmostEqual(analysis.avg_vertical_accuracy_m, 0.9, delta=0.01)

    def test_missing_g1_crossing_reports_entry_not_detected(self):
        g1, g5 = self.geometry['gates']
        t0 = 1_757_000_000.0
        points = [
            _point_relative_to_gate(g1, -30.0, 500.0, t0),  # way off to the side, never crosses
            _point_relative_to_gate(g1, -29.0, 500.0, t0 + 1),
        ]
        self.flight.flare_idx = 0
        self.flight.landing_idx = 1
        self.flight.store_gps_data(points)

        analysis = apply_course_to_flight(self.flight, self.course, actor=self.user)

        self.assertEqual(analysis.entry_status, 'entry_not_detected')
        self.assertEqual(analysis.metrics, {})
        self.assertTrue(any('No valid' in w for w in analysis.warnings))

    def test_recompute_marks_previous_analysis_non_primary(self):
        points = self._build_track(0, 0)
        self.flight.flare_idx = 1
        self.flight.landing_idx = len(points) - 1
        self.flight.store_gps_data(points)

        first = apply_course_to_flight(self.flight, self.course, actor=self.user)
        second = apply_course_to_flight(self.flight, self.course, actor=self.user)

        first.refresh_from_db()
        self.assertFalse(first.is_primary)
        self.assertTrue(second.is_primary)
        self.assertEqual(FlightCourseAnalysis.objects.filter(flight=self.flight).count(), 2)

    def test_no_gps_data_raises(self):
        with self.assertRaises(ApplyCourseError):
            apply_course_to_flight(self.flight, self.course, actor=self.user)

    def test_course_without_revision_raises(self):
        course_set = CourseSet.objects.create(name='Empty', owner=self.user, visibility='private')
        empty_course = Course.objects.create(course_set=course_set, discipline='distance')
        self.flight.store_gps_data(self._build_track(0, 0))
        with self.assertRaises(ApplyCourseError):
            apply_course_to_flight(self.flight, empty_course, actor=self.user)


class ApplySpeedCourseTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.setup = _entry_setup()
        self.geometry = generate_speed_geometry(self.setup)
        self.course = _make_course(self.user, 'speed', self.geometry, self.setup)
        self.flight = Flight.objects.create(pilot=self.user, device_id='d1', session_id='s1')

    def test_full_gate_run_computes_sector_times(self):
        gates = self.geometry['gates']
        t0 = 1_757_000_000.0
        points = []
        for i, gate in enumerate(gates):
            base_t = t0 + i * 2.0
            points.append(_point_relative_to_gate(gate, -1.0, 0.0, base_t))
            points.append(_point_relative_to_gate(gate, 1.0, 0.0, base_t + 0.25))

        self.flight.flare_idx = 0
        self.flight.landing_idx = len(points) - 1
        self.flight.store_gps_data(points)

        analysis = apply_course_to_flight(self.flight, self.course, actor=self.user)

        self.assertEqual(analysis.entry_status, 'crossed')
        self.assertEqual(len(analysis.metrics['sector_times_s']), 4)
        self.assertEqual(len(analysis.metrics['gate_speeds_mps']), 5)
        self.assertTrue(analysis.metrics['all_gates_within_span'])
