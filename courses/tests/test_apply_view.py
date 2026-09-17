"""
View-level tests for applying a course to a flight through the actual
HTTP endpoints (not just the service function) -- ownership checks and
the upload->apply->view round trip.
"""

from django.contrib.auth import get_user_model
from django.test import Client, TestCase
from django.urls import reverse

from courses.analysis.crossings import GateFrame
from courses.geometry import EntrySetup, generate_distance_geometry
from courses.models import Course, CourseRevision, CourseSet, FlightCourseAnalysis
from flights.models import Flight

User = get_user_model()

EASTERN_G1 = (-105.165343056, 40.161709722)
EASTERN_HEADING = 129.6456265


def _point_relative_to_gate(gate, forward_m, tangential_m, t):
    frame = GateFrame.from_gate(gate)
    east = forward_m * frame.forward[0] + tangential_m * frame.tangent[0]
    north = forward_m * frame.forward[1] + tangential_m * frame.tangent[1]
    lon, lat = frame.projection.to_lonlat(east, north)
    return {'timestamp': t, 'lon': lon, 'lat': lat, 'altitude_agl': 20.0, 'ground_speed': 30.0, 'heading': 130.0}


class ApplyCourseViewTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')

        setup = EntrySetup(g1_lon=EASTERN_G1[0], g1_lat=EASTERN_G1[1], entry_heading_deg_true=EASTERN_HEADING,
                            carve_direction='left', course_width_m=10.0)
        geometry = generate_distance_geometry(setup)
        course_set = CourseSet.objects.create(name='Mile-Hi', owner=self.alice, visibility='private')
        self.course = Course.objects.create(course_set=course_set, discipline='distance')
        revision = CourseRevision.objects.create(
            course=self.course, revision_number=1, geometry=geometry,
            g1_lon=setup.g1_lon, g1_lat=setup.g1_lat, entry_heading_deg_true=setup.entry_heading_deg_true,
            course_width_m=setup.course_width_m, carve_direction=setup.carve_direction,
            source='generated_from_g1', created_by=self.alice,
        )
        self.course.current_revision = revision
        self.course.save()

        self.flight = Flight.objects.create(pilot=self.alice, device_id='d1', session_id='s1')
        g1, g5 = geometry['gates']
        t0 = 1_757_000_000.0
        points = [
            _point_relative_to_gate(g1, -1.0, 0.0, t0),
            _point_relative_to_gate(g1, 1.0, 0.0, t0 + 0.25),
            _point_relative_to_gate(g5, -1.0, 0.0, t0 + 1.75),
            _point_relative_to_gate(g5, 1.0, 0.0, t0 + 2.0),
        ]
        self.flight.flare_idx = 0
        self.flight.landing_idx = len(points) - 1
        self.flight.store_gps_data(points)

        self.client = Client()
        self.client.force_login(self.alice)

    def test_apply_form_lists_course_and_commits_analysis(self):
        resp = self.client.get(reverse('courses:apply_to_flight', args=[self.flight.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Mile-Hi')

        resp = self.client.post(reverse('courses:apply_to_flight', args=[self.flight.id]), {
            'course_id': self.course.id,
        })
        self.assertEqual(resp.status_code, 302)

        analysis = FlightCourseAnalysis.objects.get(flight=self.flight)
        self.assertEqual(analysis.entry_status, 'crossed')

        resp = self.client.get(reverse('courses:flight_analysis', args=[self.flight.id, analysis.id]))
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Competition View')
        self.assertContains(resp, 'distance:G1')

    def test_flight_detail_shows_apply_course_action(self):
        resp = self.client.get(reverse('flight_detail', args=[self.flight.id]))
        self.assertContains(resp, 'Apply Course')

    def test_other_user_cannot_apply_course_to_someone_elses_flight(self):
        bob_client = Client()
        bob_client.force_login(self.bob)
        resp = bob_client.get(reverse('courses:apply_to_flight', args=[self.flight.id]))
        self.assertEqual(resp.status_code, 404)

    def test_other_user_cannot_view_someone_elses_analysis(self):
        self.client.post(reverse('courses:apply_to_flight', args=[self.flight.id]), {'course_id': self.course.id})
        analysis = FlightCourseAnalysis.objects.get(flight=self.flight)

        bob_client = Client()
        bob_client.force_login(self.bob)
        resp = bob_client.get(reverse('courses:flight_analysis', args=[self.flight.id, analysis.id]))
        self.assertEqual(resp.status_code, 404)

    def test_private_course_owned_by_someone_else_not_offered(self):
        # Bob has his own flight but shouldn't see Alice's private course.
        bob_flight = Flight.objects.create(pilot=self.bob, device_id='d2', session_id='s2')
        bob_flight.store_gps_data([
            {'timestamp': 1, 'lat': 40.0, 'lon': -105.0, 'ground_speed': 30.0},
            {'timestamp': 2, 'lat': 40.001, 'lon': -105.0, 'ground_speed': 30.0},
        ])
        bob_client = Client()
        bob_client.force_login(self.bob)
        resp = bob_client.get(reverse('courses:apply_to_flight', args=[bob_flight.id]))
        self.assertNotContains(resp, 'Mile-Hi')
