"""
Phase 0 tests for the legacy competition-gate system.

Two kinds of tests live here on purpose:

1. Characterization tests that pin down bugs documented in
   docs/course_planning/Swoopr_Course_System_Requirements.md (section 2).
   These are not fixed here -- the parser/geometry/crossing engine is being
   replaced wholesale in the course-system rebuild -- they exist so the
   rebuild has a documented "this is what was broken" baseline and so no one
   accidentally treats current legacy output as trustworthy in the meantime.
2. Regression tests for the Phase 0 ownership/visibility fixes actually
   applied now (CompetitionGate.objects.visible_to, scoped upload, scoped
   gate assignment and map access).
"""

import tempfile
from io import StringIO

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import TestCase, Client
from django.urls import reverse
from unittest.mock import patch

from flights.models import CompetitionGate, Flight
from flights.utils.gate_parser import GateFileParser
from flights.utils.gate_calculator import GateCalculator

User = get_user_model()


# ---------------------------------------------------------------------------
# 1. Characterization tests for known, not-yet-fixed legacy bugs
# ---------------------------------------------------------------------------

class GateFileParserKnownBugsTests(TestCase):
    """Pins the $-row filtering bug in GateFileParser._read_flysight_format."""

    def test_gnss_rows_are_currently_dropped_before_being_recognized(self):
        # `if not row or row[0].startswith('$'): continue` runs before the
        # `if row[0].startswith('$GNSS')` branch is ever reached, so every
        # FlySight v2 $GNSS data row is silently discarded.
        content = (
            "$FLYS,1\n"
            "$GNSS,2026-09-05T12:00:00.00Z,40.161700,-105.165300,1600.0,"
            "0,0,0,1,1,1,3,10,0,0\n"
            "$GNSS,2026-09-05T12:00:00.25Z,40.161701,-105.165301,1600.0,"
            "0,0,0,1,1,1,3,10,0,0\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        parser = GateFileParser(tmp_path, gate_type='standard')
        parser._read_gps_file()

        # Documents the bug: two syntactically valid $GNSS rows parse to zero
        # points. This must change before survey import is trustworthy.
        self.assertEqual(parser.gps_points, [])


class GateCalculatorKnownBugsTests(TestCase):
    """Pins the width-check and unbounded-fallback bugs in GateCalculator."""

    def setUp(self):
        # Two gate markers exactly 10 m apart running east-west.
        self.gate_positions = {
            'gate_1_inside': {'lat': 40.161700, 'lon': -105.165300, 'alt': 1600.0},
            'gate_2_outside': {'lat': 40.161700, 'lon': -105.165300 + (10.0 / 111320.0), 'alt': 1600.0},
        }
        self.calc = GateCalculator(self.gate_positions)

    def test_width_check_cannot_pass_even_for_a_dead_center_crossing(self):
        # A flight path crossing exactly through the midpoint of the two
        # gate markers is as clean a gate pass as GPS can represent.
        mid_lat = self.gate_positions['gate_1_inside']['lat']
        mid_lon = (self.gate_positions['gate_1_inside']['lon']
                   + self.gate_positions['gate_2_outside']['lon']) / 2

        gps_data = [
            {'lat': mid_lat - 0.0002, 'lon': mid_lon, 'ground_speed': 30.0, 'altitude_agl': 20.0},
            {'lat': mid_lat + 0.0002, 'lon': mid_lon, 'ground_speed': 30.0, 'altitude_agl': 20.0},
        ]

        result = self.calc.calculate_entry_gate_metrics(gps_data)

        self.assertIsNotNone(result, "expected a crossing to be detected at all")
        # By the triangle inequality, no point can be strictly <5 m from both
        # markers of an exact 10 m gate -- so this is always False today,
        # including for a perfectly centered crossing.
        self.assertFalse(
            result['passed_between_gates'],
            "documents the current impossible width test; a real fix "
            "should make a centered crossing register as clean"
        )

    def test_closest_approach_fallback_has_no_distance_cutoff(self):
        # A track that never comes near the gate should not be reported as
        # a "crossing" at all, but the legacy fallback has no maximum
        # distance and always returns something.
        gps_data = [
            {'lat': 40.170000, 'lon': -105.170000, 'ground_speed': 30.0, 'altitude_agl': 20.0},
            {'lat': 40.171000, 'lon': -105.171000, 'ground_speed': 30.0, 'altitude_agl': 20.0},
        ]

        result = self.calc.calculate_entry_gate_metrics(gps_data)

        self.assertIsNotNone(
            result,
            "documents the bug: a track roughly 1 km away still produces a "
            "'crossing' result via the unbounded closest-approach fallback"
        )
        self.assertGreater(result['gate_1_distance'], 500.0)


# ---------------------------------------------------------------------------
# 2. Regression tests for the Phase 0 ownership/visibility fixes
# ---------------------------------------------------------------------------

class CompetitionGateVisibilityTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')
        self.staff = User.objects.create_user(username='carol', password='pw', is_staff=True)

        self.alice_gate = CompetitionGate.objects.create(
            name="Alice's course", gate_type='standard', created_by=self.alice, is_parsed=True
        )
        self.bob_gate = CompetitionGate.objects.create(
            name="Bob's course", gate_type='standard', created_by=self.bob, is_parsed=True
        )
        self.global_gate = CompetitionGate.objects.create(
            name='Mile-Hi Eastern', gate_type='standard', created_by=self.staff, is_parsed=True
        )

    def test_owner_sees_own_and_staff_gates_not_other_users_gates(self):
        visible = set(CompetitionGate.objects.visible_to(self.alice).values_list('id', flat=True))
        self.assertEqual(visible, {self.alice_gate.id, self.global_gate.id})

    def test_other_user_does_not_see_alice_private_gate(self):
        visible = set(CompetitionGate.objects.visible_to(self.bob).values_list('id', flat=True))
        self.assertNotIn(self.alice_gate.id, visible)

    def test_anonymous_sees_nothing(self):
        from django.contrib.auth.models import AnonymousUser
        visible = CompetitionGate.objects.visible_to(AnonymousUser())
        self.assertEqual(list(visible), [])


class UploadGateFileOwnershipTests(TestCase):
    """A shared course name must not let one user overwrite another's gate."""

    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')

    @patch.object(GateFileParser, 'extract_gate_positions')
    def test_same_name_from_two_users_creates_two_gates(self, mock_extract):
        mock_extract.return_value = {
            'gate_1_inside': {'lat': 40.0, 'lon': -105.0, 'alt': 1600.0},
            'gate_2_outside': {'lat': 40.0001, 'lon': -105.0, 'alt': 1600.0},
        }

        def upload_as(user):
            client = Client()
            client.force_login(user)
            upload = SimpleUploadedFile('course.csv', b'time,lat,lon,hMSL\n', content_type='text/csv')
            return client.post(reverse('upload_gate_file'), {
                'gate_name': 'Shared Course Name',
                'gate_type': 'standard',
                'gate_file': upload,
            })

        resp_alice = upload_as(self.alice)
        resp_bob = upload_as(self.bob)

        self.assertEqual(resp_alice.status_code, 200)
        self.assertEqual(resp_bob.status_code, 200)

        gates = CompetitionGate.objects.filter(name='Shared Course Name')
        self.assertEqual(gates.count(), 2, "bob's upload must not overwrite alice's gate")
        self.assertEqual(set(gates.values_list('created_by', flat=True)), {self.alice.id, self.bob.id})


class GateMapAccessControlTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')
        self.gate = CompetitionGate.objects.create(
            name="Alice's private course", gate_type='standard',
            created_by=self.alice, is_parsed=True,
            gate_positions={'gate_1_inside': {'lat': 40.0, 'lon': -105.0, 'alt': 1600.0}},
        )

    def test_owner_can_view_map(self):
        client = Client()
        client.force_login(self.alice)
        resp = client.get(reverse('flights:gate_map', args=[self.gate.id]))
        self.assertEqual(resp.status_code, 200)

    def test_other_user_cannot_view_map(self):
        client = Client()
        client.force_login(self.bob)
        resp = client.get(reverse('flights:gate_map', args=[self.gate.id]))
        self.assertEqual(resp.status_code, 404)

    def test_other_user_cannot_fetch_course_data(self):
        client = Client()
        client.force_login(self.bob)
        resp = client.get(reverse('flights:gate_course_data', args=[self.gate.id]))
        self.assertEqual(resp.status_code, 404)


class UpdateFlightGateOwnershipTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')
        self.bob_gate = CompetitionGate.objects.create(
            name="Bob's private course", gate_type='standard',
            created_by=self.bob, is_parsed=True,
        )
        self.flight = Flight.objects.create(
            pilot=self.alice, device_id='dev1', session_id='sess1',
        )

    def test_cannot_assign_another_users_private_gate_to_own_flight(self):
        client = Client()
        client.force_login(self.alice)
        resp = client.post(
            reverse('update_flight_gate', args=[self.flight.id]),
            data='{"gate_id": %d}' % self.bob_gate.id,
            content_type='application/json',
        )
        self.assertEqual(resp.status_code, 404)
        self.flight.refresh_from_db()
        self.assertIsNone(self.flight.competition_gate)


# ---------------------------------------------------------------------------
# 3. Sanity check for the read-only audit command
# ---------------------------------------------------------------------------

class AuditCompetitionGatesCommandTests(TestCase):
    def test_runs_without_error_and_reports_ownerless_rows(self):
        CompetitionGate.objects.create(name='Ownerless legacy row', gate_type='standard')

        out = StringIO()
        call_command('audit_competition_gates', stdout=out)
        output = out.getvalue()

        self.assertIn('Ownerless legacy row', output)
        self.assertIn('NO OWNER', output)
        self.assertIn('This command makes no changes', output)
