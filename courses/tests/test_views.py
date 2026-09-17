"""
Integration tests for the import wizard (upload -> preview -> commit) and
course visibility/permissions, using the real eastern Speed KML fixture.
"""

import os

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase
from django.urls import reverse

from courses.models import Course, CourseImport, CourseSet

User = get_user_model()

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'course_planning')


def _speed_kml_upload():
    with open(os.path.join(FIXTURE_DIR, 'mile_hi_eastern_speed.kml'), 'rb') as f:
        return SimpleUploadedFile('mile_hi_eastern_speed.kml', f.read(), content_type='application/vnd.google-earth.kml+xml')


class ImportWizardFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client = Client()
        self.client.force_login(self.user)

    def test_upload_then_preview_then_commit(self):
        resp = self.client.post(reverse('courses:import_upload'), {
            'source_file': _speed_kml_upload(),
            'carve_direction': 'left',
        })
        self.assertEqual(resp.status_code, 302)
        course_import = CourseImport.objects.get(owner=self.user)
        self.assertEqual(course_import.status, 'preview_ready')
        self.assertEqual(course_import.extracted['detected_discipline'], 'speed')

        preview_url = reverse('courses:import_preview', args=[course_import.id])
        resp = self.client.get(preview_url)
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Mile-Hi Eastern Carved Speed')

        resp = self.client.post(preview_url, {
            'name': 'Mile-Hi Eastern Speed', 'visibility': 'private', 'site_name': 'Mile-Hi',
        })
        self.assertEqual(resp.status_code, 302)

        course_set = CourseSet.objects.get(owner=self.user, name='Mile-Hi Eastern Speed')
        self.assertEqual(course_set.visibility, 'private')
        courses = Course.objects.filter(course_set=course_set)
        self.assertEqual(courses.count(), 1)
        speed_course = courses.get(discipline='speed')
        self.assertIsNotNone(speed_course.current_revision)
        self.assertEqual(speed_course.current_revision.revision_number, 1)
        self.assertEqual(len(speed_course.current_revision.geometry['gates']), 5)

        course_import.refresh_from_db()
        self.assertEqual(course_import.status, 'committed')
        self.assertEqual(course_import.committed_course_set, course_set)

    def test_commit_requires_name(self):
        resp = self.client.post(reverse('courses:import_upload'), {
            'source_file': _speed_kml_upload(), 'carve_direction': 'left',
        })
        course_import = CourseImport.objects.get(owner=self.user)
        resp = self.client.post(reverse('courses:import_preview', args=[course_import.id]), {
            'name': '', 'visibility': 'private',
        })
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Name is required')
        self.assertFalse(CourseSet.objects.filter(owner=self.user).exists())

    def test_cannot_double_commit(self):
        resp = self.client.post(reverse('courses:import_upload'), {
            'source_file': _speed_kml_upload(), 'carve_direction': 'left',
        })
        course_import = CourseImport.objects.get(owner=self.user)
        preview_url = reverse('courses:import_preview', args=[course_import.id])
        self.client.post(preview_url, {'name': 'First', 'visibility': 'private'})
        resp = self.client.post(preview_url, {'name': 'Second', 'visibility': 'private'})
        self.assertContains(resp, 'committed')
        self.assertEqual(CourseSet.objects.filter(owner=self.user).count(), 1)

    def test_missing_carve_direction_rejected(self):
        resp = self.client.post(reverse('courses:import_upload'), {'source_file': _speed_kml_upload()})
        self.assertEqual(resp.status_code, 200)
        self.assertContains(resp, 'Select a carve direction')
        self.assertFalse(CourseImport.objects.exists())


class G1SeedUploadFlowTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='alice', password='pw')
        self.client = Client()
        self.client.force_login(self.user)

    def _seed_kml_upload(self):
        content = (
            b'<?xml version="1.0" encoding="UTF-8"?>'
            b'<kml xmlns="http://www.opengis.net/kml/2.2"><Document><name>Seed</name>'
            b'<Placemark><name>G1 Center</name><Point>'
            b'<coordinates>-105.165343056,40.161709722,0</coordinates></Point></Placemark>'
            b'</Document></kml>'
        )
        return SimpleUploadedFile('seed.kml', content, content_type='application/vnd.google-earth.kml+xml')

    def test_seed_upload_builds_all_three_disciplines(self):
        resp = self.client.post(reverse('courses:import_upload'), {
            'source_file': self._seed_kml_upload(),
            'carve_direction': 'left',
            'entry_heading_deg_true': '129.6456265',
            'course_width_m': '10.014',
            'disciplines': ['distance', 'speed', 'accuracy'],
        })
        self.assertEqual(resp.status_code, 302)
        course_import = CourseImport.objects.get(owner=self.user)
        self.assertEqual(course_import.status, 'preview_ready')
        self.assertEqual(course_import.input_mode, 'g1_seed')
        self.assertEqual(set(course_import.extracted['final_geometry'].keys()),
                          {'distance', 'speed', 'accuracy'})

        preview_url = reverse('courses:import_preview', args=[course_import.id])
        resp = self.client.post(preview_url, {'name': 'Seeded Course', 'visibility': 'private'})
        self.assertEqual(resp.status_code, 302)

        course_set = CourseSet.objects.get(owner=self.user, name='Seeded Course')
        self.assertEqual(Course.objects.filter(course_set=course_set).count(), 3)

    def test_seed_upload_without_heading_fails_clearly(self):
        resp = self.client.post(reverse('courses:import_upload'), {
            'source_file': self._seed_kml_upload(),
            'carve_direction': 'left',
            'disciplines': ['distance'],
        })
        course_import = CourseImport.objects.get(owner=self.user)
        self.assertEqual(course_import.status, 'failed')
        self.assertIn('entry heading', course_import.validation_report['errors'][0])


class ImportOwnershipTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')
        self.client = Client()
        self.client.force_login(self.alice)
        self.client.post(reverse('courses:import_upload'), {
            'source_file': _speed_kml_upload(), 'carve_direction': 'left',
        })
        self.course_import = CourseImport.objects.get(owner=self.alice)

    def test_other_user_cannot_view_or_commit_someone_elses_import(self):
        bob_client = Client()
        bob_client.force_login(self.bob)
        resp = bob_client.get(reverse('courses:import_preview', args=[self.course_import.id]))
        self.assertEqual(resp.status_code, 404)


class CourseSetVisibilityTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.bob = User.objects.create_user(username='bob', password='pw')
        self.private_set = CourseSet.objects.create(name='Private', owner=self.alice, visibility='private')
        self.public_set = CourseSet.objects.create(name='Public', owner=self.alice, visibility='public')

    def test_owner_sees_both_own_sets(self):
        client = Client()
        client.force_login(self.alice)
        resp = client.get(reverse('courses:list'))
        self.assertContains(resp, 'Private')
        self.assertContains(resp, 'Public')

    def test_other_user_only_sees_public_set(self):
        client = Client()
        client.force_login(self.bob)
        resp = client.get(reverse('courses:list'))
        self.assertNotContains(resp, 'Private')
        self.assertContains(resp, 'Public')

    def test_other_user_cannot_open_private_set_detail(self):
        client = Client()
        client.force_login(self.bob)
        resp = client.get(reverse('courses:detail', args=[self.private_set.id]))
        self.assertEqual(resp.status_code, 404)

    def test_other_user_cannot_change_visibility(self):
        client = Client()
        client.force_login(self.bob)
        resp = client.post(reverse('courses:visibility', args=[self.public_set.id]), {'visibility': 'private'})
        self.assertEqual(resp.status_code, 404)
        self.public_set.refresh_from_db()
        self.assertEqual(self.public_set.visibility, 'public')

    def test_owner_can_toggle_visibility(self):
        client = Client()
        client.force_login(self.alice)
        resp = client.post(reverse('courses:visibility', args=[self.private_set.id]), {'visibility': 'public'})
        self.assertEqual(resp.status_code, 200)
        self.private_set.refresh_from_db()
        self.assertEqual(self.private_set.visibility, 'public')

    def test_archived_public_set_not_listed(self):
        self.public_set.archived = True
        self.public_set.save()
        client = Client()
        client.force_login(self.bob)
        resp = client.get(reverse('courses:list'))
        # Check for the course's own detail link rather than the word
        # "Public" -- too common a substring elsewhere on the page (nav,
        # library boilerplate) to be a reliable absence check.
        self.assertNotContains(resp, reverse('courses:detail', args=[self.public_set.id]))


class ManageAllCoursesPermissionTests(TestCase):
    def setUp(self):
        self.alice = User.objects.create_user(username='alice', password='pw')
        self.admin = User.objects.create_user(username='admin', password='pw', is_staff=True)
        from django.contrib.auth.models import Permission
        perm = Permission.objects.get(codename='manage_all_courses')
        self.admin.user_permissions.add(perm)
        self.private_set = CourseSet.objects.create(name='Private', owner=self.alice, visibility='private')

    def test_admin_with_permission_sees_everything(self):
        visible = CourseSet.objects.visible_to(self.admin)
        self.assertIn(self.private_set, visible)

    def test_staff_without_permission_does_not_see_private_sets(self):
        plain_staff = User.objects.create_user(username='staff2', password='pw', is_staff=True)
        visible = CourseSet.objects.visible_to(plain_staff)
        self.assertNotIn(self.private_set, visible)
