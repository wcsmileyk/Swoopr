import hashlib
import json
import uuid

from django.conf import settings
from django.contrib.gis.db import models as gis_models
from django.db import models


DISCIPLINE_CHOICES = [
    ('distance', 'Drag Distance'),
    ('speed', 'Carved Speed'),
    ('accuracy', 'Zone Accuracy'),
]

VISIBILITY_CHOICES = [
    ('private', 'Private'),
    ('public', 'Public'),
]

GEOMETRY_SOURCE_CHOICES = [
    ('generated_from_g1', 'Generated from G1 seed'),
    ('kml_explicit', 'Explicit KML geometry'),
    ('kml_seed', 'KML G1 seed'),
    ('survey', 'gSwoop-style survey'),
]

IMPORT_FORMAT_CHOICES = [
    ('kml', 'KML'),
    ('survey_csv', 'Survey CSV/GSW'),
]

IMPORT_INPUT_MODE_CHOICES = [
    ('explicit_geometry', 'Explicit geometry'),
    ('g1_seed', 'G1 seed + heading'),
    ('survey', 'gSwoop-style survey'),
]

IMPORT_STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('preview_ready', 'Preview ready'),
    ('committed', 'Committed'),
    ('failed', 'Failed'),
    ('expired', 'Expired'),
]


class CourseSetQuerySet(models.QuerySet):
    def visible_to(self, user):
        """Read visibility: owner sees their own sets regardless of
        visibility; everyone (including anonymous) sees public, non-archived
        sets; a user with the manage_all_courses permission sees everything.
        """
        if user is not None and user.is_authenticated and user.has_perm('courses.manage_all_courses'):
            return self.all()

        public_qs = models.Q(visibility='public', archived=False)
        if user is not None and user.is_authenticated:
            return self.filter(public_qs | models.Q(owner=user))
        return self.filter(public_qs)

    def editable_by(self, user):
        """Write visibility: owner, or a manage_all_courses admin."""
        if user is None or not user.is_authenticated:
            return self.none()
        if user.has_perm('courses.manage_all_courses'):
            return self.all()
        return self.filter(owner=user)


class CourseSet(models.Model):
    """A named entry setup that can carry one shared geometry per
    discipline (Distance/Speed/Accuracy)."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=200)
    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='course_sets'
    )
    site_name = models.CharField(max_length=200, blank=True, help_text="Dropzone or location name, free text for now")
    visibility = models.CharField(max_length=10, choices=VISIBILITY_CHOICES, default='private')
    archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    objects = CourseSetQuerySet.as_manager()

    class Meta:
        ordering = ['-created_at']
        permissions = [
            ('manage_all_courses', 'Can view, edit, and moderate all course sets'),
        ]

    def __str__(self):
        return f'{self.name} ({self.owner})'


class Course(models.Model):
    """One discipline's course within a CourseSet. Stable identity across
    revisions -- the geometry itself lives on CourseRevision."""

    course_set = models.ForeignKey(CourseSet, on_delete=models.CASCADE, related_name='courses')
    discipline = models.CharField(max_length=10, choices=DISCIPLINE_CHOICES)
    current_revision = models.ForeignKey(
        'CourseRevision', on_delete=models.SET_NULL, null=True, blank=True, related_name='+'
    )
    archived = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['course_set', 'discipline'], name='unique_discipline_per_course_set'),
        ]

    def __str__(self):
        return f'{self.course_set.name} - {self.get_discipline_display()}'

    def set_current_revision(self, revision: 'CourseRevision'):
        assert revision.course_id == self.id
        self.current_revision = revision
        self.save(update_fields=['current_revision'])


def _geometry_hash(geometry: dict) -> str:
    canonical = json.dumps(geometry, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


class CourseRevision(models.Model):
    """Immutable geometry snapshot. Never edit an existing revision --
    create a new one and repoint Course.current_revision."""

    course = models.ForeignKey(Course, on_delete=models.CASCADE, related_name='revisions')
    revision_number = models.PositiveIntegerField()

    geometry = models.JSONField(help_text="Discipline geometry dict as produced by courses.geometry")
    geometry_hash = models.CharField(max_length=64, editable=False)
    algorithm_version = models.CharField(max_length=50, blank=True)

    # Entry setup this revision was generated/validated from
    g1_lon = models.FloatField()
    g1_lat = models.FloatField()
    entry_heading_deg_true = models.FloatField()
    course_width_m = models.FloatField()
    carve_direction = models.CharField(max_length=5, blank=True, choices=[('left', 'Left'), ('right', 'Right')])
    center = gis_models.PointField(srid=4326, null=True, blank=True, help_text="G1 center, for location filtering")

    source = models.CharField(max_length=20, choices=GEOMETRY_SOURCE_CHOICES)
    source_import = models.ForeignKey(
        'CourseImport', on_delete=models.SET_NULL, null=True, blank=True, related_name='revisions'
    )

    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    change_reason = models.TextField(blank=True)

    class Meta:
        ordering = ['course', '-revision_number']
        constraints = [
            models.UniqueConstraint(fields=['course', 'revision_number'], name='unique_revision_number_per_course'),
        ]

    def __str__(self):
        return f'{self.course} rev {self.revision_number}'

    def save(self, *args, **kwargs):
        if not self.geometry_hash:
            self.geometry_hash = _geometry_hash(self.geometry)
        if self.center is None:
            from django.contrib.gis.geos import Point
            self.center = Point(self.g1_lon, self.g1_lat, srid=4326)
        super().save(*args, **kwargs)


class CourseImport(models.Model):
    """A private, staged upload (KML or survey) with a preview/commit
    workflow. The source file stays private even if the resulting course
    is made public."""

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='course_imports')

    source_file = models.FileField(upload_to='course_imports/')
    file_hash = models.CharField(max_length=64, blank=True)
    format = models.CharField(max_length=20, choices=IMPORT_FORMAT_CHOICES)
    input_mode = models.CharField(max_length=20, choices=IMPORT_INPUT_MODE_CHOICES)

    status = models.CharField(max_length=20, choices=IMPORT_STATUS_CHOICES, default='pending')
    validation_report = models.JSONField(default=dict, blank=True, help_text="Warnings/errors from parsing")
    extracted = models.JSONField(
        null=True, blank=True,
        help_text="Preview data: detected entry setup(s) and per-discipline geometry candidates"
    )

    committed_course_set = models.ForeignKey(
        CourseSet, on_delete=models.SET_NULL, null=True, blank=True, related_name='imports'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'Import {self.id} ({self.status}) by {self.owner}'


class FlightCourseAnalysis(models.Model):
    """Result of evaluating one flight's GPS track against one immutable
    course revision. Multiple analyses can exist per flight (different
    courses, or recomputed after a track/course change); is_primary marks
    the one shown by default. Never edited in place -- a recompute creates
    a new row and un-primaries the old one, so a historical result stays
    reproducible even if the flight is reanalyzed later."""

    ENTRY_STATUS_CHOICES = [
        ('crossed', 'Crossed'),
        ('not_crossed', 'Not crossed'),
        ('entry_not_detected', 'Entry not detected'),
    ]

    flight = models.ForeignKey('flights.Flight', on_delete=models.CASCADE, related_name='course_analyses')
    course = models.ForeignKey(Course, on_delete=models.PROTECT, related_name='flight_analyses')
    course_revision = models.ForeignKey(CourseRevision, on_delete=models.PROTECT, related_name='flight_analyses')

    engine_version = models.CharField(max_length=50)
    is_primary = models.BooleanField(default=False)

    window_start_idx = models.IntegerField(null=True, blank=True)
    window_end_idx = models.IntegerField(null=True, blank=True)

    avg_vertical_accuracy_m = models.FloatField(
        null=True, blank=True,
        help_text="Mean receiver-reported vertical accuracy (vAcc) over the analyzed window; "
                   "a quality indicator, not a verified error bound on any specific result"
    )

    entry_status = models.CharField(max_length=20, choices=ENTRY_STATUS_CHOICES)
    gate_results = models.JSONField(default=dict, help_text="gate_id -> serialized GateCrossingResult")
    metrics = models.JSONField(default=dict, help_text="Discipline-specific derived metrics")
    warnings = models.JSONField(default=list)

    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['flight', 'is_primary']),
        ]

    def __str__(self):
        return f'{self.flight_id} vs {self.course} ({self.entry_status})'


class CourseAuditEvent(models.Model):
    """Records privileged mutations (visibility changes, admin moderation,
    imports committed) for review. Append-only."""

    actor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    action = models.CharField(max_length=50)
    course_set = models.ForeignKey(CourseSet, on_delete=models.SET_NULL, null=True, blank=True, related_name='audit_events')
    detail = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.action} by {self.actor} @ {self.created_at:%Y-%m-%d %H:%M}'
