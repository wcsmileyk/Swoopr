from django.conf import settings
from django.db import models

RATING_TYPE_CHOICES = [
    ('AFF-I', 'AFF Instructor'),
    ('TI', 'Tandem Instructor'),
    ('Coach', 'Coach'),
    ('IAD-I', 'IAD/Static Line Instructor'),
]

PROGRAM_TYPE_CHOICES = [
    ('AFF', 'AFF'),
    ('IAD_SL', 'IAD / Static Line'),
    ('Coach', 'Coach'),
    ('Tandem', 'Tandem'),
]

STUDENT_JUMP_METHOD_CHOICES = [
    ('AFF_2I', 'AFF — 2 Instructor'),
    ('AFF_solo', 'AFF — Solo JM'),
    ('Tandem', 'Tandem'),
    ('IAD', 'IAD'),
    ('SL', 'Static Line'),
    ('Coach', 'Coach'),
]


class InstructorRating(models.Model):
    """
    A USPA instructor rating held by a user. Portable — valid at any DZ.
    Currency is determined by UserProfile.uspa_expiry, not stored here.
    One row per rating type; a TI who is also an AFF-I has two rows.
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='instructor_ratings'
    )
    rating_type = models.CharField(max_length=10, choices=RATING_TYPE_CHOICES)
    earned_date = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('user', 'rating_type')]
        ordering = ['user', 'rating_type']

    def __str__(self):
        return f'{self.user.username} — {self.get_rating_type_display()}'

    @property
    def is_current(self):
        """True if the instructor's USPA membership is current."""
        import datetime
        try:
            expiry = self.user.profile.uspa_expiry
        except Exception:
            return False
        return bool(expiry and expiry >= datetime.date.today())


class InstructorPersonalLimits(models.Model):
    """
    Instructor-set or DZ-agreed personal operating limits for a specific rating.
    Travels with the instructor; not DZ-specific.
    """
    instructor_rating = models.OneToOneField(
        InstructorRating, on_delete=models.CASCADE, related_name='personal_limits'
    )
    max_student_exit_weight = models.FloatField(
        null=True, blank=True, help_text='Maximum student exit weight in lbs'
    )
    max_wind_speed = models.FloatField(
        null=True, blank=True, help_text='Personal wind limit in knots'
    )
    notes = models.TextField(blank=True, help_text='Any other personal operating limits')

    class Meta:
        verbose_name_plural = 'Instructor personal limits'

    def __str__(self):
        return f'Limits: {self.instructor_rating}'


class RatingQualification(models.Model):
    """
    Defines a specific sub-qualification that exists within a rating type.
    Seeded data — e.g. TI gets 'Handcam', AFF-I gets 'Solo JM' and '2-Instructor JM'.
    """
    rating_type = models.CharField(
        max_length=10, choices=RATING_TYPE_CHOICES,
        help_text='Which rating type this qualification belongs to'
    )
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    portable = models.BooleanField(
        default=True,
        help_text='If True, earned once and valid everywhere. If False, each DZ signs off independently.'
    )

    class Meta:
        unique_together = [('rating_type', 'name')]
        ordering = ['rating_type', 'name']

    def __str__(self):
        return f'{self.get_rating_type_display()} — {self.name}'


class InstructorQualificationEarned(models.Model):
    """
    Records that an instructor has earned a specific sub-qualification.
    Portable qualifications (e.g. handcam) travel with the instructor.
    """
    instructor_rating = models.ForeignKey(
        InstructorRating, on_delete=models.CASCADE, related_name='qualifications_earned'
    )
    qualification = models.ForeignKey(
        RatingQualification, on_delete=models.CASCADE, related_name='earned_by'
    )
    signed_off_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='qualifications_signed_off'
    )
    signed_off_date = models.DateField(null=True, blank=True)
    dropzone = models.ForeignKey(
        'organizations.Dropzone', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='qualifications_issued',
        help_text='DZ where this qualification was earned'
    )
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('instructor_rating', 'qualification')]
        ordering = ['-signed_off_date']

    def __str__(self):
        return f'{self.instructor_rating.user.username} — {self.qualification.name}'


class DropzoneAuthorization(models.Model):
    """
    A DZ-specific clearance for a user. Covers two cases:
    1. Progression stages within an instructor rating at a specific DZ
       (e.g. restricted TI → full TI at this dropzone).
    2. Non-USPA roles the DZ grants, such as videographer.

    For instructor rating authorizations, instructor_rating links to the
    underlying InstructorRating. For videographer (and future DZ-only roles),
    instructor_rating is null and authorization_type carries the role name.
    """
    AUTHORIZATION_TYPES = RATING_TYPE_CHOICES + [
        ('Videographer', 'Videographer'),
    ]

    STAGE_CHOICES = [
        ('restricted', 'Restricted'),
        ('provisional', 'Provisional'),
        ('full', 'Full'),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='dz_authorizations'
    )
    dropzone = models.ForeignKey(
        'organizations.Dropzone', on_delete=models.CASCADE, related_name='authorizations'
    )
    authorization_type = models.CharField(max_length=20, choices=AUTHORIZATION_TYPES)
    instructor_rating = models.ForeignKey(
        InstructorRating, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='dz_authorizations',
        help_text='Leave null for non-USPA roles like videographer'
    )
    stage = models.CharField(max_length=20, choices=STAGE_CHOICES, default='restricted')
    cleared_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='authorizations_granted'
    )
    cleared_date = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)
    active = models.BooleanField(default=True)

    class Meta:
        unique_together = [('user', 'dropzone', 'authorization_type')]
        ordering = ['dropzone', 'authorization_type', 'user']

    def __str__(self):
        return f'{self.user.username} — {self.get_authorization_type_display()} @ {self.dropzone.name} ({self.stage})'


class USPACategory(models.Model):
    """
    A USPA-defined training category (e.g. First Jump Course, Category A–H).
    Seeded from docs/uspa_isp.json + SIM dive flows.
    Treated as read-only reference data — DZs define their own jump structure
    via ProgramJump, optionally using recommended_jumps as a starting point.
    """
    program_type = models.CharField(max_length=10, choices=PROGRAM_TYPE_CHOICES)
    code = models.CharField(max_length=10, help_text="'fjc', 'a'–'h' for AFF")
    name = models.CharField(max_length=50, help_text="e.g. 'First Jump Course', 'Category A'")
    order = models.PositiveIntegerField(help_text='Sort order within program type')
    criteria = models.JSONField(
        default=dict,
        help_text='Full criteria pool: {"ground": [...], "freefall": [...], "canopy": [...], "AFF": [...]}'
    )
    recommended_jumps = models.JSONField(
        default=dict,
        help_text=(
            'USPA-recommended jump structure keyed by method group. '
            'Each entry is a list of jump definitions with dive_flow and allowed_methods. '
            'Used by setup_dz_program to seed a DZ\'s default structure.'
        )
    )

    class Meta:
        ordering = ['program_type', 'order']
        unique_together = [('program_type', 'code')]
        verbose_name = 'USPA Category'
        verbose_name_plural = 'USPA Categories'

    def __str__(self):
        return f'{self.get_program_type_display()} — {self.name}'


class ProgramJump(models.Model):
    """
    A DZ-defined jump within a USPA category.

    method_group scopes the jump to a specific training pathway — early
    categories have different jump counts for AFF/Tandem vs IAD/SL.
    Later categories (E–H) use 'all' since method doesn't affect structure.

    DZs can split a category into as many jumps as they like per method group,
    each with its own dive_flow and assigned_criteria subset.
    """
    METHOD_GROUP_CHOICES = [
        ('all', 'All Methods'),
        ('AFF_Tandem', 'AFF / Tandem'),
        ('AFF', 'AFF Only'),
        ('Tandem', 'Tandem Only'),
        ('IAD_SL', 'IAD / Static Line'),
        ('Coach', 'Coach'),
    ]

    dropzone = models.ForeignKey(
        'organizations.Dropzone', on_delete=models.CASCADE, related_name='program_jumps'
    )
    category = models.ForeignKey(
        USPACategory, on_delete=models.PROTECT, related_name='program_jumps'
    )
    method_group = models.CharField(
        max_length=20, choices=METHOD_GROUP_CHOICES, default='all',
        help_text='Which training pathway this jump belongs to within the category'
    )
    jump_number = models.PositiveIntegerField(
        help_text='Ordering within the category + method group at this DZ (1, 2, …)'
    )
    name = models.CharField(
        max_length=100, blank=True,
        help_text='Optional label, e.g. "90° Turns", "Clear & Pull"'
    )
    allowed_methods = models.JSONField(
        default=list,
        help_text='Specific jump methods valid for this jump, e.g. ["AFF_2I", "AFF_solo"]'
    )
    dive_flow = models.JSONField(
        default=list,
        help_text='Ordered sequence of maneuvers/actions for this jump, from the SIM'
    )
    assigned_criteria = models.JSONField(
        default=dict,
        help_text='Criteria from the category pool evaluated on this jump: {"ground": [...], ...}'
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['category__order', 'method_group', 'jump_number']
        unique_together = [('dropzone', 'category', 'method_group', 'jump_number')]

    def __str__(self):
        label = f' — {self.name}' if self.name else ''
        return f'{self.dropzone.name}: {self.category.name} [{self.get_method_group_display()}] #{self.jump_number}{label}'


class StudentEnrollment(models.Model):
    """
    A student enrolled in a training program at a specific DZ.
    One row per (student, dropzone, program_type) — a student can be in
    AFF and Coach simultaneously if the DZ allows it.
    """
    STATUS_CHOICES = [
        ('active', 'Active'),
        ('paused', 'Paused'),
        ('graduated', 'Graduated'),
        ('withdrawn', 'Withdrawn'),
    ]

    student = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='enrollments'
    )
    dropzone = models.ForeignKey(
        'organizations.Dropzone', on_delete=models.CASCADE, related_name='enrollments'
    )
    program_type = models.CharField(max_length=10, choices=PROGRAM_TYPE_CHOICES)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    enrolled_date = models.DateField(auto_now_add=True)
    enrolled_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='enrollments_created'
    )
    current_jump = models.ForeignKey(
        ProgramJump, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='students_at_jump',
        help_text='The specific DZ program jump the student is currently working on'
    )
    notes = models.TextField(blank=True)

    class Meta:
        unique_together = [('student', 'dropzone', 'program_type')]
        ordering = ['-enrolled_date']

    def __str__(self):
        return (
            f'{self.student.username} — {self.get_program_type_display()} '
            f'@ {self.dropzone.name} ({self.get_status_display()})'
        )


class StudentJump(models.Model):
    """
    The training record for a single student jump.
    Tracks which DZ program jump was being attempted, the instructor,
    outcome, and which criteria were evaluated. Links optionally to the
    student's personal Jump logbook entry.

    This model runs alongside the existing StudentSignoff flow during
    transition. StudentSignoff will be deprecated once this is fully
    integrated into the student-facing UI.
    """
    OUTCOME_CHOICES = [
        ('pass', 'Pass'),
        ('repeat', 'Repeat'),
        ('partial', 'Partial Pass'),
        ('incomplete', 'Incomplete'),
    ]

    enrollment = models.ForeignKey(
        StudentEnrollment, on_delete=models.CASCADE, related_name='student_jumps'
    )
    jump = models.OneToOneField(
        'logbook.Jump', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='student_jump',
        help_text="Link to the student's personal logbook entry, if it exists"
    )
    program_jump = models.ForeignKey(
        ProgramJump, on_delete=models.PROTECT, related_name='student_jumps'
    )
    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT,
        related_name='instructed_student_jumps'
    )
    jump_method = models.CharField(max_length=10, choices=STUDENT_JUMP_METHOD_CHOICES)
    jump_date = models.DateField()
    outcome = models.CharField(
        max_length=20, choices=OUTCOME_CHOICES,
        null=True, blank=True,
        help_text='Set when the instructor completes the debrief'
    )
    criteria_passed = models.JSONField(
        default=dict,
        help_text='{"ground": [...], "freefall": [...], "canopy": [...]}'
    )
    notes = models.TextField(blank=True)
    signed_off_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='student_jumps_signed'
    )
    signed_off_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-jump_date']

    def __str__(self):
        return (
            f'{self.enrollment.student.username} — '
            f'{self.program_jump} ({self.get_outcome_display() or "pending"})'
        )


class PendingInstructorRequest(models.Model):
    """
    Created when a student submits a jump to one or more instructors for sign-off.
    The first instructor to sign off completes the loop; the rest are deleted.
    """
    jump = models.ForeignKey(
        'logbook.Jump', on_delete=models.CASCADE, related_name='pending_instructor_requests'
    )
    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='pending_sign_requests'
    )
    student_criteria = models.JSONField(
        default=dict,
        help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('jump', 'instructor')]
        ordering = ['-created_at']
        db_table = 'logbook_pendinginstructorrequest'

    def __str__(self):
        return f'Pending: Jump #{self.jump.jump_number} → {self.instructor.username}'


class StudentSignoff(models.Model):
    """
    Immutable record created when an instructor signs off on a student jump.
    Credentials are snapshotted at sign time so profile changes don't alter records.
    """
    RATING_CHOICES = [
        ('AFF-I', 'AFF Instructor'),
        ('TI', 'Tandem Instructor'),
        ('IAD-I', 'IAD/Static Line Instructor'),
        ('Coach', 'Coach'),
    ]

    jump = models.OneToOneField(
        'logbook.Jump', on_delete=models.CASCADE, related_name='student_signoff'
    )
    signed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='student_signoffs_given'
    )
    signed_at = models.DateTimeField(auto_now_add=True)

    instructor_name = models.CharField(max_length=100)
    instructor_license = models.CharField(max_length=20, blank=True)
    instructor_rating = models.CharField(max_length=10, choices=RATING_CHOICES)

    criteria_passed = models.JSONField(
        default=dict,
        help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}'
    )

    criteria_modified = models.BooleanField(
        default=False,
        help_text="True if the instructor changed the student's pre-filled criteria"
    )
    student_notified = models.BooleanField(default=False)

    class Meta:
        db_table = 'logbook_studentsignoff'

    def __str__(self):
        return f'Signoff: Jump #{self.jump.jump_number} by {self.instructor_name}'
