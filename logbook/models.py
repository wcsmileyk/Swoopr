from django.conf import settings
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Max


STUDENT_CATEGORY_CHOICES = [
    ('a', 'Category A'), ('b', 'Category B'), ('c', 'Category C'),
    ('d', 'Category D'), ('e', 'Category E'), ('f', 'Category F'),
    ('g', 'Category G'), ('h', 'Category H'),
]

JUMP_METHOD_CHOICES = [
    ('AFF', 'AFF'),
    ('Tandem', 'Tandem'),
    ('IAD_SL', 'IAD / Static Line'),
    ('Coach', 'Coach'),
]


class Dropzone(models.Model):
    name = models.CharField(max_length=100, unique=True)
    city = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)
    country = models.CharField(max_length=100, blank=True, default='USA')
    icao = models.CharField(max_length=10, blank=True, help_text="Nearest airport ICAO code")

    class Meta:
        ordering = ['name']

    def __str__(self):
        parts = [self.name]
        if self.city and self.state:
            parts.append(f'{self.city}, {self.state}')
        elif self.city or self.state:
            parts.append(self.city or self.state)
        return ' — '.join(parts)


class Aircraft(models.Model):
    model = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100, blank=True)

    class Meta:
        ordering = ['manufacturer', 'model']
        verbose_name_plural = 'Aircraft'

    def __str__(self):
        if self.manufacturer:
            return f'{self.manufacturer} {self.model}'
        return self.model


class JumpType(models.Model):
    name = models.CharField(max_length=100, unique=True)
    working_jump = models.BooleanField(
        default=True,
        help_text='Counts as a training/working jump (vs fun/leisure)'
    )

    class Meta:
        ordering = ['name']

    def __str__(self):
        return self.name


class Jump(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='jumps')

    # Sequential number within this user's jump log
    jump_number = models.PositiveIntegerField(null=True, blank=True)

    # Core logbook fields
    date = models.DateField()
    dropzone = models.ForeignKey(
        Dropzone, on_delete=models.PROTECT, related_name='jumps'
    )
    aircraft = models.ForeignKey(
        Aircraft, on_delete=models.PROTECT, related_name='jumps',
        null=True, blank=True
    )
    canopy = models.ForeignKey(
        'users.Canopy', on_delete=models.SET_NULL, related_name='jumps',
        null=True, blank=True
    )
    jump_type = models.ForeignKey(
        JumpType, on_delete=models.PROTECT, related_name='jumps',
        null=True, blank=True
    )

    # Jump details
    altitude = models.IntegerField(
        null=True, blank=True, help_text='Exit altitude in feet'
    )
    formation_size = models.IntegerField(
        null=True, blank=True, help_text='Number of people in formation'
    )
    swoop = models.BooleanField(default=False)
    turn_rotation = models.IntegerField(
        null=True, blank=True,
        help_text='Manually entered turn rotation (degrees) if no GPS file'
    )
    notes = models.TextField(blank=True)

    # Student jump fields (auto-populated when user has no license)
    student_category = models.CharField(
        max_length=1, choices=STUDENT_CATEGORY_CHOICES, null=True, blank=True
    )
    student_jump_method = models.CharField(
        max_length=10, choices=JUMP_METHOD_CHOICES, null=True, blank=True
    )

    # GPS file link (optional — attached during upload or retroactively)
    flight = models.OneToOneField(
        'flights.Flight', on_delete=models.SET_NULL,
        null=True, blank=True, related_name='jump'
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['user', 'date', 'created_at']
        indexes = [
            models.Index(fields=['user', 'date']),
            models.Index(fields=['user', 'jump_number']),
            models.Index(fields=['user', 'swoop']),
            models.Index(fields=['flight']),
        ]

    def __str__(self):
        num = f'#{self.jump_number} ' if self.jump_number else ''
        return f'{num}{self.date} — {self.dropzone.name}'

    def save(self, *args, **kwargs):
        if not self.pk and self.jump_number is None:
            max_num = Jump.objects.filter(user=self.user).aggregate(
                Max('jump_number')
            )['jump_number__max']
            if max_num is None:
                # First jump — start at the user's configured offset
                try:
                    start = self.user.profile.jump_number_offset or 1
                except Exception:
                    start = 1
                self.jump_number = start
            else:
                self.jump_number = max_num + 1
        super().save(*args, **kwargs)

    @classmethod
    def renumber_for_user(cls, user):
        """
        Reassign jump numbers for a user in strict date + created_at order,
        starting from the user's configured jump_number_offset.
        Call this after retroactive inserts or deletions to keep numbering clean.
        """
        try:
            start = user.profile.jump_number_offset or 1
        except Exception:
            start = 1
        jumps = list(
            cls.objects.filter(user=user).order_by('date', 'created_at').values_list('pk', flat=True)
        )
        for i, pk in enumerate(jumps, start=start):
            cls.objects.filter(pk=pk).update(jump_number=i)

    @property
    def has_gps(self):
        return self.flight_id is not None

    @property
    def has_swoop_analysis(self):
        return self.flight_id is not None and self.flight.analysis_successful and self.flight.is_swoop

    @property
    def is_student_jump(self):
        return self.student_category is not None

    @property
    def signoff_status(self):
        """Returns 'signed', 'pending', or 'unsigned' for student jumps."""
        if not self.is_student_jump:
            return None
        if hasattr(self, 'student_signoff'):
            return 'signed'
        if self.pending_instructor_requests.exists():
            return 'pending'
        return 'unsigned'


class InstructorRate(models.Model):
    """Maps a jump type to a dollar amount earned per jump for an instructor."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='instructor_rates')
    jump_type = models.ForeignKey(JumpType, on_delete=models.CASCADE, related_name='instructor_rates')
    amount = models.DecimalField(max_digits=8, decimal_places=2, help_text='Pay per jump in USD')

    class Meta:
        unique_together = [('user', 'jump_type')]

    def __str__(self):
        return f'{self.user.username} — {self.jump_type.name}: ${self.amount}'


class PendingInstructorRequest(models.Model):
    """
    Created when a student submits a jump to one or more instructors for sign-off.
    The first instructor to sign off completes the loop; the rest are deleted.
    """
    jump = models.ForeignKey(Jump, on_delete=models.CASCADE, related_name='pending_instructor_requests')
    instructor = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='pending_sign_requests'
    )
    # What the student thinks they passed — stored so the instructor sees it pre-filled
    student_criteria = models.JSONField(
        default=dict,
        help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}'
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = [('jump', 'instructor')]
        ordering = ['-created_at']

    def __str__(self):
        return f'Pending: Jump #{self.jump.jump_number} → {self.instructor.username}'


class StudentSignoff(models.Model):
    """
    Immutable record created when an instructor signs off on a student jump.
    Credentials are snapshotted at sign time so profile changes don't alter records.
    Only the signing instructor (or co-instructors added as pending) can request changes.
    """
    RATING_CHOICES = [
        ('AFF-I', 'AFF Instructor'),
        ('TI', 'Tandem Instructor'),
        ('IAD-I', 'IAD/Static Line Instructor'),
        ('Coach', 'Coach'),
    ]

    jump = models.OneToOneField(Jump, on_delete=models.CASCADE, related_name='student_signoff')
    signed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name='student_signoffs_given'
    )
    signed_at = models.DateTimeField(auto_now_add=True)

    # Credential snapshot
    instructor_name = models.CharField(max_length=100)
    instructor_license = models.CharField(max_length=20, blank=True)
    instructor_rating = models.CharField(max_length=10, choices=RATING_CHOICES)

    # Criteria that passed on this jump
    criteria_passed = models.JSONField(
        default=dict,
        help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}'
    )

    # Modification tracking for student notification
    criteria_modified = models.BooleanField(
        default=False,
        help_text='True if the instructor changed the student\'s pre-filled criteria'
    )
    student_notified = models.BooleanField(default=False)

    def __str__(self):
        return f'Signoff: Jump #{self.jump.jump_number} by {self.instructor_name}'
