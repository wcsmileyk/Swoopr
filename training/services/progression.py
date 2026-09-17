"""
Progression service — eligibility and instructor sign-off logic for the
StudentJump-based training pipeline (the successor to the legacy
StudentSignoff/PendingInstructorRequest flow — see training/models.py
docstrings and training/management/commands/backfill_student_jumps.py).

Public API
----------
get_eligible_next_jumps(enrollment)
instructor_can_sign(instructor, dropzone, jump_method)
log_and_sign_student_jump(enrollment, program_jump, instructor, jump_method,
                           jump_date, outcome, criteria_passed, notes,
                           e_signature_data, jump=None)
sign_off_student_jump(student_jump, instructor, e_signature_data=None,
                       outcome=None, criteria_passed=None, notes=None)
"""

from django.db import models as dj_models
from django.db import transaction
from django.utils import timezone

# training.STUDENT_JUMP_METHOD_CHOICES -> the InstructorRating.rating_type
# required to sign off a jump using that method.
JUMP_METHOD_TO_RATING = {
    'AFF_2I': 'AFF-I',
    'AFF_solo': 'AFF-I',
    'Tandem': 'TI',
    'IAD': 'IAD-I',
    'SL': 'IAD-I',
    'Coach': 'Coach',
}


class SignOffNotAuthorized(Exception):
    """Raised when an instructor lacks the rating required to sign a jump."""


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def get_passed_criteria(enrollment):
    """
    Set of every criteria string the student has been signed off on across
    all of this enrollment's StudentJump rows (any section, any past jump).
    """
    from training.models import StudentJump

    passed = set()
    rows = (
        StudentJump.objects
        .filter(enrollment=enrollment, signed_off_at__isnull=False)
        .values_list('criteria_passed', flat=True)
    )
    for criteria_passed in rows:
        for section_items in criteria_passed.values():
            passed.update(section_items)
    return passed


def get_blocking_criteria(enrollment, program_jump):
    """
    {'required': {...}, 'missing': {...}} for a program_jump — the criteria
    it's assigned, and which of those the student hasn't passed yet on any
    prior jump in this enrollment. `missing` only includes sections with at
    least one outstanding item.
    """
    if not program_jump:
        return {'required': {}, 'missing': {}}

    passed = get_passed_criteria(enrollment)
    required = program_jump.assigned_criteria or {}
    missing = {
        section: [c for c in items if c not in passed]
        for section, items in required.items()
    }
    missing = {section: items for section, items in missing.items() if items}
    return {'required': required, 'missing': missing}


def _find_next_program_jump(dropzone, enrollment, program_jump):
    """
    Next active ProgramJump after program_jump in this enrollment's program:
    either the first jump of a later category, or a higher jump_number
    within the same category + method_group.
    """
    from training.models import ProgramJump

    return (
        ProgramJump.objects
        .filter(dropzone=dropzone, category__program_type=enrollment.program_type, is_active=True)
        .exclude(pk=program_jump.pk)
        .filter(
            dj_models.Q(category__order__gt=program_jump.category.order) |
            dj_models.Q(
                category__order=program_jump.category.order,
                method_group=program_jump.method_group,
                jump_number__gt=program_jump.jump_number,
            )
        )
        .order_by('category__order', 'method_group', 'jump_number')
        .first()
    )


def get_eligible_next_jumps(enrollment):
    """
    What this student is cleared for next.

    Returns:
    {
      'current_jump': ProgramJump | None,   # the jump they're working on now
      'blocking': {'required': {...}, 'missing': {...}},
      'is_cleared': bool,                   # no outstanding criteria on current_jump
      'next_jump': ProgramJump | None,      # only set when is_cleared
    }

    A brand-new enrollment with no current_jump assigned yet is treated as
    cleared for the first active ProgramJump in its program.
    """
    from training.models import ProgramJump

    current = enrollment.current_jump
    if current is None:
        first_jump = (
            ProgramJump.objects
            .filter(dropzone=enrollment.dropzone, category__program_type=enrollment.program_type, is_active=True)
            .order_by('category__order', 'method_group', 'jump_number')
            .first()
        )
        return {
            'current_jump': None,
            'blocking': {'required': {}, 'missing': {}},
            'is_cleared': True,
            'next_jump': first_jump,
        }

    blocking = get_blocking_criteria(enrollment, current)
    is_cleared = not blocking['missing']
    next_jump = _find_next_program_jump(enrollment.dropzone, enrollment, current) if is_cleared else None
    return {
        'current_jump': current,
        'blocking': blocking,
        'is_cleared': is_cleared,
        'next_jump': next_jump,
    }


# ---------------------------------------------------------------------------
# Instructor authorization
# ---------------------------------------------------------------------------

def instructor_can_sign(instructor, dropzone, jump_method):
    """
    True if `instructor` holds a current InstructorRating for the rating
    required by jump_method, and hasn't been explicitly revoked at this DZ
    (an inactive DropzoneAuthorization for that rating type).
    """
    from training.models import DropzoneAuthorization, InstructorRating

    required = JUMP_METHOD_TO_RATING.get(jump_method)
    if not required:
        return False

    rating = InstructorRating.objects.filter(user=instructor, rating_type=required).first()
    if not rating or not rating.is_current:
        return False

    revoked = DropzoneAuthorization.objects.filter(
        user=instructor, dropzone=dropzone, authorization_type=required, active=False,
    ).exists()
    return not revoked


# ---------------------------------------------------------------------------
# Sign-off
# ---------------------------------------------------------------------------

@transaction.atomic
def sign_off_student_jump(student_jump, instructor, e_signature_data=None, outcome=None, criteria_passed=None, notes=None):
    """
    Complete the instructor sign-off on an existing StudentJump. Validates
    authorization, records the e-signature, advances enrollment.current_jump
    on a pass, and notifies the student.
    """
    dropzone = student_jump.enrollment.dropzone
    if not instructor_can_sign(instructor, dropzone, student_jump.jump_method):
        raise SignOffNotAuthorized(
            f'{instructor} is not authorized to sign off '
            f'{student_jump.get_jump_method_display()} jumps at {dropzone}.'
        )

    if outcome is not None:
        student_jump.outcome = outcome
    if criteria_passed is not None:
        student_jump.criteria_passed = criteria_passed
    if notes is not None:
        student_jump.notes = notes
    student_jump.signed_off_by = instructor
    student_jump.signed_off_at = timezone.now()
    student_jump.e_signature_data = e_signature_data or {}
    student_jump.save()

    if student_jump.outcome == 'pass':
        next_jump = _find_next_program_jump(dropzone, student_jump.enrollment, student_jump.program_jump)
        if next_jump:
            student_jump.enrollment.current_jump = next_jump
            student_jump.enrollment.save(update_fields=['current_jump'])

    _notify_student(student_jump)
    return student_jump


def log_and_sign_student_jump(
    enrollment, program_jump, instructor, jump_method, jump_date,
    outcome, criteria_passed, notes='', e_signature_data=None, jump=None,
):
    """
    Create a StudentJump and immediately sign it off in one step — matches
    the DZ UI, where the instructor completing the form is the signer.
    Pass `jump` to link an existing logbook.Jump (e.g. one already
    auto-drafted by the manifest sync) rather than leaving it unset.
    """
    from training.models import StudentJump

    with transaction.atomic():
        student_jump = StudentJump.objects.create(
            enrollment=enrollment,
            jump=jump,
            program_jump=program_jump,
            instructor=instructor,
            jump_method=jump_method,
            jump_date=jump_date,
            criteria_passed=criteria_passed or {},
            notes=notes,
        )
        return sign_off_student_jump(
            student_jump, instructor,
            e_signature_data=e_signature_data,
            outcome=outcome,
            criteria_passed=criteria_passed,
        )


def _notify_student(student_jump):
    from accounts.models import Notification

    outcome_label = student_jump.get_outcome_display() or 'reviewed'
    Notification.objects.create(
        user=student_jump.enrollment.student,
        category='student_signoff',
        title=f'{student_jump.program_jump} — {outcome_label}',
        body=student_jump.notes,
    )
