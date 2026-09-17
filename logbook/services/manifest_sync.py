"""
Manifest sync — turns a landed manifest slot into a draft personal logbook entry.

Called from dz/services/load_service.py::land_load() once a Load transitions to
'landed'. Auto-created Jumps are flagged is_draft=True until the jumper reviews
and confirms them (see logbook/views.py::confirm_jump).

Public API
----------
sync_slots_for_load(load) -> list[Jump]
create_draft_jump_from_slot(slot) -> Jump | None
"""

# LoadSlot.jump_type codes -> logbook.JumpType.name (free-text, user-extensible;
# get_or_create keeps this working even if a DZ has never logged this type before).
SLOT_JUMP_TYPE_TO_JUMPTYPE_NAME = {
    'tandem':    'Tandem',
    'aff':       'AFF',
    'coach':     'Coach',
    'fun_jump':  'Fun Jump',
    'hop_n_pop': 'Hop & Pop',
    'tracking':  'Tracking',
    'wingsuit':  'Wingsuit',
}

# Roles that get an auto-drafted logbook entry when a platform user is linked.
# Guests (no user) never sync regardless of role.
SYNCED_ROLES = {'student', 'instructor', 'fun_jumper', 'videographer', 'hop_n_pop'}


def create_draft_jump_from_slot(slot):
    """
    Auto-create a draft Jump for a single LoadSlot, if eligible.
    No-op (returns None) for guest slots, un-synced roles, or slots already linked.
    Idempotent — safe to call more than once for the same slot.
    """
    from logbook.models import Jump, JumpType

    if not slot.user_id:
        return None
    if slot.role not in SYNCED_ROLES:
        return None
    if hasattr(slot, 'jump'):
        return None

    load = slot.load
    jump_type = None
    type_name = SLOT_JUMP_TYPE_TO_JUMPTYPE_NAME.get(slot.jump_type)
    if type_name:
        jump_type, _ = JumpType.objects.get_or_create(name=type_name)

    jump = Jump.objects.create(
        user=slot.user,
        date=load.date,
        dropzone=load.dropzone,
        aircraft=load.aircraft,
        jump_type=jump_type,
        altitude=slot.exit_altitude or load.exit_altitude,
        load_slot=slot,
        is_draft=True,
    )

    _link_student_jump(slot, jump)
    return jump


def _link_student_jump(slot, jump):
    """
    If this slot is tied to a student enrollment, attach the new draft Jump to
    the matching (already-existing) StudentJump record for the same date, if
    one exists and isn't linked yet. Never fabricates a StudentJump here —
    StudentJump requires an instructor/program_jump/outcome that only the
    instructor sign-off flow (training/services/progression.py, Phase 3) can
    supply.
    """
    if not slot.enrollment_id:
        return

    from training.models import StudentJump

    # jump is a OneToOneField on StudentJump — match at most one row and save()
    # it individually rather than a bulk .update(), which could violate that
    # uniqueness if more than one StudentJump exists for the same day.
    student_jump = (
        StudentJump.objects
        .filter(enrollment_id=slot.enrollment_id, jump_date=jump.date, jump__isnull=True)
        .order_by('id')
        .first()
    )
    if student_jump:
        student_jump.jump = jump
        student_jump.save(update_fields=['jump'])


def sync_slots_for_load(load):
    """Create draft Jumps for every eligible slot on a just-landed load."""
    jumps = []
    for slot in load.slots.select_related('user', 'enrollment'):
        jump = create_draft_jump_from_slot(slot)
        if jump:
            jumps.append(jump)
    return jumps
