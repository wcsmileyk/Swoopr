import datetime

from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render
from django.http import HttpResponse
from django.utils import timezone

from organizations.models import Dropzone, DropzoneMembership
from .models import DailyRoster, JumpRunConditions, Load, LoadSlot
from .services import load_service
from .views import _check_access, _get_user_op_dzs, _parse_date

User = get_user_model()


def _get_dz_or_403(request, dz_id, min_role='staff'):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role=min_role):
        from django.http import HttpResponseForbidden
        return dz, HttpResponseForbidden()
    return dz, None


def _loads_for_date(dz, date):
    return (
        Load.objects
        .filter(dropzone=dz, date=date)
        .select_related('aircraft', 'created_by')
        .prefetch_related('slots__user', 'slots__added_by', 'slots__reservation', 'slots__jump')
        .order_by('load_number')
    )


def _jump_run_current(dz, date):
    return (
        JumpRunConditions.objects
        .filter(dropzone=dz, date=date)
        .order_by('-set_at')
        .first()
    )


def _available_roster(dz, date):
    """Instructors/staff checked in today and marked available."""
    return (
        DailyRoster.objects
        .filter(dropzone=dz, date=date, availability_state='available')
        .select_related('user')
        .prefetch_related('user__instructor_ratings')
        .order_by('user__last_name', 'user__first_name')
    )


def _board_context(dz, date):
    loads = list(_loads_for_date(dz, date))
    # Group slots by group_key within each load
    for load in loads:
        slots = list(load.slots.all())
        # Annotate with group label (A, B, C…) for display
        key_map = {}
        label_index = 0
        for slot in slots:
            if slot.group_key is not None:
                if slot.group_key not in key_map:
                    key_map[slot.group_key] = chr(65 + label_index % 26)
                    label_index += 1
                slot.group_label = key_map[slot.group_key]
            else:
                slot.group_label = None
        load.annotated_slots = slots
    return loads


def _manifest_page_context(request, dz, date):
    return {
        'dz': dz,
        'date': date,
        'prev_date': date - datetime.timedelta(days=1),
        'next_date': date + datetime.timedelta(days=1),
        'today': timezone.localdate(),
        'op_dzs': _get_user_op_dzs(request.user),
        'is_admin': _check_access(request.user, dz, min_role='admin'),
        'in_dz_ops': True,
    }


def _render_board(request, dz, date):
    loads = _board_context(dz, date)
    jump_run = _jump_run_current(dz, date)
    roster = _available_roster(dz, date)
    ctx = {
        'dz': dz,
        'date': date,
        'loads': loads,
        'jump_run': jump_run,
        'available_roster': roster,
        'role_choices': LoadSlot.ROLE_CHOICES,
        'jump_type_choices': LoadSlot.JUMP_TYPE_CHOICES,
    }
    return render(request, 'dz/partials/manifest_board.html', ctx)


# ---------------------------------------------------------------------------
# Full page
# ---------------------------------------------------------------------------

@login_required
def dz_manifest(request, dz_id):
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    ctx = _manifest_page_context(request, dz, date)
    ctx.update({
        'loads': _board_context(dz, date),
        'jump_run': _jump_run_current(dz, date),
        'available_roster': _available_roster(dz, date),
        'role_choices': LoadSlot.ROLE_CHOICES,
        'jump_type_choices': LoadSlot.JUMP_TYPE_CHOICES,
    })
    return render(request, 'dz/manifest.html', ctx)


# ---------------------------------------------------------------------------
# HTMX board fragment (polled every 20s and returned by all action views)
# ---------------------------------------------------------------------------

@login_required
def dz_manifest_board(request, dz_id):
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    return _render_board(request, dz, date)


# ---------------------------------------------------------------------------
# Load actions
# ---------------------------------------------------------------------------

@login_required
def dz_manifest_load_new(request, dz_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    from aircraft.models import Aircraft
    aircraft_id = request.POST.get('aircraft')
    aircraft = None
    if aircraft_id:
        try:
            aircraft = Aircraft.objects.get(pk=aircraft_id)
        except Aircraft.DoesNotExist:
            pass
    load_service.create_load(dz=dz, date=date, created_by=request.user, aircraft=aircraft)
    return _render_board(request, dz, date)


@login_required
def dz_manifest_load_call(request, dz_id, load_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    load = get_object_or_404(Load, pk=load_id, dropzone=dz)
    try:
        load_service.call_load(load, called_by=request.user)
    except ValueError:
        pass
    return _render_board(request, dz, date)


@login_required
def dz_manifest_load_uncall(request, dz_id, load_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    load = get_object_or_404(Load, pk=load_id, dropzone=dz)
    try:
        load_service.uncall_load(load, user=request.user)
    except ValueError:
        pass
    return _render_board(request, dz, date)


@login_required
def dz_manifest_load_depart(request, dz_id, load_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    load = get_object_or_404(Load, pk=load_id, dropzone=dz)
    try:
        load_service.depart_load(load, user=request.user)
    except ValueError:
        pass
    return _render_board(request, dz, date)


@login_required
def dz_manifest_load_land(request, dz_id, load_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    load = get_object_or_404(Load, pk=load_id, dropzone=dz)
    try:
        load_service.land_load(load, user=request.user)
    except ValueError:
        pass
    return _render_board(request, dz, date)


# ---------------------------------------------------------------------------
# Slot actions
# ---------------------------------------------------------------------------

@login_required
def dz_manifest_slot_add(request, dz_id, load_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    load = get_object_or_404(Load, pk=load_id, dropzone=dz)

    if load.status not in ('building', 'on_call'):
        return _render_board(request, dz, date)

    user_id = request.POST.get('user')
    guest_name = request.POST.get('guest_name', '').strip()
    role = request.POST.get('role', '')
    jump_type = request.POST.get('jump_type', '')
    group_key_raw = request.POST.get('group_key', '').strip()
    weight_raw = request.POST.get('weight_lbs', '').strip()
    exit_alt_raw = request.POST.get('exit_altitude', '').strip()

    user = None
    if user_id:
        try:
            user = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            pass

    if not user and not guest_name:
        return _render_board(request, dz, date)

    slot_kwargs = dict(
        load=load,
        role=role,
        jump_type=jump_type,
        added_by=request.user,
    )
    if user:
        slot_kwargs['user'] = user
    else:
        slot_kwargs['guest_name'] = guest_name
    if group_key_raw.isdigit():
        slot_kwargs['group_key'] = int(group_key_raw)
    if weight_raw.isdigit():
        slot_kwargs['weight_lbs'] = int(weight_raw)
    if exit_alt_raw.isdigit():
        slot_kwargs['exit_altitude'] = int(exit_alt_raw)

    LoadSlot.objects.create(**slot_kwargs)
    return _render_board(request, dz, date)


@login_required
def dz_manifest_slot_remove(request, dz_id, load_id, slot_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    slot = get_object_or_404(LoadSlot, pk=slot_id, load_id=load_id, load__dropzone=dz)
    if slot.load.status in ('building', 'on_call'):
        slot.delete()
    return _render_board(request, dz, date)


# ---------------------------------------------------------------------------
# Jump run
# ---------------------------------------------------------------------------

@login_required
def dz_manifest_jump_run(request, dz_id):
    if request.method != 'POST':
        return redirect('dz_manifest', dz_id=dz_id)
    dz, err = _get_dz_or_403(request, dz_id)
    if err:
        return err
    date = _parse_date(request)
    exit_altitude = request.POST.get('exit_altitude', '').strip()
    spot_notes = request.POST.get('spot_notes', '').strip()
    if exit_altitude.isdigit():
        JumpRunConditions.objects.create(
            dropzone=dz,
            date=date,
            exit_altitude=int(exit_altitude),
            spot_notes=spot_notes,
            set_by=request.user,
        )
    return _render_board(request, dz, date)
