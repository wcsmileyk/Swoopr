import csv
import difflib
import io
import re
import json
import os
import tempfile
from datetime import date, datetime

from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404, redirect, render

from flights.flight_manager import process_flysight_file
from flights.models import Flight
from users.forms import FlightUploadForm
from users.models import Canopy as UserCanopy

from .forms import JumpEditForm, SessionHeaderForm
from .models import Aircraft, Dropzone, InstructorRate, Jump, JumpType, PendingInstructorRequest, StudentSignoff, STUDENT_CATEGORY_CHOICES, JUMP_METHOD_CHOICES


# ---------------------------------------------------------------------------
# CSV Import helpers
# ---------------------------------------------------------------------------

# Maps our internal field names → candidate CSV header substrings (lowercase)
_FIELD_HINTS = {
    'date':           ['date', 'jump date', 'day'],
    'jump_number':    ['jump #', 'jump#', 'jump number', 'jump num', 'number', '#', 'no.', 'jump no'],
    'dropzone':       ['dropzone', 'drop zone', 'dz', 'location', 'place', 'site', 'tunnel'],
    'aircraft':       ['aircraft', 'plane', 'a/c', 'airplane', 'ship', 'plane type'],
    'canopy':         ['canopy', 'main', 'parachute', 'wing', 'canopy/size', 'equipment'],
    'jump_type':      ['type', 'jump type', 'discipline', 'category', 'jump cat'],
    'altitude':       ['altitude', 'alt', 'exit alt', 'exit altitude', 'height'],
    'formation_size': ['formation', 'formation size', 'fs', 'group size', 'team size'],
    'swoop':          ['swoop', 'hook turn', 'ht', 'swoop?'],
    'turn_rotation':  ['rotation', 'turn rotation', 'degrees', 'turn degrees', 'deg'],
    'notes':          ['notes', 'comments', 'remarks', 'description', 'memo'],
}

_DATE_FMTS = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
              '%Y/%m/%d', '%m/%d/%y', '%d/%m/%y', '%b %d %Y', '%d %b %Y']

_FIELD_LABELS = {
    'date':           'Date',
    'jump_number':    'Jump Number',
    'dropzone':       'Dropzone',
    'aircraft':       'Aircraft',
    'canopy':         'Canopy',
    'jump_type':      'Jump Type',
    'altitude':       'Altitude (ft)',
    'formation_size': 'Formation Size',
    'swoop':          'Swoop',
    'turn_rotation':  'Turn Rotation (°)',
    'notes':          'Notes',
}


def _similarity(a, b):
    return difflib.SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def _auto_map_headers(headers):
    """Return {field: col_index} best-guess mapping, ignoring weak matches."""
    mapping = {}
    used_cols = set()
    for field, hints in _FIELD_HINTS.items():
        best_score, best_idx = 0, None
        for idx, header in enumerate(headers):
            if idx in used_cols:
                continue
            h = header.lower().strip()
            for hint in hints:
                score = _similarity(h, hint)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_score >= 0.70 and best_idx is not None:
            mapping[field] = best_idx
            used_cols.add(best_idx)
    return mapping


def _parse_date(raw):
    raw = raw.strip()
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def _parse_bool(raw):
    return raw.strip().lower() in ('yes', 'true', '1', 'x', 'y', 'swoop', 'ht')


def _best_db_match(query, choices_qs, attr='name', threshold=0.72):
    """Return (obj, score) for the best fuzzy match, or (None, 0)."""
    query = query.strip().lower()
    best_obj, best_score = None, 0
    for obj in choices_qs:
        val = str(getattr(obj, attr, '') or '').lower()
        score = _similarity(query, val)
        if score > best_score:
            best_score = score
            best_obj = obj
    if best_score >= threshold:
        return best_obj, best_score
    return None, best_score


def _parse_canopy_raw(raw):
    """
    Try to extract model name and size from a raw CSV canopy string.
    e.g. "Katana 107" → ("Katana", 107), "Sabre 2 170" → ("Sabre 2", 170)
    Returns (model_hint, size_hint_or_None).
    """
    raw = raw.strip()
    m = re.search(r'\b(\d{2,3})\s*(?:sq\.?\s*ft\.?|sqft)?\s*$', raw, re.IGNORECASE)
    if m:
        size = int(m.group(1))
        model = raw[:m.start()].strip().rstrip('-,').strip()
        return model or raw, size
    return raw, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_dz_options(user):
    """Dropzones sorted by user's jump count descending, then alphabetically."""
    dzs = (
        Dropzone.objects
        .annotate(user_jumps=Count('jumps', filter=Q(jumps__user=user)))
        .order_by('-user_jumps', 'name')
    )
    return json.dumps([{'value': dz.id, 'label': dz.name} for dz in dzs])


def _user_defaults(user):
    """Primary canopy id and home DZ id for a user."""
    try:
        primary_canopy_id = user.canopies.filter(is_primary=True).values_list('id', flat=True).first() or ''
    except Exception:
        primary_canopy_id = ''
    try:
        home_dz_id = user.profile.home_dropzone_id or ''
    except Exception:
        home_dz_id = ''
    return primary_canopy_id, home_dz_id


def _extract_gps_datetime(path):
    """
    Quickly read just the first GPS timestamp from a FlySight CSV without full analysis.
    Handles FlySight V1 ($GNSS rows) and standard CSV (time column) formats.
    Returns a datetime or None.
    """
    try:
        with open(path, 'r', errors='replace') as f:
            header_cols = None
            time_idx = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # FlySight V1 format: data rows start with $GNSS
                if line.startswith('$GNSS,'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            return datetime.fromisoformat(parts[1].replace('Z', '+00:00'))
                        except ValueError:
                            continue
                # Skip other $ metadata lines
                if line.startswith('$'):
                    continue
                # Standard CSV: first non-$ line is the header row
                if header_cols is None:
                    cols = [c.strip().lower() for c in line.split(',')]
                    if 'time' in cols:
                        header_cols = cols
                        time_idx = cols.index('time')
                    continue
                # First data row after header
                if time_idx is not None:
                    parts = line.split(',')
                    if len(parts) > time_idx:
                        ts = parts[time_idx].strip()
                        if ts.startswith('('):
                            continue  # skip units row
                        try:
                            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        except ValueError:
                            pass
    except Exception:
        pass
    return None


def _resolve_inline_fields(request, prefix, user):
    """
    Resolve dropzone, aircraft, jump_type, canopy from POST data for a given prefix.
    Inline 'add new' values take precedence over select values.
    Returns a dict: {dropzone_id, aircraft_id, jump_type_id, canopy_id}
    Uses get_or_create for new records so duplicates are prevented.
    """
    def _p(name):
        return request.POST.get(f'{prefix}{name}', '').strip()

    # Dropzone
    new_dz = _p('new_dropzone')
    if new_dz:
        dz, _ = Dropzone.objects.get_or_create(name=new_dz)
        dropzone_id = dz.id
    else:
        dropzone_id = _p('dropzone') or None

    # Aircraft
    new_ac = _p('new_aircraft')
    if new_ac:
        ac, _ = Aircraft.objects.get_or_create(model=new_ac)
        aircraft_id = ac.id
    else:
        aircraft_id = _p('aircraft') or None

    # Jump type
    new_jt = _p('new_jump_type')
    if new_jt:
        jt, _ = JumpType.objects.get_or_create(name=new_jt)
        jump_type_id = jt.id
    else:
        jump_type_id = _p('jump_type') or None

    # Canopy (user-owned — needs model + size)
    new_model = _p('new_canopy_model')
    new_size  = _p('new_canopy_size')
    if new_model and new_size:
        try:
            canopy, _ = UserCanopy.objects.get_or_create(
                user=user,
                model=new_model,
                size=int(new_size),
                defaults={'manufacturer': _p('new_canopy_mfr')},
            )
            canopy_id = canopy.id
        except (ValueError, Exception):
            canopy_id = _p('canopy') or None
    else:
        canopy_id = _p('canopy') or None

    return {
        'dropzone_id': dropzone_id,
        'aircraft_id': aircraft_id,
        'jump_type_id': jump_type_id,
        'canopy_id': canopy_id,
    }


# ---------------------------------------------------------------------------
# Jump list
# ---------------------------------------------------------------------------

_SORT_MAP = {
    'date':       'date',
    'dropzone':   'dropzone__name',
    'aircraft':   'aircraft__model',
    'canopy':     'canopy__model',
    'jump_type':  'jump_type__name',
    'altitude':   'altitude',
    'formation':  'formation_size',
    'jump_num':   'jump_number',
}


@login_required
def jump_list(request):
    jumps = (
        Jump.objects.filter(user=request.user)
        .select_related('dropzone', 'aircraft', 'canopy', 'jump_type', 'student_signoff')
        .prefetch_related('pending_instructor_requests')
    )

    # Filters
    date_from    = request.GET.get('date_from')
    date_to      = request.GET.get('date_to')
    dropzone_id  = request.GET.get('dropzone')
    jump_type_id = request.GET.get('jump_type')
    canopy_id    = request.GET.get('canopy')
    swoop_only   = request.GET.get('swoop_only')

    if date_from:    jumps = jumps.filter(date__gte=date_from)
    if date_to:      jumps = jumps.filter(date__lte=date_to)
    if dropzone_id:  jumps = jumps.filter(dropzone_id=dropzone_id)
    if jump_type_id: jumps = jumps.filter(jump_type_id=jump_type_id)
    if canopy_id:    jumps = jumps.filter(canopy_id=canopy_id)
    if swoop_only:   jumps = jumps.filter(swoop=True)

    # Sorting
    sort_key = request.GET.get('sort', 'jump_num')
    sort_dir = request.GET.get('dir', 'desc')
    if sort_key not in _SORT_MAP:
        sort_key = 'jump_num'
    if sort_dir not in ('asc', 'desc'):
        sort_dir = 'desc'

    sort_field = _SORT_MAP[sort_key]
    if sort_dir == 'desc':
        sort_field = f'-{sort_field}'
    jumps = jumps.order_by(sort_field, '-id')

    context = {
        'jumps': jumps,
        'dropzones': Dropzone.objects.all().order_by('name'),
        'jump_types': JumpType.objects.all().order_by('name'),
        'canopies': request.user.canopies.filter(is_active=True).order_by('model'),
        'filters': request.GET,
        'total_count': jumps.count(),
        'current_sort': sort_key,
        'current_dir': sort_dir,
    }

    if request.headers.get('HX-Request'):
        return render(request, 'logbook/partials/jump_table_body.html', context)

    return render(request, 'logbook/jump_list.html', context)


# ---------------------------------------------------------------------------
# Log jumps
# ---------------------------------------------------------------------------

@login_required
def log_jumps(request):
    primary_canopy_id, home_dz_id = _user_defaults(request.user)

    if request.method == 'POST':
        error = None

        # Resolve header fields (dropzone, aircraft support inline add)
        header = _resolve_inline_fields(request, '', request.user)
        jump_date_raw = request.POST.get('date', '').strip()
        dropzone_id = header['dropzone_id']
        aircraft_id = header['aircraft_id']

        if not jump_date_raw or not dropzone_id:
            error = 'Please fill in the date and dropzone.'
        else:
            row_count = int(request.POST.get('row_count', 0))
            jumps_to_create = []

            for i in range(row_count):
                row = _resolve_inline_fields(request, f'row_{i}_', request.user)
                jump_type_id = row['jump_type_id']
                if not jump_type_id:
                    continue

                altitude_raw  = request.POST.get(f'row_{i}_altitude', '').strip()
                turn_raw      = request.POST.get(f'row_{i}_turn_rotation', '').strip()
                formation_raw = request.POST.get(f'row_{i}_formation_size', '').strip()
                notes         = request.POST.get(f'row_{i}_notes', '').strip()

                student_category = request.POST.get(f'row_{i}_student_category', '').strip() or None
                student_jump_method = request.POST.get(f'row_{i}_student_jump_method', '').strip() or None

                jumps_to_create.append(Jump(
                    user=request.user,
                    date=jump_date_raw,
                    dropzone_id=dropzone_id,
                    aircraft_id=aircraft_id,
                    canopy_id=row['canopy_id'],
                    jump_type_id=jump_type_id,
                    altitude=int(altitude_raw) if altitude_raw else None,
                    swoop=bool(request.POST.get(f'row_{i}_swoop')),
                    turn_rotation=int(turn_raw) if turn_raw else None,
                    formation_size=int(formation_raw) if formation_raw else None,
                    notes=notes,
                    student_category=student_category,
                    student_jump_method=student_jump_method,
                ))

            if jumps_to_create:
                Jump.objects.bulk_create(jumps_to_create)
                Jump.renumber_for_user(request.user)
                return redirect('jump_list')

            error = 'No valid jump rows found — make sure each row has a jump type.'

        profile = request.user.profile
        is_student = getattr(profile, 'license_level', None) in ('', None, 'S')
        return render(request, 'logbook/log_jumps.html', {
            'canopies': request.user.canopies.filter(is_active=True).order_by('-is_primary', 'model'),
            'jump_types': JumpType.objects.all().order_by('name'),
            'dz_options_json': _sorted_dz_options(request.user),
            'aircraft_list': Aircraft.objects.all().order_by('model'),
            'primary_canopy_id': primary_canopy_id,
            'home_dz_id': home_dz_id,
            'error': error,
            'is_student': is_student,
            'default_student_method': getattr(profile, 'student_program', '') if is_student else '',
            'student_categories': STUDENT_CATEGORY_CHOICES,
            'jump_methods': JUMP_METHOD_CHOICES,
        })

    today = date.today().isoformat()
    profile = request.user.profile
    is_student = getattr(profile, 'license_level', None) in ('', None, 'S')
    return render(request, 'logbook/log_jumps.html', {
        'canopies': request.user.canopies.filter(is_active=True).order_by('-is_primary', 'model'),
        'jump_types': JumpType.objects.all().order_by('name'),
        'dz_options_json': _sorted_dz_options(request.user),
        'aircraft_list': Aircraft.objects.all().order_by('model'),
        'primary_canopy_id': primary_canopy_id,
        'home_dz_id': home_dz_id,
        'today': today,
        'is_student': is_student,
        'default_student_method': getattr(profile, 'student_program', '') if is_student else '',
        'student_categories': STUDENT_CATEGORY_CHOICES,
        'jump_methods': JUMP_METHOD_CHOICES,
    })


# ---------------------------------------------------------------------------
# Edit / delete a jump
# ---------------------------------------------------------------------------

@login_required
def edit_jump(request, pk):
    jump = get_object_or_404(Jump, pk=pk, user=request.user)

    if request.method == 'POST':
        if request.POST.get('_action') == 'delete':
            jump.delete()
            Jump.renumber_for_user(request.user)
            return redirect('jump_list')

        resolved = _resolve_inline_fields(request, '', request.user)
        date_raw      = request.POST.get('date', '').strip()
        altitude_raw  = request.POST.get('altitude', '').strip()
        turn_raw      = request.POST.get('turn_rotation', '').strip()
        formation_raw = request.POST.get('formation_size', '').strip()

        if date_raw and resolved['dropzone_id']:
            jump.date          = date_raw
            jump.dropzone_id   = resolved['dropzone_id']
            jump.aircraft_id   = resolved['aircraft_id']
            jump.canopy_id     = resolved['canopy_id']
            jump.jump_type_id  = resolved['jump_type_id']
            jump.altitude      = int(altitude_raw) if altitude_raw else None
            jump.swoop         = bool(request.POST.get('swoop'))
            jump.turn_rotation = int(turn_raw) if turn_raw else None
            jump.formation_size = int(formation_raw) if formation_raw else None
            jump.notes         = request.POST.get('notes', '').strip()
            jump.save()
            return redirect('jump_list')

    unlinked_flights = (
        Flight.objects
        .filter(pilot=request.user, jump__isnull=True)
        .order_by('-created_at')
        .values('id', 'created_at', 'filename', 'flight_name', 'is_swoop')
    )

    return render(request, 'logbook/edit_jump.html', {
        'jump': jump,
        'canopies': request.user.canopies.filter(is_active=True).order_by('-is_primary', 'model'),
        'jump_types': JumpType.objects.all().order_by('name'),
        'dz_options_json': _sorted_dz_options(request.user),
        'aircraft_list': Aircraft.objects.all().order_by('model'),
        'unlinked_flights': list(unlinked_flights),
    })


@login_required
def delete_jump(request, pk):
    jump = get_object_or_404(Jump, pk=pk, user=request.user)
    if request.method == 'POST':
        jump.delete()
        Jump.renumber_for_user(request.user)
        return redirect('jump_list')
    return redirect('edit_jump', pk=pk)


@login_required
def bulk_delete_jumps(request):
    if request.method == 'POST':
        ids = request.POST.getlist('jump_ids')
        if ids:
            Jump.objects.filter(pk__in=ids, user=request.user).delete()
            Jump.renumber_for_user(request.user)
    return redirect('jump_list')


@login_required
def attach_gps(request, pk):
    """Upload a GPS file and link the resulting Flight to an existing jump."""
    jump = get_object_or_404(Jump, pk=pk, user=request.user)
    if request.method != 'POST':
        return redirect('edit_jump', pk=pk)

    gps_file = request.FILES.get('gps_file')
    if not gps_file:
        return redirect('edit_jump', pk=pk)

    try:
        canopy_obj = jump.canopy  # reuse canopy already on the jump, if any
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            for chunk in gps_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        flight = process_flysight_file(tmp_path, pilot=request.user, canopy=canopy_obj)
        os.unlink(tmp_path)

        # If jump is flagged as a swoop but analysis disagrees, honour the jump
        if jump.swoop and not flight.is_swoop:
            flight.is_swoop = True
            flight.save(update_fields=['is_swoop'])

        # Unlink any previous flight first (OneToOne constraint)
        if jump.flight_id:
            old_flight = jump.flight
            jump.flight = None
            jump.save(update_fields=['flight'])
            # Leave the old Flight record intact — just detached

        jump.flight = flight
        jump.save(update_fields=['flight'])

    except Exception:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return redirect('edit_jump', pk=pk)


@login_required
def link_flight(request, pk):
    """Link an existing unattached Flight record to a jump."""
    jump = get_object_or_404(Jump, pk=pk, user=request.user)
    if request.method != 'POST':
        return redirect('edit_jump', pk=pk)

    flight_id = request.POST.get('flight_id', '').strip()
    if flight_id:
        flight = Flight.objects.filter(pk=flight_id, pilot=request.user, jump__isnull=True).first()
        if flight:
            jump.flight = flight
            jump.save(update_fields=['flight'])

    return redirect('edit_jump', pk=pk)


@login_required
def unlink_flight(request, pk):
    """Detach the GPS flight from a jump (leaves the Flight record intact)."""
    jump = get_object_or_404(Jump, pk=pk, user=request.user)
    if request.method == 'POST' and jump.flight_id:
        jump.flight = None
        jump.save(update_fields=['flight'])
    return redirect('edit_jump', pk=pk)


# ---------------------------------------------------------------------------
# GPS Upload Wizard
# ---------------------------------------------------------------------------

@login_required
def upload_wizard_step1(request):
    """
    Step 1: Save uploaded GPS files to temp storage and extract just the date/time.
    Analysis is deferred to step 2 so the user can fill in logbook details first.
    """
    if request.method == 'POST':
        form = FlightUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            files = form.cleaned_data['file']
            default_canopy = form.cleaned_data.get('canopy')
            if not isinstance(files, list):
                files = [files] if files else []

            pending = []
            for f in files:
                entry = {
                    'filename': f.name,
                    'temp_path': None,
                    'date': date.today().isoformat(),
                    'canopy_id': default_canopy.id if default_canopy else None,
                    'save_error': None,   # error saving/reading the file before step 2
                }
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
                        for chunk in f.chunks():
                            tmp.write(chunk)
                        entry['temp_path'] = tmp.name

                    # Extract GPS date without running full analysis
                    gps_dt = _extract_gps_datetime(entry['temp_path'])
                    if gps_dt:
                        entry['date'] = gps_dt.date().isoformat()
                        entry['sort_dt'] = gps_dt.isoformat()

                except Exception as e:
                    if entry['temp_path'] and os.path.exists(entry['temp_path']):
                        os.unlink(entry['temp_path'])
                        entry['temp_path'] = None
                    entry['save_error'] = str(e)

                pending.append(entry)

            # Sort oldest → newest so jump numbers are assigned in chronological order
            pending.sort(key=lambda e: e.get('sort_dt') or e.get('date') or '')

            request.session['pending_jump_flights'] = pending
            return redirect('logbook_upload_log')
    else:
        form = FlightUploadForm(user=request.user)

    return render(request, 'logbook/upload_step1.html', {'form': form})


@login_required
def upload_wizard_step2(request):
    """
    Step 2: User fills in logbook details (DZ, canopy, swoop flag, etc.).
    On submit: run full GPS analysis, create Flight + Jump atomically.
    """
    pending = request.session.get('pending_jump_flights')
    if not pending:
        return redirect('logbook_upload')

    if request.method == 'POST':
        del request.session['pending_jump_flights']

        # Resolve shared dropzone and aircraft (apply to all cards)
        shared_new_dz = request.POST.get('shared_new_dropzone', '').strip()
        if shared_new_dz:
            dz, _ = Dropzone.objects.get_or_create(name=shared_new_dz)
            shared_dz_id = str(dz.id)
        else:
            shared_dz_id = request.POST.get('shared_dropzone', '').strip() or None

        shared_new_ac = request.POST.get('shared_new_aircraft', '').strip()
        if shared_new_ac:
            ac, _ = Aircraft.objects.get_or_create(model=shared_new_ac)
            shared_ac_id = str(ac.id)
        else:
            shared_ac_id = request.POST.get('shared_aircraft', '').strip() or None

        if not shared_dz_id:
            # No dropzone set — clean up all temp files and bounce back
            for entry in pending:
                if entry.get('temp_path') and os.path.exists(entry['temp_path']):
                    os.unlink(entry['temp_path'])
            return redirect('logbook_upload')

        jumps_to_create = []

        with transaction.atomic():
            for i, entry in enumerate(pending):
                resolved = _resolve_inline_fields(request, f'{i}_', request.user)
                # Per-card aircraft overrides shared; DZ always comes from shared header
                aircraft_id = resolved['aircraft_id'] or shared_ac_id

                altitude_raw  = request.POST.get(f'{i}_altitude', '').strip()
                turn_raw      = request.POST.get(f'{i}_turn_rotation', '').strip()
                jump_date_raw = request.POST.get(f'{i}_date', entry['date'])
                user_swoop    = bool(request.POST.get(f'{i}_swoop'))

                # Run full GPS analysis now that we have the user's intent
                flight_id = None
                turn_rotation_val = int(turn_raw) if turn_raw else None

                if entry.get('temp_path') and os.path.exists(entry['temp_path']):
                    try:
                        canopy_obj = None
                        if resolved['canopy_id']:
                            canopy_obj = UserCanopy.objects.filter(
                                pk=resolved['canopy_id'], user=request.user
                            ).first()

                        flight = process_flysight_file(
                            entry['temp_path'],
                            pilot=request.user,
                            canopy=canopy_obj,
                        )
                        os.unlink(entry['temp_path'])
                        flight_id = flight.id

                        # If user checked swoop but analysis didn't detect one, honour the user
                        if user_swoop and not flight.is_swoop:
                            flight.is_swoop = True
                            flight.save(update_fields=['is_swoop'])

                        # Pre-fill turn_rotation from analysis if user left it blank
                        if turn_rotation_val is None and flight.turn_rotation:
                            turn_rotation_val = int(abs(flight.turn_rotation))

                    except Exception:
                        # Analysis failed — still create the jump without GPS
                        if os.path.exists(entry['temp_path']):
                            os.unlink(entry['temp_path'])

                jumps_to_create.append(Jump(
                    user=request.user,
                    date=jump_date_raw,
                    dropzone_id=shared_dz_id,
                    aircraft_id=aircraft_id,
                    canopy_id=resolved['canopy_id'],
                    jump_type_id=resolved['jump_type_id'],
                    altitude=int(altitude_raw) if altitude_raw else None,
                    swoop=user_swoop,
                    turn_rotation=turn_rotation_val,
                    notes=request.POST.get(f'{i}_notes', '').strip(),
                    flight_id=flight_id,
                ))

            if jumps_to_create:
                Jump.objects.bulk_create(jumps_to_create)
                Jump.renumber_for_user(request.user)

        return redirect('jump_list')

    # GET: build context
    canopies = request.user.canopies.filter(is_active=True).order_by('-is_primary', 'model')

    return render(request, 'logbook/upload_step2.html', {
        'pending': pending,
        'canopies': canopies,
        'jump_types': JumpType.objects.all().order_by('name'),
        'aircraft_list': Aircraft.objects.all().order_by('model'),
        'dz_options_json': _sorted_dz_options(request.user),
        'home_dz_id': _user_defaults(request.user)[1],
        'file_count': len(pending),
        'readable_count': sum(1 for e in pending if e.get('temp_path')),
        'error_count': sum(1 for e in pending if e.get('save_error')),
    })


# ---------------------------------------------------------------------------
# CSV Logbook Import Wizard
# ---------------------------------------------------------------------------

@login_required
def import_csv_step1(request):
    """Step 1: Upload CSV, auto-detect headers and field mapping, show preview."""
    error = None

    if request.method == 'POST':
        uploaded = request.FILES.get('csv_file')
        if not uploaded:
            error = 'Please select a CSV file.'
        elif uploaded.size > 10 * 1024 * 1024:
            error = 'File must be under 10 MB.'
        else:
            try:
                raw = uploaded.read().decode('utf-8-sig', errors='replace')
                # Detect delimiter
                dialect = csv.Sniffer().sniff(raw[:4096], delimiters=',\t;|')
                delimiter = dialect.delimiter
            except Exception:
                raw = uploaded.read().decode('utf-8-sig', errors='replace') if hasattr(uploaded, 'read') else raw
                delimiter = ','

            reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
            rows = list(reader)
            if len(rows) < 2:
                error = 'CSV appears to be empty or has only a header row.'
            else:
                headers = [h.strip() for h in rows[0]]
                preview = [[str(c).strip() for c in r] for r in rows[1:6]]
                auto_map = _auto_map_headers(headers)

                request.session['csv_import'] = {
                    'raw': raw,
                    'delimiter': delimiter,
                    'headers': headers,
                    'preview': preview,
                    'total_rows': len(rows) - 1,
                    'auto_map': auto_map,
                }
                return redirect('logbook_import_map')

    return render(request, 'logbook/import_step1.html', {'error': error})


@login_required
def import_csv_step2(request):
    """Step 2: Confirm field mapping, resolve entities, set starting jump number."""
    state = request.session.get('csv_import')
    if not state:
        return redirect('logbook_import')

    headers = state['headers']
    preview = state['preview']
    all_fields = list(_FIELD_HINTS.keys())
    user = request.user

    if request.method == 'POST':
        # Build confirmed mapping: {field: col_index or ''}
        mapping = {}
        for field in all_fields:
            val = request.POST.get(f'map_{field}', '')
            if val != '':
                try:
                    mapping[field] = int(val)
                except ValueError:
                    pass

        if 'date' not in mapping:
            return render(request, 'logbook/import_step2.html', {
                'headers': headers,
                'preview': preview,
                'fields': [(f, _FIELD_LABELS[f]) for f in all_fields],
                'auto_map': state['auto_map'],
                'total_rows': state['total_rows'],
                'error': 'You must map the Date column.',
                'step': 'map',
            })

        # Parse full CSV to collect unique categorical values
        delimiter = state['delimiter']
        reader = csv.reader(io.StringIO(state['raw']), delimiter=delimiter)
        all_rows = list(reader)[1:]  # skip header

        def col(row, field):
            idx = mapping.get(field)
            if idx is None or idx >= len(row):
                return ''
            return row[idx].strip()

        unique_dzs = sorted({col(r, 'dropzone') for r in all_rows if col(r, 'dropzone')})
        unique_aircraft = sorted({col(r, 'aircraft') for r in all_rows if col(r, 'aircraft')})
        unique_jump_types = sorted({col(r, 'jump_type') for r in all_rows if col(r, 'jump_type')})
        unique_canopies = sorted({col(r, 'canopy') for r in all_rows if col(r, 'canopy')})

        # Auto-resolve with fuzzy matching
        dz_qs = Dropzone.objects.all()
        ac_qs = Aircraft.objects.all()
        jt_qs = JumpType.objects.all()
        canopy_qs = user.canopies.filter(is_active=True)

        def resolve_list(unique_vals, qs, attr='name'):
            out = []
            for val in unique_vals:
                obj, score = _best_db_match(val, qs, attr=attr)
                out.append({'raw': val, 'match': obj, 'score': round(score * 100), 'action': 'match' if obj else 'create'})
            return out

        def resolve_canopies(unique_vals, qs):
            canopy_list = list(qs)
            out = []
            for val in unique_vals:
                best_obj, best_score = _best_db_match(val, canopy_list, attr='model')
                # Also try manufacturer+model combined string
                for canopy in canopy_list:
                    s = _similarity(val, f"{canopy.manufacturer} {canopy.model}")
                    if s > best_score:
                        best_score = s
                        best_obj = canopy if s >= 0.72 else None
                model_hint, size_hint = _parse_canopy_raw(val)
                out.append({
                    'raw': val,
                    'match': best_obj,
                    'score': round(best_score * 100),
                    'action': 'match' if best_obj else 'create',
                    'model_hint': model_hint,
                    'size_hint': size_hint or '',
                })
            return out

        # Starting jump number default
        max_jn = Jump.objects.filter(user=user).order_by('-jump_number').values_list('jump_number', flat=True).first()
        default_start = (max_jn or 0) + 1
        try:
            offset = user.profile.jump_number_offset or 1
        except Exception:
            offset = 1
        if default_start == 1:
            default_start = offset

        state['mapping'] = mapping
        request.session['csv_import'] = state

        mapping_summary = [(f, _FIELD_LABELS[f], headers[idx]) for f, idx in mapping.items()]
        return render(request, 'logbook/import_step2.html', {
            'headers': headers,
            'preview': preview,
            'fields': [(f, _FIELD_LABELS[f]) for f in all_fields],
            'mapping': mapping,
            'mapping_summary': mapping_summary,
            'dz_resolutions': resolve_list(unique_dzs, dz_qs),
            'aircraft_resolutions': resolve_list(unique_aircraft, ac_qs, attr='model'),
            'jt_resolutions': resolve_list(unique_jump_types, jt_qs),
            'canopy_resolutions': resolve_canopies(unique_canopies, canopy_qs),
            'user_canopies': list(canopy_qs),
            'total_rows': state['total_rows'],
            'default_start': default_start,
            'step': 'resolve',
        })

    # GET — show field mapping form
    return render(request, 'logbook/import_step2.html', {
        'headers': headers,
        'preview': preview,
        'fields': [(f, _FIELD_LABELS[f]) for f in all_fields],
        'auto_map': state['auto_map'],
        'total_rows': state['total_rows'],
        'step': 'map',
    })


@login_required
def import_csv_confirm(request):
    """Step 3: Execute the import with entity resolutions from POST."""
    if request.method != 'POST':
        return redirect('logbook_import')

    state = request.session.get('csv_import')
    if not state or 'mapping' not in state:
        return redirect('logbook_import')

    user = request.user
    mapping = state['mapping']
    delimiter = state['delimiter']
    reader = csv.reader(io.StringIO(state['raw']), delimiter=delimiter)
    all_rows = list(reader)[1:]

    def col(row, field):
        idx = mapping.get(field)
        if idx is None or idx >= len(row):
            return ''
        return row[idx].strip()

    # Read starting jump number
    try:
        start_num = int(request.POST.get('start_jump_number', '1'))
    except ValueError:
        start_num = 1

    # Build entity lookup maps from POST resolutions
    def build_entity_map(prefix, qs, attr='name', create_fn=None):
        """Returns {raw_value: db_id_or_None}."""
        result = {}
        for key, val in request.POST.items():
            if not key.startswith(f'{prefix}_action_'):
                continue
            raw = key[len(f'{prefix}_action_'):]
            action = val
            if action == 'match':
                match_id = request.POST.get(f'{prefix}_match_{raw}', '')
                result[raw] = int(match_id) if match_id else None
            elif action == 'create':
                if create_fn:
                    obj = create_fn(raw)
                    result[raw] = obj.id if obj else None
                else:
                    result[raw] = None
            else:  # skip
                result[raw] = None
        return result

    with transaction.atomic():
        # Create new DZs
        dz_map = {}
        for key, val in request.POST.items():
            if not key.startswith('dz_action_'):
                continue
            raw = key[len('dz_action_'):]
            action = val
            if action == 'match':
                mid = request.POST.get(f'dz_match_{raw}', '')
                dz_map[raw] = int(mid) if mid else None
            elif action == 'create':
                dz, _ = Dropzone.objects.get_or_create(name=raw)
                dz_map[raw] = dz.id

        # Create new Aircraft
        ac_map = {}
        for key, val in request.POST.items():
            if not key.startswith('ac_action_'):
                continue
            raw = key[len('ac_action_'):]
            action = val
            if action == 'match':
                mid = request.POST.get(f'ac_match_{raw}', '')
                ac_map[raw] = int(mid) if mid else None
            elif action == 'create':
                ac, _ = Aircraft.objects.get_or_create(model=raw)
                ac_map[raw] = ac.id

        # Create new JumpTypes
        jt_map = {}
        for key, val in request.POST.items():
            if not key.startswith('jt_action_'):
                continue
            raw = key[len('jt_action_'):]
            action = val
            if action == 'match':
                mid = request.POST.get(f'jt_match_{raw}', '')
                jt_map[raw] = int(mid) if mid else None
            elif action == 'create':
                jt, _ = JumpType.objects.get_or_create(name=raw)
                jt_map[raw] = jt.id

        # Canopy map — match existing or create new from provided fields
        canopy_map = {}
        for key, val in request.POST.items():
            if not key.startswith('canopy_action_'):
                continue
            raw = key[len('canopy_action_'):]
            action = val
            if action == 'match':
                mid = request.POST.get(f'canopy_match_{raw}', '')
                canopy_map[raw] = int(mid) if mid else None
            elif action == 'create':
                mfr = request.POST.get(f'canopy_mfr_{raw}', '').strip()
                model = request.POST.get(f'canopy_model_{raw}', '').strip()
                size_str = request.POST.get(f'canopy_size_{raw}', '').strip()
                if model and size_str:
                    try:
                        canopy, _ = UserCanopy.objects.get_or_create(
                            user=user,
                            model=model,
                            size=int(size_str),
                            defaults={'manufacturer': mfr},
                        )
                        canopy_map[raw] = canopy.id
                    except (ValueError, Exception):
                        canopy_map[raw] = None
            # action == 'skip': leave out of map → jumps get no canopy

        # Set primary canopy if requested
        primary_raw = request.POST.get('canopy_primary', '')
        if primary_raw and canopy_map.get(primary_raw):
            UserCanopy.objects.filter(user=user, is_primary=True).update(is_primary=False)
            UserCanopy.objects.filter(pk=canopy_map[primary_raw]).update(is_primary=True)

        # Parse rows and create Jump objects
        created = 0
        skipped = 0
        errors = []

        jumps_to_create = []
        for row_idx, row in enumerate(all_rows):
            date_raw = col(row, 'date')
            if not date_raw:
                skipped += 1
                continue
            parsed_date = _parse_date(date_raw)
            if not parsed_date:
                errors.append(f'Row {row_idx + 2}: Could not parse date "{date_raw}"')
                skipped += 1
                continue

            dz_raw = col(row, 'dropzone')
            dz_id = dz_map.get(dz_raw) if dz_raw else None
            if not dz_id:
                skipped += 1
                continue  # dropzone required

            altitude_raw = col(row, 'altitude')
            turn_raw = col(row, 'turn_rotation')
            formation_raw = col(row, 'formation_size')
            swoop_raw = col(row, 'swoop')
            notes_raw = col(row, 'notes')
            ac_raw = col(row, 'aircraft')
            jt_raw = col(row, 'jump_type')
            canopy_raw = col(row, 'canopy')

            try:
                altitude_val = int(altitude_raw) if altitude_raw else None
            except ValueError:
                altitude_val = None
            try:
                turn_val = int(float(turn_raw)) if turn_raw else None
            except ValueError:
                turn_val = None
            try:
                formation_val = int(formation_raw) if formation_raw else None
            except ValueError:
                formation_val = None

            jumps_to_create.append(Jump(
                user=user,
                date=parsed_date,
                dropzone_id=dz_id,
                aircraft_id=ac_map.get(ac_raw) if ac_raw else None,
                jump_type_id=jt_map.get(jt_raw) if jt_raw else None,
                canopy_id=canopy_map.get(canopy_raw) if canopy_raw else None,
                altitude=altitude_val,
                swoop=_parse_bool(swoop_raw) if swoop_raw else False,
                turn_rotation=turn_val,
                formation_size=formation_val,
                notes=notes_raw,
                jump_number=None,  # will be set by renumber below
            ))
            created += 1

        if jumps_to_create:
            # Temporarily set offset on profile then renumber
            try:
                old_offset = user.profile.jump_number_offset
                user.profile.jump_number_offset = start_num
                user.profile.save(update_fields=['jump_number_offset'])
            except Exception:
                old_offset = 1

            Jump.objects.bulk_create(jumps_to_create)
            Jump.renumber_for_user(user)

    # Clean up session
    if 'csv_import' in request.session:
        del request.session['csv_import']

    return render(request, 'logbook/import_results.html', {
        'created': created,
        'skipped': skipped,
        'errors': errors[:20],  # cap at 20 displayed errors
    })


# ---------------------------------------------------------------------------
# Instructor earnings
# ---------------------------------------------------------------------------

@login_required
def instructor_earnings_view(request):
    from django.urls import reverse
    return redirect(reverse('instructor_hub') + '?tab=earnings')



# ---------------------------------------------------------------------------
# Student ISP sign-off views
# ---------------------------------------------------------------------------

from django.contrib.auth import get_user_model
from django.http import JsonResponse
from django.views.decorators.http import require_POST

from .isp_criteria import (
    can_use_coach_method,
    get_category_progress,
    get_criteria_for_jump,
    get_signing_rating,
)
_UserModel = get_user_model()


@login_required
def instructor_search(request):
    """
    AJAX endpoint: returns matching instructors for a given jump method.
    Query params: q (name or license), method (AFF/Tandem/IAD_SL/Coach)
    """
    q = request.GET.get('q', '').strip()
    method = request.GET.get('method', '')
    if not q or len(q) < 2:
        return JsonResponse({'results': []})

    # Determine which profile flag is required
    method_flag_map = {
        'AFF': 'affi',
        'Tandem': 'ti',
        'IAD_SL': 'iad_sl',
        'Coach': None,  # any instructor rating works
    }
    flag = method_flag_map.get(method)

    qs = _UserModel.objects.select_related('profile').filter(
        Q(first_name__icontains=q) |
        Q(last_name__icontains=q) |
        Q(profile__license_number__icontains=q)
    )

    # Filter to users who have a qualifying rating
    if flag:
        qs = qs.filter(**{f'profile__{flag}': True})
    else:
        qs = qs.filter(
            Q(profile__coach=True) | Q(profile__affi=True) |
            Q(profile__ti=True) | Q(profile__iad_sl=True)
        )

    results = []
    for user in qs[:10]:
        p = user.profile
        ratings = []
        if p.affi:    ratings.append('AFF-I')
        if p.ti:      ratings.append('TI')
        if p.iad_sl:  ratings.append('IAD-I')
        if p.coach:   ratings.append('Coach')
        results.append({
            'id': user.id,
            'name': p.display_name,
            'license': p.license_number,
            'ratings': ', '.join(ratings),
        })

    return JsonResponse({'results': results})


@login_required
def request_signoff(request, pk):
    """
    Student selects which criteria they think they passed and submits
    the jump to one or more instructors for sign-off.
    """
    jump = get_object_or_404(Jump, pk=pk, user=request.user)

    if not jump.is_student_jump:
        return redirect('jump_list')

    # If already signed, show it instead
    if hasattr(jump, 'student_signoff'):
        return redirect('view_signoff', pk=pk)

    criteria = get_criteria_for_jump(jump.student_category, jump.student_jump_method)
    existing_requests = jump.pending_instructor_requests.select_related('instructor__profile').all()

    if request.method == 'POST':
        # Build student_criteria dict from POST checkboxes
        student_criteria = {
            'ground':   request.POST.getlist('ground'),
            'freefall': request.POST.getlist('freefall'),
            'canopy':   request.POST.getlist('canopy'),
            'method':   request.POST.getlist('method'),
        }
        instructor_ids = request.POST.getlist('instructor_ids')

        if not instructor_ids:
            return render(request, 'logbook/student_signoff_request.html', {
                'jump': jump,
                'criteria': criteria,
                'existing_requests': existing_requests,
                'error': 'Please select at least one instructor.',
                'method_label': dict(JUMP_METHOD_CHOICES).get(jump.student_jump_method, ''),
                'category_label': dict(STUDENT_CATEGORY_CHOICES).get(jump.student_category, ''),
            })

        created = 0
        for iid in instructor_ids:
            try:
                instructor = _UserModel.objects.get(pk=iid)
                # Verify they can sign this method
                if get_signing_rating(instructor.profile, jump.student_jump_method) is None:
                    continue
                PendingInstructorRequest.objects.get_or_create(
                    jump=jump,
                    instructor=instructor,
                    defaults={'student_criteria': student_criteria},
                )
                created += 1
            except _UserModel.DoesNotExist:
                continue

        if created:
            return redirect('jump_list')

        return render(request, 'logbook/student_signoff_request.html', {
            'jump': jump,
            'criteria': criteria,
            'existing_requests': existing_requests,
            'error': 'None of the selected instructors are qualified to sign this jump type.',
            'method_label': dict(JUMP_METHOD_CHOICES).get(jump.student_jump_method, ''),
            'category_label': dict(STUDENT_CATEGORY_CHOICES).get(jump.student_category, ''),
        })

    return render(request, 'logbook/student_signoff_request.html', {
        'jump': jump,
        'criteria': criteria,
        'existing_requests': existing_requests,
        'method_label': dict(JUMP_METHOD_CHOICES).get(jump.student_jump_method, ''),
        'category_label': dict(STUDENT_CATEGORY_CHOICES).get(jump.student_category, ''),
    })


@login_required
def instructor_queue(request):
    return redirect('instructor_hub')


@login_required
def signoff_form(request, pk):
    """
    Instructor reviews criteria for a pending request and signs off.
    pk = PendingInstructorRequest pk.
    """
    pending_req = get_object_or_404(PendingInstructorRequest, pk=pk, instructor=request.user)
    jump = pending_req.jump

    # Verify this jump isn't already signed
    if hasattr(jump, 'student_signoff'):
        return redirect('instructor_hub')

    rating = get_signing_rating(request.user.profile, jump.student_jump_method)
    if not rating:
        return redirect('instructor_hub')

    criteria = get_criteria_for_jump(jump.student_category, jump.student_jump_method)
    student_criteria = pending_req.student_criteria

    if request.method == 'POST':
        instructor_criteria = {
            'ground':   request.POST.getlist('ground'),
            'freefall': request.POST.getlist('freefall'),
            'canopy':   request.POST.getlist('canopy'),
            'method':   request.POST.getlist('method'),
        }

        # Check if instructor modified the student's pre-fill
        modified = instructor_criteria != student_criteria

        profile = request.user.profile
        signoff = StudentSignoff.objects.create(
            jump=jump,
            signed_by=request.user,
            instructor_name=profile.display_name,
            instructor_license=profile.license_number,
            instructor_rating=rating,
            criteria_passed=instructor_criteria,
            criteria_modified=modified,
        )

        # Remove all pending requests for this jump (first to sign wins)
        jump.pending_instructor_requests.all().delete()

        return redirect('instructor_hub')

    return render(request, 'logbook/signoff_form.html', {
        'pending_req': pending_req,
        'jump': jump,
        'criteria': criteria,
        'student_criteria': student_criteria,
        'rating': rating,
        'method_label': dict(JUMP_METHOD_CHOICES).get(jump.student_jump_method, ''),
        'category_label': dict(STUDENT_CATEGORY_CHOICES).get(jump.student_category, ''),
    })


@login_required
def view_signoff(request, pk):
    """Student views their signed-off jump. Clears the modification notification."""
    jump = get_object_or_404(Jump, pk=pk, user=request.user)
    signoff = get_object_or_404(StudentSignoff, jump=jump)

    # Clear notification once viewed
    if not signoff.student_notified:
        StudentSignoff.objects.filter(pk=signoff.pk).update(student_notified=True)

    criteria = get_criteria_for_jump(jump.student_category, jump.student_jump_method)

    return render(request, 'logbook/view_signoff.html', {
        'jump': jump,
        'signoff': signoff,
        'criteria': criteria,
        'method_label': dict(JUMP_METHOD_CHOICES).get(jump.student_jump_method, ''),
        'category_label': dict(STUDENT_CATEGORY_CHOICES).get(jump.student_category, ''),
    })


@login_required
def instructor_hub(request):
    """Combined instructor view: sign-off queue, history, and earnings."""
    from decimal import Decimal
    from datetime import timedelta
    from collections import defaultdict
    from django.urls import reverse

    active_tab = request.GET.get('tab', 'queue')

    # ── Earnings POST: save rates ────────────────────────────────────────
    if request.method == 'POST' and request.POST.get('action') == 'save_rates':
        for jt in JumpType.objects.all():
            raw = request.POST.get(f'rate_{jt.id}', '').strip()
            if raw:
                try:
                    InstructorRate.objects.update_or_create(
                        user=request.user,
                        jump_type=jt,
                        defaults={'amount': Decimal(raw)},
                    )
                except Exception:
                    pass
            else:
                InstructorRate.objects.filter(user=request.user, jump_type=jt).delete()
        return redirect(reverse('instructor_hub') + '?tab=earnings')

    # ── Queue tab ────────────────────────────────────────────────────────
    pending = (
        PendingInstructorRequest.objects
        .filter(instructor=request.user)
        .select_related('jump__user__profile', 'jump__dropzone')
        .order_by('-created_at')
    )

    # ── History tab ──────────────────────────────────────────────────────
    history = (
        StudentSignoff.objects
        .filter(signed_by=request.user)
        .select_related('jump__user__profile', 'jump__dropzone')
        .order_by('-signed_at')
    )

    # ── Earnings tab ─────────────────────────────────────────────────────
    rates = {r.jump_type_id: r.amount for r in InstructorRate.objects.filter(user=request.user)}
    paid_type_ids = list(rates.keys())

    today = date.today()
    date_from_raw = request.GET.get('from', '')
    date_to_raw = request.GET.get('to', '')
    try:
        date_from = date.fromisoformat(date_from_raw) if date_from_raw else None
    except ValueError:
        date_from = None
    try:
        date_to = date.fromisoformat(date_to_raw) if date_to_raw else None
    except ValueError:
        date_to = None

    base_qs = (
        request.user.jumps
        .filter(jump_type_id__in=paid_type_ids)
        .select_related('dropzone', 'jump_type')
        .order_by('-date', '-jump_number')
    ) if paid_type_ids else Jump.objects.none()

    table_qs = base_qs
    if date_from:
        table_qs = table_qs.filter(date__gte=date_from)
    if date_to:
        table_qs = table_qs.filter(date__lte=date_to)

    table_rows = [{'jump': j, 'amount': rates.get(j.jump_type_id, Decimal('0'))} for j in table_qs]
    table_total = sum(r['amount'] for r in table_rows)

    daily_totals = defaultdict(Decimal)
    for row in table_rows:
        daily_totals[row['jump'].date] += row['amount']
    dow_buckets = defaultdict(list)
    for d, total in daily_totals.items():
        dow_buckets[d.weekday()].append(total)
    dow_averages = [
        round(float(sum(dow_buckets[i]) / len(dow_buckets[i])), 2) if dow_buckets[i] else 0.0
        for i in range(7)
    ]

    seven_days_ago = today - timedelta(days=7)
    last_7_qs = base_qs.filter(date__gte=seven_days_ago)
    last_7_total = sum(rates.get(j.jump_type_id, Decimal('0')) for j in last_7_qs)
    last_7_count = last_7_qs.count()

    last_jump = request.user.jumps.order_by('-date', '-jump_number').first()
    last_jump_day_total = Decimal('0')
    last_jump_day_count = 0
    if last_jump:
        last_day_qs = base_qs.filter(date=last_jump.date)
        last_jump_day_total = sum(rates.get(j.jump_type_id, Decimal('0')) for j in last_day_qs)
        last_jump_day_count = last_day_qs.count()

    return render(request, 'logbook/instructor_hub.html', {
        'active_tab': active_tab,
        # queue
        'pending': pending,
        # history
        'history': history,
        # earnings
        'jump_types_with_rates': [
            (jt, rates.get(jt.id)) for jt in JumpType.objects.all().order_by('name')
        ],
        'has_rates': bool(rates),
        'table_rows': table_rows,
        'table_total': table_total,
        'dow_averages_json': json.dumps(dow_averages),
        'dow_has_data': any(dow_averages),
        'last_7_total': last_7_total,
        'last_7_count': last_7_count,
        'last_jump_date': last_jump.date if last_jump else None,
        'last_jump_day_total': last_jump_day_total,
        'last_jump_day_count': last_jump_day_count,
        'date_from': date_from_raw,
        'date_to': date_to_raw,
    })


@login_required
def student_hub(request):
    """Student app — ISP progress with sign-off info + jump history."""
    active_tab = request.GET.get('tab', 'progress')
    progress = get_category_progress(request.user)
    coach_unlocked = can_use_coach_method(request.user)
    category_labels = dict(STUDENT_CATEGORY_CHOICES)

    # Most recent sign-off per category
    signoff_info = {}
    for so in (
        StudentSignoff.objects
        .filter(jump__user=request.user, jump__student_category__isnull=False)
        .select_related('jump')
        .order_by('signed_at')
    ):
        signoff_info[so.jump.student_category] = {
            'signed_at': so.signed_at,
            'instructor_name': so.instructor_name,
        }

    # Jump history with sign-off status
    student_jumps = (
        Jump.objects
        .filter(user=request.user, student_category__isnull=False)
        .select_related('dropzone', 'jump_type', 'student_signoff')
        .prefetch_related('pending_instructor_requests')
        .order_by('-date', '-created_at')
    )
    jump_history = []
    for jump in student_jumps:
        try:
            so = jump.student_signoff
            status = 'signed'
        except Exception:
            so = None
            status = 'pending' if list(jump.pending_instructor_requests.all()) else 'unsigned'
        jump_history.append({'jump': jump, 'status': status, 'signoff': so})

    return render(request, 'logbook/student_hub.html', {
        'active_tab': active_tab,
        'progress': progress,
        'coach_unlocked': coach_unlocked,
        'category_labels': category_labels,
        'signoff_info': signoff_info,
        'jump_history': jump_history,
    })


@login_required
def student_progress(request):
    """Shows the student's ISP category completion progress."""
    progress = get_category_progress(request.user)
    coach_unlocked = can_use_coach_method(request.user)
    category_labels = dict(STUDENT_CATEGORY_CHOICES)
    return render(request, 'logbook/student_progress.html', {
        'progress': progress,
        'coach_unlocked': coach_unlocked,
        'category_labels': category_labels,
    })
