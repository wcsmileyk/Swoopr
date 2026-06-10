"""
Public-facing booking flow — no authentication required.

URL layout:
  /book/<dz_id>/                       booking_home   — date/type/time/guest form
  /book/<dz_id>/confirm/               booking_confirm — review + stub payment
  /book/<dz_id>/thanks/<token>/        booking_thanks  — confirmation page
  /book/<dz_id>/price/                 booking_price   — HTMX fragment: live price update
"""

import datetime
import secrets

from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from organizations.models import Dropzone

from .services.payment_service import capture_payment, create_payment
from .services.pricing_service import get_current_price


SESSION_KEY = 'booking_{dz_id}'


def _session_key(dz_id):
    return f'booking_{dz_id}'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_time(raw):
    if not raw:
        return None
    try:
        h, m = raw.strip().split(':')
        return datetime.time(int(h), int(m))
    except (ValueError, TypeError):
        return None


def _parse_date(raw):
    try:
        return datetime.date.fromisoformat(raw.strip())
    except (ValueError, TypeError, AttributeError):
        return None


def _available_time_slots(dz, jump_type, date):
    """
    Return a list of (time_value, label, price_cents) for valid time slots on this date.
    Builds from OperatingDay.slot_capacities if it's time-keyed, else returns an empty list
    (meaning the DZ accepts any time / no structured slots configured).
    """
    from .models import OperatingDay

    try:
        op_day = OperatingDay.objects.get(dropzone=dz, day_of_week=date.weekday())
    except OperatingDay.DoesNotExist:
        return []

    caps = op_day.slot_capacities or {}
    # If capacity is a flat dict ({"tandem": 8}) rather than time-keyed, no structured slots
    slots = []
    for key, val in caps.items():
        if ':' in key:  # "09:00" style key
            t = _parse_time(key)
            if t:
                price = get_current_price(dz, jump_type, date, time_slot=t)
                cap = val.get(jump_type) if isinstance(val, dict) else None
                slots.append({'time': t, 'label': t.strftime('%-I:%M %p'), 'price_cents': price, 'capacity': cap})
    return slots


# ---------------------------------------------------------------------------
# Step 1: Main booking form
# ---------------------------------------------------------------------------

def booking_home(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    error = None
    today = datetime.date.today()
    min_date = today.isoformat()

    if request.method == 'POST':
        jump_type   = request.POST.get('jump_type', '').strip()
        date_raw    = request.POST.get('date', '').strip()
        time_raw    = request.POST.get('time_slot', '').strip()
        guest_name  = request.POST.get('guest_name', '').strip()
        guest_email = request.POST.get('guest_email', '').strip()
        guest_phone = request.POST.get('guest_phone', '').strip()
        weight_raw  = request.POST.get('weight_lbs', '').strip()
        video       = request.POST.get('video') == 'on'
        photos      = request.POST.get('photos') == 'on'

        res_date = _parse_date(date_raw)
        if not res_date:
            error = 'Please select a valid date.'
        elif res_date < today:
            error = 'Date must be today or in the future.'
        elif jump_type not in ('tandem', 'aff', 'coach'):
            error = 'Please select a jump type.'
        elif not guest_name:
            error = 'Please enter your name.'
        elif not guest_email:
            error = 'Please enter your email.'

        if not error:
            time_slot = _parse_time(time_raw)
            try:
                weight_lbs = int(weight_raw) if weight_raw else None
            except (ValueError, TypeError):
                weight_lbs = None

            price = get_current_price(dz, jump_type, res_date, time_slot)

            request.session[_session_key(dz_id)] = {
                'jump_type':   jump_type,
                'date':        res_date.isoformat(),
                'time_slot':   time_slot.strftime('%H:%M') if time_slot else '',
                'guest_name':  guest_name,
                'guest_email': guest_email,
                'guest_phone': guest_phone,
                'weight_lbs':  weight_lbs,
                'add_ons':     {'video': video, 'photos': photos},
                'price_cents': price,
            }
            return redirect('booking_confirm', dz_id=dz_id)

    # Default date = today or nearest open date
    initial_date = today.isoformat()
    initial_type = request.GET.get('type', 'tandem')
    initial_price = None
    try:
        initial_price = get_current_price(dz, initial_type, today)
    except Exception:
        pass

    initial_price_display = f'${initial_price / 100:.2f}' if initial_price is not None else None
    type_choices = [('tandem', 'Tandem'), ('aff', 'AFF'), ('coach', 'Coach Jump')]

    return render(request, 'booking/home.html', {
        'dz': dz,
        'error': error,
        'today': today,
        'min_date': min_date,
        'initial_date': initial_date,
        'initial_type': initial_type,
        'initial_price': initial_price,
        'initial_price_display': initial_price_display,
        'type_choices': type_choices,
    })


# ---------------------------------------------------------------------------
# HTMX: live price widget
# ---------------------------------------------------------------------------

def booking_price(request, dz_id):
    """Returns a small HTML fragment with the current price. Called via HTMX."""
    dz = get_object_or_404(Dropzone, pk=dz_id)
    jump_type = request.GET.get('type', 'tandem')
    date_raw  = request.GET.get('date', '')
    time_raw  = request.GET.get('time', '')

    price_cents = None
    res_date = _parse_date(date_raw)
    if res_date:
        time_slot = _parse_time(time_raw)
        price_cents = get_current_price(dz, jump_type, res_date, time_slot)

    price_display = f'${price_cents / 100:.2f}' if price_cents is not None else None
    return render(request, 'booking/partials/price_widget.html', {
        'price_cents': price_cents,
        'price_display': price_display,
    })


# ---------------------------------------------------------------------------
# Step 2: Confirm / payment stub
# ---------------------------------------------------------------------------

def booking_confirm(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    session_data = request.session.get(_session_key(dz_id))

    if not session_data:
        return redirect('booking_home', dz_id=dz_id)

    error = None

    if request.method == 'POST':
        # Stub payment — always "succeeds"
        data = session_data
        res_date = datetime.date.fromisoformat(data['date'])
        time_slot = _parse_time(data.get('time_slot'))
        price_cents = data.get('price_cents')

        token = secrets.token_urlsafe(32)

        from .models import Reservation
        reservation = Reservation.objects.create(
            dropzone=dz,
            jump_type=data['jump_type'],
            date=res_date,
            time_slot=time_slot,
            guest_name=data['guest_name'],
            guest_email=data['guest_email'],
            guest_phone=data.get('guest_phone', ''),
            weight_lbs=data.get('weight_lbs'),
            status='confirmed',
            source='online',
            quoted_price_cents=price_cents,
            charged_price_cents=price_cents,
            add_ons=data.get('add_ons', {}),
            confirmation_token=token,
        )

        payment = create_payment(reservation, provider='stub')
        capture_payment(payment)

        del request.session[_session_key(dz_id)]
        return redirect('booking_thanks', dz_id=dz_id, token=token)

    # Re-compute price at confirm time (demand may have shifted)
    res_date = datetime.date.fromisoformat(session_data['date'])
    time_slot = _parse_time(session_data.get('time_slot'))
    current_price = get_current_price(dz, session_data['jump_type'], res_date, time_slot)
    session_data['price_cents'] = current_price
    request.session[_session_key(dz_id)] = session_data
    request.session.modified = True

    price_display = f'${current_price / 100:.2f}' if current_price is not None else None
    return render(request, 'booking/confirm.html', {
        'dz': dz,
        'data': session_data,
        'res_date': res_date,
        'time_slot': time_slot,
        'price_cents': current_price,
        'price_display': price_display,
        'error': error,
    })


# ---------------------------------------------------------------------------
# Step 3: Thanks / confirmation
# ---------------------------------------------------------------------------

def booking_thanks(request, dz_id, token):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    from .models import Reservation
    reservation = get_object_or_404(Reservation, confirmation_token=token, dropzone=dz)
    payment = getattr(reservation, 'payment', None)
    return render(request, 'booking/thanks.html', {
        'dz': dz,
        'reservation': reservation,
        'payment': payment,
    })
