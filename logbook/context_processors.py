from django.urls import reverse

from .models import PendingInstructorRequest, StudentSignoff

_STUDENT_URLS = {
    'student_hub', 'student_progress', 'request_signoff', 'view_signoff',
}
_SWOOPER_URLS = {
    'swooper_dashboard', 'flights', 'flight_detail', 'upload_flight', 'upload_results',
    'select_swoop_window', 'toggle_data_incorrect', 'toggle_swoop_rejected',
    'update_flight_name', 'update_flight_gate', 'calculate_flight_gate_metrics',
    'upload_gate_file', 'delete_flight', 'bulk_delete_flights',
    'public_swoops', 'user_search', 'public_profile',
}
_INSTRUCTOR_URLS = {
    'instructor_hub', 'instructor_queue', 'signoff_form', 'instructor_earnings',
}


def notifications(request):
    if not request.user.is_authenticated:
        return {
            'notif_count': 0,
            'notif_items': [],
            'show_student': False,
            'show_swooper': False,
            'show_instructor': False,
            'active_section': 'logbook',
        }

    # ── Notifications ────────────────────────────────────────────────────
    items = []

    pending_count = (
        PendingInstructorRequest.objects
        .filter(instructor=request.user)
        .count()
    )
    if pending_count:
        items.append({
            'text': f'{pending_count} sign-off request{"s" if pending_count != 1 else ""} waiting',
            'url': reverse('instructor_hub') + '?tab=queue',
        })

    unread_count = (
        StudentSignoff.objects
        .filter(jump__user=request.user, student_notified=False)
        .count()
    )
    if unread_count:
        items.append({
            'text': f'{unread_count} jump{"s" if unread_count != 1 else ""} signed off',
            'url': reverse('jump_list'),
        })

    # ── Sidebar visibility ───────────────────────────────────────────────
    from logbook.models import Jump
    from flights.models import Flight

    profile = getattr(request.user, 'profile', None)

    show_student = (
        getattr(profile, 'license_level', None) == 'S'
        or Jump.objects.filter(user=request.user, student_category__isnull=False).exists()
    )

    show_swooper = (
        Jump.objects.filter(user=request.user, swoop=True).exists()
        or Flight.objects.filter(pilot=request.user).exists()
    )

    has_rating = any([
        getattr(profile, 'coach', False),
        getattr(profile, 'affi', False),
        getattr(profile, 'ti', False),
        getattr(profile, 'iad_sl', False),
    ])
    show_instructor = (
        has_rating
        or StudentSignoff.objects.filter(signed_by=request.user).exists()
    )

    # ── Active section ───────────────────────────────────────────────────
    url_name = getattr(getattr(request, 'resolver_match', None), 'url_name', '') or ''
    if url_name in _STUDENT_URLS:
        active_section = 'student'
    elif url_name in _SWOOPER_URLS:
        active_section = 'swooper'
    elif url_name in _INSTRUCTOR_URLS:
        active_section = 'instructor'
    else:
        active_section = 'logbook'

    return {
        'notif_count': pending_count + unread_count,
        'notif_items': items,
        'show_student': show_student,
        'show_swooper': show_swooper,
        'show_instructor': show_instructor,
        'active_section': active_section,
    }
