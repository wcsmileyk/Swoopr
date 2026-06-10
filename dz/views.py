import datetime

from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.db import models, transaction
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from organizations.models import Dropzone, DropzoneMembership
from .models import BreakRequest, CheckIn, DailyRoster
from training.models import (
    PROGRAM_TYPE_CHOICES,
    RATING_TYPE_CHOICES,
    STUDENT_JUMP_METHOD_CHOICES,
    DropzoneAuthorization,
    InstructorRating,
    ProgramJump,
    StudentEnrollment,
    StudentJump,
)

User = get_user_model()


def _get_user_op_dzs(user):
    """Return active admin/staff memberships for the nav DZ switcher."""
    return (
        DropzoneMembership.objects
        .filter(user=user, role__in=['admin', 'staff'], active=True)
        .select_related('dropzone')
        .order_by('dropzone__name')
    )


def _check_access(user, dz, min_role='staff'):
    """Return True if user is site staff or has at least min_role at dz."""
    if user.is_staff:
        return True
    roles = ['admin', 'staff'] if min_role == 'staff' else ['admin']
    return DropzoneMembership.objects.filter(
        user=user, dropzone=dz, role__in=roles, active=True
    ).exists()


@login_required
def dz_redirect(request):
    """Redirect to the user's most recent active admin/staff DZ."""
    membership = (
        DropzoneMembership.objects
        .filter(user=request.user, role__in=['admin', 'staff'], active=True)
        .select_related('dropzone')
        .order_by('-granted_date')
        .first()
    )
    if membership:
        return redirect('dz_dashboard', dz_id=membership.dropzone.pk)
    messages.error(request, "You don't have access to any DZ operations.")
    return redirect('dashboard')


@login_required
def dz_dashboard(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    memberships = (
        DropzoneMembership.objects
        .filter(dropzone=dz, active=True)
        .select_related('user')
        .order_by('role', 'user__last_name', 'user__first_name')
    )
    auth_count = DropzoneAuthorization.objects.filter(dropzone=dz, active=True).count()

    return render(request, 'dz/dashboard.html', {
        'dz': dz,
        'memberships': memberships,
        'member_count': memberships.count(),
        'auth_count': auth_count,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_staff_list(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    memberships = (
        DropzoneMembership.objects
        .filter(dropzone=dz, active=True)
        .select_related('user', 'user__profile')
        .prefetch_related(
            models.Prefetch(
                'user__dz_authorizations',
                queryset=DropzoneAuthorization.objects.filter(dropzone=dz, active=True),
                to_attr='this_dz_auths',
            ),
            'user__instructor_ratings',
        )
        .order_by('role', 'user__last_name', 'user__first_name')
    )

    return render(request, 'dz/staff_list.html', {
        'dz': dz,
        'memberships': memberships,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_staff_detail(request, dz_id, user_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    is_admin = _check_access(request.user, dz, min_role='admin')
    staff_user = get_object_or_404(User, pk=user_id)
    membership = get_object_or_404(DropzoneMembership, user=staff_user, dropzone=dz)

    ratings = (
        InstructorRating.objects
        .filter(user=staff_user)
        .prefetch_related('personal_limits', 'qualifications_earned__qualification')
    )
    dz_auths = (
        DropzoneAuthorization.objects
        .filter(user=staff_user, dropzone=dz)
        .select_related('instructor_rating')
    )

    # Build set of existing rating types so the add-rating form offers the remainder
    existing_rating_types = set(ratings.values_list('rating_type', flat=True))
    available_rating_types = [
        (val, label) for val, label in RATING_TYPE_CHOICES
        if val not in existing_rating_types
    ]

    # Build set of already-authorized types so the form can offer the remainder
    existing_auth_types = set(dz_auths.values_list('authorization_type', flat=True))
    available_auth_types = [
        (val, label)
        for val, label in DropzoneAuthorization.AUTHORIZATION_TYPES
        if val not in existing_auth_types
    ]

    return render(request, 'dz/staff_detail.html', {
        'dz': dz,
        'staff_user': staff_user,
        'membership': membership,
        'ratings': ratings,
        'dz_auths': dz_auths,
        'available_rating_types': available_rating_types,
        'available_auth_types': available_auth_types,
        'stage_choices': DropzoneAuthorization.STAGE_CHOICES,
        'is_admin': is_admin,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_add_member(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role='admin'):
        messages.error(request, "Only DZ admins can add members.")
        return redirect('dz_staff_list', dz_id=dz_id)

    search_results = []
    query = ''

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'search':
            query = request.POST.get('query', '').strip()
            if query:
                existing_ids = DropzoneMembership.objects.filter(
                    dropzone=dz, active=True
                ).values_list('user_id', flat=True)
                search_results = (
                    User.objects
                    .filter(
                        models.Q(username__icontains=query) |
                        models.Q(first_name__icontains=query) |
                        models.Q(last_name__icontains=query)
                    )
                    .exclude(pk__in=existing_ids)
                    .select_related('profile')
                    [:20]
                )

        elif action == 'add':
            user_id = request.POST.get('user_id')
            role = request.POST.get('role', 'member')

            if role == 'admin' and not request.user.is_staff:
                messages.error(request, "Only site admins can assign the DZ Admin role.")
                return redirect('dz_staff_list', dz_id=dz_id)

            add_user = get_object_or_404(User, pk=user_id)
            membership, created = DropzoneMembership.objects.get_or_create(
                user=add_user,
                dropzone=dz,
                defaults={'role': role, 'granted_by': request.user},
            )
            if not created:
                membership.role = role
                membership.active = True
                membership.granted_by = request.user
                membership.save()

            name = add_user.get_full_name() or add_user.username
            messages.success(request, f"{name} added as {membership.get_role_display()}.")
            return redirect('dz_staff_detail', dz_id=dz_id, user_id=add_user.pk)

    return render(request, 'dz/add_member.html', {
        'dz': dz,
        'query': query,
        'search_results': search_results,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_set_authorization(request, dz_id, user_id):
    """Add or update a DropzoneAuthorization for a staff member."""
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role='admin'):
        messages.error(request, "Only DZ admins can manage authorizations.")
        return redirect('dz_staff_detail', dz_id=dz_id, user_id=user_id)

    staff_user = get_object_or_404(User, pk=user_id)

    if request.method == 'POST':
        auth_type = request.POST.get('authorization_type')
        stage = request.POST.get('stage', 'restricted')

        instructor_rating = None
        if auth_type != 'Videographer':
            instructor_rating = InstructorRating.objects.filter(
                user=staff_user, rating_type=auth_type
            ).first()

        DropzoneAuthorization.objects.update_or_create(
            user=staff_user,
            dropzone=dz,
            authorization_type=auth_type,
            defaults={
                'stage': stage,
                'active': True,
                'instructor_rating': instructor_rating,
                'cleared_by': request.user,
                'cleared_date': datetime.date.today(),
            },
        )
        messages.success(request, f"{auth_type} authorization updated.")

    return redirect('dz_staff_detail', dz_id=dz_id, user_id=user_id)


@login_required
def dz_set_rating(request, dz_id, user_id):
    """Add or update a USPA InstructorRating for a user. Accessible to staff and admin."""
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role='staff'):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    staff_user = get_object_or_404(User, pk=user_id)

    if request.method == 'POST':
        rating_type = request.POST.get('rating_type')
        earned_date = request.POST.get('earned_date') or None

        InstructorRating.objects.update_or_create(
            user=staff_user,
            rating_type=rating_type,
            defaults={
                'earned_date': earned_date,
                'notes': f'Verified by {request.user.get_full_name() or request.user.username} at {dz.name}',
            },
        )
        messages.success(request, f"{rating_type} rating recorded for {staff_user.get_full_name() or staff_user.username}.")

    return redirect('dz_staff_detail', dz_id=dz_id, user_id=user_id)


@login_required
def dz_update_membership(request, dz_id, user_id):
    """Change a member's role or deactivate their membership."""
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role='admin'):
        messages.error(request, "Only DZ admins can change roles.")
        return redirect('dz_staff_detail', dz_id=dz_id, user_id=user_id)

    staff_user = get_object_or_404(User, pk=user_id)
    membership = get_object_or_404(DropzoneMembership, user=staff_user, dropzone=dz)

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'set_role':
            role = request.POST.get('role')
            if role == 'admin' and not request.user.is_staff:
                messages.error(request, "Only site admins can assign the DZ Admin role.")
            else:
                membership.role = role
                membership.save()
                messages.success(request, f"Role updated to {membership.get_role_display()}.")

        elif action == 'deactivate':
            membership.active = False
            membership.save()
            messages.success(request, f"{staff_user.get_full_name() or staff_user.username} removed from DZ.")
            return redirect('dz_staff_list', dz_id=dz_id)

    return redirect('dz_staff_detail', dz_id=dz_id, user_id=user_id)


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

@login_required
def dz_student_list(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    status_filter = request.GET.get('status', 'active')
    program_filter = request.GET.get('program', '')

    enrollments = (
        StudentEnrollment.objects
        .filter(dropzone=dz)
        .select_related('student', 'current_jump', 'current_jump__category')
        .order_by('program_type', 'student__last_name', 'student__first_name')
    )
    if status_filter:
        enrollments = enrollments.filter(status=status_filter)
    if program_filter:
        enrollments = enrollments.filter(program_type=program_filter)

    return render(request, 'dz/student_list.html', {
        'dz': dz,
        'enrollments': enrollments,
        'status_filter': status_filter,
        'program_filter': program_filter,
        'program_choices': PROGRAM_TYPE_CHOICES,
        'status_choices': StudentEnrollment.STATUS_CHOICES,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_enroll_student(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    search_results = []
    query = ''

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'search':
            query = request.POST.get('query', '').strip()
            if query:
                search_results = (
                    User.objects
                    .filter(
                        models.Q(username__icontains=query) |
                        models.Q(first_name__icontains=query) |
                        models.Q(last_name__icontains=query)
                    )
                    .select_related('profile')
                    [:20]
                )

        elif action == 'enroll':
            student = get_object_or_404(User, pk=request.POST.get('user_id'))
            program_type = request.POST.get('program_type')
            enrollment, created = StudentEnrollment.objects.get_or_create(
                student=student,
                dropzone=dz,
                program_type=program_type,
                defaults={
                    'enrolled_by': request.user,
                    'notes': request.POST.get('notes', ''),
                },
            )
            name = student.get_full_name() or student.username
            if not created:
                messages.warning(request, f"{name} is already enrolled in {program_type}.")
            else:
                messages.success(request, f"{name} enrolled in {program_type}.")
            return redirect('dz_student_detail', dz_id=dz_id, enrollment_id=enrollment.pk)

    return render(request, 'dz/enroll_student.html', {
        'dz': dz,
        'query': query,
        'search_results': search_results,
        'program_choices': PROGRAM_TYPE_CHOICES,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_student_detail(request, dz_id, enrollment_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    enrollment = get_object_or_404(
        StudentEnrollment.objects.select_related(
            'student', 'enrolled_by', 'current_jump', 'current_jump__category'
        ),
        pk=enrollment_id, dropzone=dz,
    )
    student_jumps = (
        StudentJump.objects
        .filter(enrollment=enrollment)
        .select_related('program_jump', 'program_jump__category', 'instructor')
        .order_by('-jump_date')
    )
    program_jumps = (
        ProgramJump.objects
        .filter(dropzone=dz, category__program_type=enrollment.program_type, is_active=True)
        .select_related('category')
        .order_by('category__order', 'method_group', 'jump_number')
    )

    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'set_status':
            enrollment.status = request.POST.get('status')
            enrollment.save()
            messages.success(request, "Status updated.")
        elif action == 'set_current_jump':
            enrollment.current_jump = get_object_or_404(
                ProgramJump, pk=request.POST.get('program_jump_id'), dropzone=dz
            )
            enrollment.save()
            messages.success(request, "Current jump updated.")
        return redirect('dz_student_detail', dz_id=dz_id, enrollment_id=enrollment_id)

    return render(request, 'dz/student_detail.html', {
        'dz': dz,
        'enrollment': enrollment,
        'student_jumps': student_jumps,
        'program_jumps': program_jumps,
        'status_choices': StudentEnrollment.STATUS_CHOICES,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


@login_required
def dz_log_student_jump(request, dz_id, enrollment_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    enrollment = get_object_or_404(StudentEnrollment, pk=enrollment_id, dropzone=dz)
    program_jumps = (
        ProgramJump.objects
        .filter(dropzone=dz, category__program_type=enrollment.program_type, is_active=True)
        .select_related('category')
        .order_by('category__order', 'method_group', 'jump_number')
    )

    selected_jump_id = request.GET.get('jump_id') or enrollment.current_jump_id
    selected_jump = None
    if selected_jump_id:
        selected_jump = (
            ProgramJump.objects
            .filter(pk=selected_jump_id, dropzone=dz)
            .select_related('category')
            .first()
        )

    if request.method == 'POST':
        program_jump = get_object_or_404(ProgramJump, pk=request.POST.get('program_jump_id'), dropzone=dz)
        outcome = request.POST.get('outcome') or None

        criteria_passed = {}
        for group in ('ground', 'freefall', 'canopy', 'AFF'):
            items = request.POST.getlist(f'criteria_{group}')
            if items:
                criteria_passed[group] = items

        StudentJump.objects.create(
            enrollment=enrollment,
            program_jump=program_jump,
            instructor=request.user,
            jump_method=request.POST.get('jump_method'),
            jump_date=request.POST.get('jump_date'),
            outcome=outcome,
            criteria_passed=criteria_passed,
            notes=request.POST.get('notes', ''),
        )

        if outcome == 'pass':
            next_jump = (
                ProgramJump.objects
                .filter(
                    dropzone=dz,
                    category__program_type=enrollment.program_type,
                    is_active=True,
                )
                .exclude(pk=program_jump.pk)
                .filter(
                    models.Q(category__order__gt=program_jump.category.order) |
                    models.Q(
                        category__order=program_jump.category.order,
                        method_group=program_jump.method_group,
                        jump_number__gt=program_jump.jump_number,
                    )
                )
                .order_by('category__order', 'method_group', 'jump_number')
                .first()
            )
            if next_jump:
                enrollment.current_jump = next_jump
                enrollment.save()

        messages.success(request, "Student jump logged.")
        return redirect('dz_student_detail', dz_id=dz_id, enrollment_id=enrollment_id)

    return render(request, 'dz/log_student_jump.html', {
        'dz': dz,
        'enrollment': enrollment,
        'program_jumps': program_jumps,
        'selected_jump': selected_jump,
        'jump_method_choices': STUDENT_JUMP_METHOD_CHOICES,
        'outcome_choices': StudentJump.OUTCOME_CHOICES,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Helpers — date parsing, 30-day average
# ---------------------------------------------------------------------------

def _parse_date(request):
    """Return a date from ?date=YYYY-MM-DD, defaulting to today."""
    date_str = request.GET.get('date')
    if date_str:
        try:
            return datetime.date.fromisoformat(date_str)
        except (ValueError, TypeError):
            pass
    return timezone.localdate()


def _thirty_day_avg(user, as_of_date):
    """Jump count over the 30 days prior to as_of_date, expressed as a daily average."""
    from logbook.models import Jump
    count = Jump.objects.filter(
        user=user,
        jump_date__gte=as_of_date - datetime.timedelta(days=30),
        jump_date__lt=as_of_date,
    ).count()
    return round(count / 30, 1)


# ---------------------------------------------------------------------------
# Daily Roster
# ---------------------------------------------------------------------------

@login_required
def dz_roster(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    is_admin = _check_access(request.user, dz, min_role='admin')
    date = _parse_date(request)
    date_str = date.isoformat()
    search_results = []
    query = ''

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'search':
            query = request.POST.get('query', '').strip()
            if query:
                on_roster_ids = DailyRoster.objects.filter(
                    dropzone=dz, date=date
                ).values_list('user_id', flat=True)
                search_results = (
                    User.objects
                    .filter(
                        models.Q(username__icontains=query) |
                        models.Q(first_name__icontains=query) |
                        models.Q(last_name__icontains=query)
                    )
                    .filter(instructor_ratings__isnull=False)
                    .exclude(pk__in=on_roster_ids)
                    .distinct()
                    .select_related('profile')
                    [:20]
                )

        elif action == 'add':
            add_user = get_object_or_404(User, pk=request.POST.get('user_id'))
            check_in, _ = CheckIn.objects.get_or_create(
                dropzone=dz, user=add_user, date=date,
                defaults={'checked_in_by': request.user},
            )
            DailyRoster.objects.get_or_create(
                dropzone=dz, user=add_user, date=date,
                defaults={
                    'check_in': check_in,
                    'thirty_day_avg_at_checkin': _thirty_day_avg(add_user, date),
                },
            )
            name = add_user.get_full_name() or add_user.username
            messages.success(request, f"{name} added to roster.")
            return redirect(f"{request.path}?date={date_str}")

        elif action == 'set_availability':
            entry = get_object_or_404(DailyRoster, pk=request.POST.get('roster_id'), dropzone=dz)
            entry.availability_state = request.POST.get('availability_state')
            entry.save()
            return redirect(f"{request.path}?date={date_str}")

        elif action == 'set_jumps':
            entry = get_object_or_404(DailyRoster, pk=request.POST.get('roster_id'), dropzone=dz)
            try:
                entry.jump_count_today = max(0, int(request.POST.get('jump_count_today', 0)))
            except (ValueError, TypeError):
                pass
            entry.save()
            return redirect(f"{request.path}?date={date_str}")

        elif action == 'remove':
            DailyRoster.objects.filter(pk=request.POST.get('roster_id'), dropzone=dz).delete()
            return redirect(f"{request.path}?date={date_str}")

        elif action == 'request_break':
            entry = get_object_or_404(DailyRoster, pk=request.POST.get('roster_id'), dropzone=dz)
            if entry.user != request.user and not is_admin:
                messages.error(request, "You can only request breaks for yourself.")
            else:
                BreakRequest.objects.create(
                    dropzone=dz,
                    instructor=entry.user,
                    date=date,
                    roster=entry,
                    reason=request.POST.get('reason', ''),
                )
                messages.success(request, "Break request submitted.")
            return redirect(f"{request.path}?date={date_str}")

        elif action in ('approve_break', 'deny_break'):
            if not is_admin:
                messages.error(request, "Only DZ admins can review break requests.")
            else:
                br = get_object_or_404(BreakRequest, pk=request.POST.get('break_id'), dropzone=dz)
                if action == 'approve_break':
                    br.approve(request.user)
                    messages.success(request, "Break approved.")
                else:
                    br.deny(request.user)
                    messages.success(request, "Break denied.")
            return redirect(f"{request.path}?date={date_str}")

    roster = (
        DailyRoster.objects
        .filter(dropzone=dz, date=date)
        .select_related('user', 'check_in')
        .prefetch_related('user__instructor_ratings', 'break_requests')
        .order_by('user__last_name', 'user__first_name')
    )
    pending_breaks = (
        BreakRequest.objects
        .filter(dropzone=dz, date=date, status='pending')
        .select_related('instructor', 'roster')
        .order_by('created_at')
    )

    return render(request, 'dz/roster.html', {
        'dz': dz,
        'date': date,
        'prev_date': date - datetime.timedelta(days=1),
        'next_date': date + datetime.timedelta(days=1),
        'is_today': date == timezone.localdate(),
        'roster': roster,
        'pending_breaks': pending_breaks,
        'query': query,
        'search_results': search_results,
        'availability_choices': DailyRoster.AVAILABILITY_CHOICES,
        'is_admin': is_admin,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Check-In
# ---------------------------------------------------------------------------

@login_required
def dz_checkin(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    date = _parse_date(request)
    date_str = date.isoformat()
    search_results = []
    query = ''

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'search':
            query = request.POST.get('query', '').strip()
            if query:
                already_in_ids = CheckIn.objects.filter(
                    dropzone=dz, date=date
                ).values_list('user_id', flat=True)
                search_results = (
                    User.objects
                    .filter(
                        models.Q(username__icontains=query) |
                        models.Q(first_name__icontains=query) |
                        models.Q(last_name__icontains=query)
                    )
                    .exclude(pk__in=already_in_ids)
                    .select_related('profile')
                    [:20]
                )

        elif action == 'checkin':
            add_user = get_object_or_404(User, pk=request.POST.get('user_id'))
            _, created = CheckIn.objects.get_or_create(
                dropzone=dz, user=add_user, date=date,
                defaults={'checked_in_by': request.user},
            )
            name = add_user.get_full_name() or add_user.username
            if created:
                messages.success(request, f"{name} checked in.")
            else:
                messages.info(request, f"{name} was already checked in.")
            return redirect(f"{request.path}?date={date_str}")

        elif action == 'undo':
            CheckIn.objects.filter(pk=request.POST.get('checkin_id'), dropzone=dz).delete()
            return redirect(f"{request.path}?date={date_str}")

    check_ins = (
        CheckIn.objects
        .filter(dropzone=dz, date=date)
        .select_related('user', 'checked_in_by')
        .order_by('-checked_in_at')
    )

    return render(request, 'dz/checkin.html', {
        'dz': dz,
        'date': date,
        'prev_date': date - datetime.timedelta(days=1),
        'next_date': date + datetime.timedelta(days=1),
        'is_today': date == timezone.localdate(),
        'check_ins': check_ins,
        'query': query,
        'search_results': search_results,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Operating Schedule
# ---------------------------------------------------------------------------

@login_required
def dz_schedule(request, dz_id):
    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz, min_role='admin'):
        messages.error(request, "Only DZ admins can edit the operating schedule.")
        return redirect('dz_dashboard', dz_id=dz_id)

    from .models import OperatingDay

    # Ensure all 7 days exist
    for dow in range(7):
        OperatingDay.objects.get_or_create(dropzone=dz, day_of_week=dow)

    if request.method == 'POST':
        for dow in range(7):
            day = OperatingDay.objects.get(dropzone=dz, day_of_week=dow)
            day.is_open = f'open_{dow}' in request.POST

            time_str = request.POST.get(f'first_load_{dow}', '').strip()
            if time_str:
                try:
                    h, m = time_str.split(':')
                    day.first_load_time = datetime.time(int(h), int(m))
                except (ValueError, TypeError):
                    day.first_load_time = None
            else:
                day.first_load_time = None

            try:
                day.reservation_cutoff_hours = max(0, int(request.POST.get(f'cutoff_{dow}', 24)))
            except (ValueError, TypeError):
                day.reservation_cutoff_hours = 24

            capacities = {}
            for jump_type in ('tandem', 'aff', 'coach'):
                raw = request.POST.get(f'cap_{jump_type}_{dow}', '').strip()
                if raw:
                    try:
                        capacities[jump_type] = max(0, int(raw))
                    except (ValueError, TypeError):
                        pass
            day.slot_capacities = capacities
            day.notes = request.POST.get(f'notes_{dow}', '').strip()
            day.save()

        messages.success(request, "Schedule saved.")
        return redirect('dz_schedule', dz_id=dz_id)

    days = OperatingDay.objects.filter(dropzone=dz).order_by('day_of_week')

    return render(request, 'dz/schedule.html', {
        'dz': dz,
        'days': days,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Reservations — helpers
# ---------------------------------------------------------------------------

def _availability_summary(dz, date):
    """
    Slot capacity vs booked counts for each jump type on a given date.
    Returns dict keyed by jump_type with keys: capacity, booked, available, over.
    """
    from .models import OperatingDay, Reservation as Res
    try:
        op_day = OperatingDay.objects.get(dropzone=dz, day_of_week=date.weekday())
        capacities = op_day.slot_capacities or {}
    except OperatingDay.DoesNotExist:
        capacities = {}

    active = ['confirmed', 'checked_in', 'jumped']
    summary = {}
    for jt in ('tandem', 'aff', 'coach'):
        cap = capacities.get(jt)
        booked = Res.objects.filter(dropzone=dz, date=date, jump_type=jt, status__in=active).count()
        summary[jt] = {
            'capacity': cap,
            'booked': booked,
            'available': (cap - booked) if cap is not None else None,
            'over': cap is not None and booked >= cap,
        }
    return summary


def _assignable_instructors(dz, date, jump_type=None):
    """
    Users with the appropriate InstructorRating who are staff/admin at this DZ.
    Prefers users on the roster for the date; returns all qualified staff otherwise.
    """
    RATING_MAP = {'tandem': 'TI', 'aff': 'AFF-I', 'coach': 'Coach'}

    qs = (
        User.objects
        .filter(
            dz_memberships__dropzone=dz,
            dz_memberships__role__in=['admin', 'staff'],
            dz_memberships__active=True,
            instructor_ratings__isnull=False,
        )
        .distinct()
    )
    if jump_type and jump_type in RATING_MAP:
        qs = qs.filter(instructor_ratings__rating_type=RATING_MAP[jump_type])

    return qs.order_by('last_name', 'first_name')


# ---------------------------------------------------------------------------
# Reservations — list
# ---------------------------------------------------------------------------

@login_required
def dz_reservations(request, dz_id):
    from .models import Reservation

    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    date = _parse_date(request)
    date_str = date.isoformat()
    status_filter = request.GET.get('status', '')

    TRANSITION_MAP = {
        'confirm':  'confirmed',
        'checkin':  'checked_in',
        'jumped':   'jumped',
        'cancel':   'cancelled',
        'no_show':  'no_show',
    }

    if request.method == 'POST':
        action = request.POST.get('action')
        res = get_object_or_404(Reservation, pk=request.POST.get('reservation_id'), dropzone=dz)
        new_status = TRANSITION_MAP.get(action)
        if new_status:
            try:
                res.transition_to(new_status)
                messages.success(request, f"{res.display_name} — {res.get_status_display()}.")
            except ValueError as e:
                messages.error(request, str(e))
        return redirect(f"{request.path}?date={date_str}&status={status_filter}")

    base_qs = Reservation.objects.filter(dropzone=dz, date=date)
    all_count = base_qs.count()
    status_tabs = [
        (val, label, base_qs.filter(status=val).count())
        for val, label in Reservation.STATUS_CHOICES
    ]

    reservations = (
        base_qs.filter(status=status_filter) if status_filter else base_qs
    ).select_related('user', 'assigned_instructor').order_by('time_slot', 'created_at')

    return render(request, 'dz/reservations.html', {
        'dz': dz,
        'date': date,
        'prev_date': date - datetime.timedelta(days=1),
        'next_date': date + datetime.timedelta(days=1),
        'is_today': date == timezone.localdate(),
        'reservations': reservations,
        'status_filter': status_filter,
        'status_tabs': status_tabs,
        'all_count': all_count,
        'availability': _availability_summary(dz, date),
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Reservations — new
# ---------------------------------------------------------------------------

@login_required
def dz_reservation_new(request, dz_id):
    from .models import OperatingDay, Reservation

    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    initial_date = _parse_date(request)
    error = None

    if request.method == 'POST':
        jump_type  = request.POST.get('jump_type', '').strip()
        date_str   = request.POST.get('date', '').strip()
        time_str   = request.POST.get('time_slot', '').strip()
        guest_name  = request.POST.get('guest_name', '').strip()
        guest_email = request.POST.get('guest_email', '').strip()
        guest_phone = request.POST.get('guest_phone', '').strip()
        weight_raw  = request.POST.get('weight_lbs', '').strip()
        source     = request.POST.get('source', 'online')
        notes      = request.POST.get('notes', '').strip()
        instructor_id = request.POST.get('assigned_instructor', '').strip()
        platform_user_id = request.POST.get('platform_user_id', '').strip()

        try:
            res_date = datetime.date.fromisoformat(date_str)
        except (ValueError, TypeError):
            error = "Invalid date."
            res_date = initial_date

        if not error:
            time_slot = None
            if time_str:
                try:
                    h, m = time_str.split(':')
                    time_slot = datetime.time(int(h), int(m))
                except (ValueError, TypeError):
                    pass

            weight_lbs = None
            if weight_raw:
                try:
                    weight_lbs = int(weight_raw)
                except (ValueError, TypeError):
                    pass

            assigned = None
            if instructor_id:
                assigned = User.objects.filter(pk=instructor_id).first()

            platform_user = None
            if platform_user_id:
                platform_user = User.objects.filter(pk=platform_user_id).first()

            Reservation.objects.create(
                dropzone=dz,
                jump_type=jump_type,
                date=res_date,
                time_slot=time_slot,
                user=platform_user,
                guest_name=guest_name,
                guest_email=guest_email,
                guest_phone=guest_phone,
                weight_lbs=weight_lbs,
                status='pending',
                assigned_instructor=assigned,
                source=source,
                notes=notes,
            )
            messages.success(request, f"Reservation created for {guest_name or 'platform user'}.")
            return redirect(f"{request.path[: request.path.index('/reservations/new/')]}reservations/?date={res_date.isoformat()}")

    instructors = _assignable_instructors(dz, initial_date)
    availability = _availability_summary(dz, initial_date)

    return render(request, 'dz/reservation_new.html', {
        'dz': dz,
        'initial_date': initial_date,
        'instructors': instructors,
        'availability': availability,
        'type_choices': Reservation.TYPE_CHOICES,
        'source_choices': Reservation.SOURCE_CHOICES,
        'error': error,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Reservations — detail / edit
# ---------------------------------------------------------------------------

@login_required
def dz_reservation_detail(request, dz_id, reservation_id):
    from .models import Reservation

    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    res = get_object_or_404(Reservation, pk=reservation_id, dropzone=dz)

    TRANSITION_MAP = {
        'confirm':  'confirmed',
        'checkin':  'checked_in',
        'jumped':   'jumped',
        'cancel':   'cancelled',
        'no_show':  'no_show',
    }

    if request.method == 'POST':
        action = request.POST.get('action')

        if action == 'transition':
            new_status = TRANSITION_MAP.get(request.POST.get('to'))
            if new_status:
                try:
                    res.transition_to(new_status)
                    messages.success(request, f"Status updated to {res.get_status_display()}.")
                except ValueError as e:
                    messages.error(request, str(e))

        elif action == 'assign':
            instructor_id = request.POST.get('assigned_instructor', '').strip()
            res.assigned_instructor = User.objects.filter(pk=instructor_id).first() if instructor_id else None
            res.save()
            messages.success(request, "Instructor assigned." if res.assigned_instructor else "Instructor removed.")

        elif action == 'update':
            res.guest_name  = request.POST.get('guest_name', '').strip()
            res.guest_email = request.POST.get('guest_email', '').strip()
            res.guest_phone = request.POST.get('guest_phone', '').strip()
            res.notes       = request.POST.get('notes', '').strip()
            weight_raw = request.POST.get('weight_lbs', '').strip()
            try:
                res.weight_lbs = int(weight_raw) if weight_raw else None
            except (ValueError, TypeError):
                pass
            time_str = request.POST.get('time_slot', '').strip()
            if time_str:
                try:
                    h, m = time_str.split(':')
                    res.time_slot = datetime.time(int(h), int(m))
                except (ValueError, TypeError):
                    res.time_slot = None
            else:
                res.time_slot = None
            res.save()
            messages.success(request, "Reservation updated.")

        return redirect('dz_reservation_detail', dz_id=dz_id, reservation_id=reservation_id)

    TRANSITION_LABELS = {
        'confirmed':  ('confirm', 'Confirm'),
        'checked_in': ('checkin', 'Check In'),
        'jumped':     ('jumped',  'Mark Jumped'),
        'cancelled':  ('cancel',  'Cancel'),
        'no_show':    ('no_show', 'No Show'),
    }
    allowed = [
        (action_key, label)
        for status in Reservation.VALID_TRANSITIONS.get(res.status, [])
        for action_key, label in [TRANSITION_LABELS[status]]
    ]
    instructors = _assignable_instructors(dz, res.date, jump_type=res.jump_type)
    load_slot_count = res.load_slots.count()

    return render(request, 'dz/reservation_detail.html', {
        'dz': dz,
        'res': res,
        'allowed_transitions': allowed,
        'instructors': instructors,
        'load_slot_count': load_slot_count,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Reservation — manifest promotion
# ---------------------------------------------------------------------------

@login_required
def dz_reservation_manifest(request, dz_id, reservation_id):
    """
    Promote a confirmed or checked-in reservation onto an active load.
    Creates LoadSlot(s): always a student slot; instructor slot if assigned.
    """
    from .models import Load, LoadSlot, Reservation

    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    res = get_object_or_404(Reservation, pk=reservation_id, dropzone=dz)

    if res.status not in ('confirmed', 'checked_in'):
        messages.error(request, "Only confirmed or checked-in reservations can be manifested.")
        return redirect('dz_reservation_detail', dz_id=dz_id, reservation_id=reservation_id)

    if request.method == 'POST':
        load_id = request.POST.get('load_id')
        load = get_object_or_404(Load, pk=load_id, dropzone=dz, date=res.date)

        if load.status not in ('building', 'on_call'):
            messages.error(request, f"Load {load.load_number} is {load.get_status_display()} — cannot add slots.")
            return redirect('dz_reservation_detail', dz_id=dz_id, reservation_id=reservation_id)

        if res.load_slots.filter(load=load).exists():
            messages.warning(request, "This reservation is already on that load.")
            return redirect('dz_reservation_detail', dz_id=dz_id, reservation_id=reservation_id)

        with transaction.atomic():
            # Assign a new group_key for this party
            max_key = load.slots.aggregate(models.Max('group_key'))['group_key__max'] or 0
            group_key = max_key + 1

            ROLE_MAP = {'tandem': 'student', 'aff': 'student', 'coach': 'student'}
            role = ROLE_MAP.get(res.jump_type, 'student')

            LoadSlot.objects.create(
                load=load,
                guest_name=res.guest_name,
                user=res.user,
                role=role,
                jump_type=res.jump_type,
                group_key=group_key,
                weight_lbs=res.weight_lbs,
                media_status='requested' if res.add_ons.get('video') else 'none',
                reservation=res,
                added_by=request.user,
            )

            if res.assigned_instructor:
                INSTRUCTOR_ROLE = {'tandem': 'instructor', 'aff': 'instructor', 'coach': 'instructor'}
                LoadSlot.objects.create(
                    load=load,
                    user=res.assigned_instructor,
                    role=INSTRUCTOR_ROLE.get(res.jump_type, 'instructor'),
                    jump_type=res.jump_type,
                    group_key=group_key,
                    reservation=res,
                    added_by=request.user,
                )

        messages.success(request, f"{res.display_name} added to Load {load.load_number}.")
        return redirect('dz_reservation_detail', dz_id=dz_id, reservation_id=reservation_id)

    # GET — show active loads for the date
    active_loads = (
        Load.objects
        .filter(dropzone=dz, date=res.date, status__in=('building', 'on_call'))
        .prefetch_related('slots')
        .order_by('load_number')
    )
    already_manifested = res.load_slots.exists()

    return render(request, 'dz/reservation_manifest.html', {
        'dz': dz,
        'res': res,
        'active_loads': active_loads,
        'already_manifested': already_manifested,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })


# ---------------------------------------------------------------------------
# Pricing config management
# ---------------------------------------------------------------------------

@login_required
def dz_pricing(request, dz_id):
    """
    Staff pricing configuration page.
    POST actions: add_config, add_tier, delete_tier, add_timeslot, delete_timeslot,
                  add_override, delete_override, toggle_config, toggle_timeslot.
    """
    from .models import PricingConfig, PricingOverride, PricingTier, TimeSlotPricing

    dz = get_object_or_404(Dropzone, pk=dz_id)
    if not _check_access(request.user, dz):
        messages.error(request, "You don't have access to this DZ.")
        return redirect('dashboard')

    if request.method == 'POST':
        action = request.POST.get('action', '')

        if action == 'add_config':
            jump_type = request.POST.get('jump_type', '')
            try:
                base = int(float(request.POST.get('base_price', '0').replace('$', '')) * 100)
                minimum = int(float(request.POST.get('min_price', '0').replace('$', '')) * 100)
            except (ValueError, TypeError):
                messages.error(request, "Invalid price value.")
                return redirect('dz_pricing', dz_id=dz_id)
            PricingConfig.objects.get_or_create(
                dropzone=dz, jump_type=jump_type,
                defaults={'base_price_cents': base, 'min_price_cents': minimum},
            )
            messages.success(request, "Pricing config created.")

        elif action == 'update_config':
            config_id = request.POST.get('config_id')
            config = get_object_or_404(PricingConfig, pk=config_id, dropzone=dz)
            try:
                config.base_price_cents = int(float(request.POST.get('base_price', '0').replace('$', '')) * 100)
                config.min_price_cents  = int(float(request.POST.get('min_price', '0').replace('$', '')) * 100)
            except (ValueError, TypeError):
                messages.error(request, "Invalid price value.")
                return redirect('dz_pricing', dz_id=dz_id)
            config.save()
            messages.success(request, "Config updated.")

        elif action == 'toggle_config':
            config = get_object_or_404(PricingConfig, pk=request.POST.get('config_id'), dropzone=dz)
            config.is_active = not config.is_active
            config.save()

        elif action == 'add_tier':
            config = get_object_or_404(PricingConfig, pk=request.POST.get('config_id'), dropzone=dz)
            try:
                min_sold = int(request.POST.get('min_sold', '0'))
                price = int(float(request.POST.get('price', '0').replace('$', '')) * 100)
            except (ValueError, TypeError):
                messages.error(request, "Invalid tier values.")
                return redirect('dz_pricing', dz_id=dz_id)
            label = request.POST.get('label', '').strip()
            PricingTier.objects.update_or_create(
                config=config, min_sold=min_sold,
                defaults={'price_cents': price, 'label': label},
            )
            messages.success(request, "Tier saved.")

        elif action == 'delete_tier':
            tier = get_object_or_404(PricingTier, pk=request.POST.get('tier_id'), config__dropzone=dz)
            tier.delete()
            messages.success(request, "Tier deleted.")

        elif action == 'add_timeslot':
            config = get_object_or_404(PricingConfig, pk=request.POST.get('config_id'), dropzone=dz)
            try:
                surcharge = int(float(request.POST.get('surcharge', '0').replace('$', '')) * 100)
            except (ValueError, TypeError):
                messages.error(request, "Invalid surcharge value.")
                return redirect('dz_pricing', dz_id=dz_id)
            label      = request.POST.get('label', '').strip()
            time_from  = request.POST.get('time_from', '').strip()
            time_to    = request.POST.get('time_to', '').strip()
            days_raw   = request.POST.getlist('days_of_week')
            days       = [int(d) for d in days_raw if d.isdigit()]
            try:
                tf = datetime.time.fromisoformat(time_from)
                tt = datetime.time.fromisoformat(time_to)
            except (ValueError, TypeError):
                messages.error(request, "Invalid time values.")
                return redirect('dz_pricing', dz_id=dz_id)
            TimeSlotPricing.objects.create(
                config=config, label=label,
                time_from=tf, time_to=tt,
                surcharge_cents=surcharge, days_of_week=days,
            )
            messages.success(request, f'"{label}" time slot created.')

        elif action == 'toggle_timeslot':
            ts = get_object_or_404(TimeSlotPricing, pk=request.POST.get('timeslot_id'), config__dropzone=dz)
            ts.is_active = not ts.is_active
            ts.save()

        elif action == 'delete_timeslot':
            ts = get_object_or_404(TimeSlotPricing, pk=request.POST.get('timeslot_id'), config__dropzone=dz)
            ts.delete()
            messages.success(request, "Time slot deleted.")

        elif action == 'add_override':
            config = get_object_or_404(PricingConfig, pk=request.POST.get('config_id'), dropzone=dz)
            try:
                price = int(float(request.POST.get('price', '0').replace('$', '')) * 100)
                date  = datetime.date.fromisoformat(request.POST.get('date', ''))
            except (ValueError, TypeError):
                messages.error(request, "Invalid override values.")
                return redirect('dz_pricing', dz_id=dz_id)
            reason = request.POST.get('reason', '').strip()
            PricingOverride.objects.update_or_create(
                config=config, date=date,
                defaults={'price_cents': price, 'reason': reason, 'created_by': request.user},
            )
            messages.success(request, f"Override set for {date}.")

        elif action == 'delete_override':
            override = get_object_or_404(PricingOverride, pk=request.POST.get('override_id'), config__dropzone=dz)
            override.delete()
            messages.success(request, "Override deleted.")

        return redirect('dz_pricing', dz_id=dz_id)

    configs = (
        PricingConfig.objects
        .filter(dropzone=dz)
        .prefetch_related('tiers', 'time_slots', 'overrides')
        .order_by('jump_type')
    )
    configured_types = {c.jump_type for c in configs}
    remaining_types = [(v, l) for v, l in PricingConfig.JUMP_TYPE_CHOICES if v not in configured_types]

    DAY_NAMES = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    return render(request, 'dz/pricing.html', {
        'dz': dz,
        'configs': configs,
        'remaining_types': remaining_types,
        'day_names': DAY_NAMES,
        'user_dzs': _get_user_op_dzs(request.user),
        'in_dz_ops': True,
    })
