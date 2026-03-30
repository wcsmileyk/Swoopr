from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from django.db import transaction, models
from django.core.files.storage import default_storage
from django.http import JsonResponse
import os
import tempfile
import logging
import json

logger = logging.getLogger('flights.flight_manager')
from .forms import SignUpForm, UserLoginForm, CanopyForm, UserProfileForm, FlightUploadForm, BulkPrivacyForm, UserSearchForm
from .models import Canopy
from flights.models import Flight
from flights.flight_manager import process_flysight_file


def signup_view(request):
    """User registration view"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = SignUpForm(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save()
                    messages.success(request, f'Welcome {user.username}! Your account has been created successfully.')

                    # Log the user in automatically
                    username = form.cleaned_data.get('username')
                    password = form.cleaned_data.get('password1')
                    user = authenticate(username=username, password=password)
                    if user:
                        login(request, user)
                        return redirect('logbook_onboarding')

            except Exception as e:
                messages.error(request, f'There was an error creating your account: {str(e)}')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = SignUpForm()

    return render(request, 'users/signup.html', {'form': form})


def login_view(request):
    """User login view"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            remember_me = form.cleaned_data.get('remember_me', False)

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)

                # Set session expiry based on remember me
                if not remember_me:
                    request.session.set_expiry(0)  # Session expires when browser closes

                messages.success(request, f'Welcome back, {user.get_full_name() or user.username}!')

                # Redirect to next page or dashboard
                next_page = request.GET.get('next', 'dashboard')
                return redirect(next_page)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserLoginForm()

    return render(request, 'users/login.html', {'form': form})


@require_http_methods(["POST", "GET"])
def logout_view(request):
    """User logout view"""
    if request.method == 'POST':
        logout(request)
        messages.success(request, 'You have been logged out successfully.')
        return redirect('login')
    else:
        # For GET requests, show confirmation page
        return render(request, 'users/logout_confirm.html')


@login_required
def dashboard_view(request):
    """Logbook dashboard — activity, stats, jump types, recent jumps."""
    from logbook.models import Jump
    from django.db.models import Count
    from django.db.models.functions import TruncMonth, ExtractYear
    from datetime import date, timedelta
    user = request.user

    # Logbook stats
    logbook_qs = Jump.objects.filter(user=user)
    logbook_total = logbook_qs.count()
    last_jump = logbook_qs.order_by('-date', '-created_at').first()
    recent_jumps = (
        logbook_qs
        .select_related('dropzone', 'jump_type', 'canopy')
        .order_by('-date', '-created_at')[:100]
    )

    today = date.today()

    # Monthly activity — last 12 calendar months
    monthly_labels = []
    monthly_data_map = {}
    for i in range(11, -1, -1):
        m, y = today.month - i, today.year
        while m <= 0:
            m += 12
            y -= 1
        label = date(y, m, 1).strftime('%b %Y')
        monthly_labels.append(label)
        monthly_data_map[label] = 0
    cutoff_m, cutoff_y = today.month - 11, today.year
    while cutoff_m <= 0:
        cutoff_m += 12
        cutoff_y -= 1
    for entry in (
        logbook_qs
        .filter(date__gte=date(cutoff_y, cutoff_m, 1))
        .annotate(month=TruncMonth('date'))
        .values('month').annotate(count=Count('id')).order_by('month')
    ):
        lbl = entry['month'].strftime('%b %Y')
        if lbl in monthly_data_map:
            monthly_data_map[lbl] = entry['count']
    monthly_counts = [monthly_data_map[l] for l in monthly_labels]

    # Jump types — 5 time ranges for client-side switching
    def _type_data(qs, since=None):
        if since:
            qs = qs.filter(date__gte=since)
        rows = qs.values('jump_type__name').annotate(count=Count('id')).order_by('-count')
        labels, data, other = [], [], 0
        for r in rows:
            name = r['jump_type__name'] or 'Unspecified'
            if r['count'] < 3:
                other += r['count']
            else:
                labels.append(name)
                data.append(r['count'])
        if other:
            labels.append('Other')
            data.append(other)
        return {'labels': labels, 'data': data}

    types_json = json.dumps({
        'all':  _type_data(logbook_qs),
        '12mo': _type_data(logbook_qs, today - timedelta(days=365)),
        '6mo':  _type_data(logbook_qs, today - timedelta(days=182)),
        '90d':  _type_data(logbook_qs, today - timedelta(days=90)),
        '30d':  _type_data(logbook_qs, today - timedelta(days=30)),
    })

    # Summary stats
    yearly_stats = list(
        logbook_qs
        .annotate(year=ExtractYear('date'))
        .values('year').annotate(count=Count('id')).order_by('year')
    )
    best_entry = (
        logbook_qs
        .annotate(month=TruncMonth('date'))
        .values('month').annotate(count=Count('id')).order_by('-count').first()
    )
    best_month_label = best_entry['month'].strftime('%b %Y') if best_entry else None
    best_month_count = best_entry['count'] if best_entry else None
    last_30_count = logbook_qs.filter(date__gte=today - timedelta(days=30)).count()
    this_year_total = next((r['count'] for r in yearly_stats if r['year'] == today.year), 0)
    months_with_data = (
        logbook_qs.annotate(month=TruncMonth('date')).values('month').distinct().count()
    )
    avg_monthly = round(logbook_total / months_with_data, 1) if months_with_data > 0 else None

    return render(request, 'users/dashboard.html', {
        'logbook_total': logbook_total,
        'last_jump': last_jump,
        'recent_jumps': recent_jumps,
        'monthly_labels': json.dumps(monthly_labels),
        'monthly_counts': json.dumps(monthly_counts),
        'types_json': types_json,
        'yearly_stats': yearly_stats,
        'best_month_label': best_month_label,
        'best_month_count': best_month_count,
        'last_30_count': last_30_count,
        'this_year_total': this_year_total,
        'avg_monthly': avg_monthly,
        'has_chart_data': logbook_total > 0,
    })


@login_required
def swooper_dashboard_view(request):
    """Swooper app landing page — GPS swoop stats, canopy breakdown, recent swoops."""
    from django.db.models import Avg, Max
    from logbook.models import Jump

    user = request.user
    flights_qs = Flight.objects.filter(pilot=user)
    swoops = flights_qs.filter(is_swoop=True, swoop_rejected=False, analysis_successful=True)
    total_swoops_gps = swoops.count()

    stats = {}
    if total_swoops_gps > 0:
        rot = swoops.filter(turn_rotation__isnull=False).aggregate(
            avg=Avg(models.Func(models.F('turn_rotation'), function='ABS')),
            best=Max(models.Func(models.F('turn_rotation'), function='ABS')),
        )
        spd = swoops.filter(max_vertical_speed_mph__isnull=False).aggregate(
            avg=Avg('max_vertical_speed_mph'),
            best=Max('max_vertical_speed_mph'),
        )
        gnd = swoops.filter(max_ground_speed_mph__isnull=False).aggregate(
            best=Max('max_ground_speed_mph'),
        )
        dist = swoops.filter(
            swoop_distance_ft__isnull=False,
            swoop_avg_altitude_agl__lte=5.0,
        ).aggregate(best=Max('swoop_distance_ft'))
        stats = {
            'avg_rotation': rot['avg'],
            'best_rotation': rot['best'],
            'avg_speed': spd['avg'],
            'best_speed': spd['best'],
            'best_ground_speed': gnd['best'],
            'best_distance': dist['best'],
        }

    # Canopy breakdown — only canopies with swoop jumps
    canopy_breakdown = []
    for canopy in user.canopies.filter(is_active=True).order_by('-is_primary', 'model'):
        swoop_count = Jump.objects.filter(user=user, canopy=canopy, swoop=True).count()
        if swoop_count == 0:
            continue
        gps_agg = swoops.filter(jump__canopy=canopy).aggregate(
            avg_rot=Avg(models.Func(models.F('turn_rotation'), function='ABS')),
            best_spd=Max('max_vertical_speed_mph'),
        )
        canopy_breakdown.append({
            'canopy': canopy,
            'swoops': swoop_count,
            'avg_rotation': round(gps_agg['avg_rot']) if gps_agg['avg_rot'] else None,
            'best_speed': round(gps_agg['best_spd'], 1) if gps_agg['best_spd'] else None,
            'wing_loading': canopy.wing_loading,
        })

    # Recent swoop jumps
    recent_swoops = (
        Jump.objects
        .filter(user=user, swoop=True)
        .select_related('dropzone', 'canopy', 'flight')
        .order_by('-date', '-created_at')[:20]
    )

    return render(request, 'users/swooper_dashboard.html', {
        'total_swoops_gps': total_swoops_gps,
        'total_flights': flights_qs.count(),
        'stats': stats,
        'canopy_breakdown': canopy_breakdown,
        'recent_swoops': recent_swoops,
    })


@login_required
def logbook_onboarding_view(request):
    """One-time logbook setup page shown to new users after registration."""
    from logbook.models import Jump

    if request.method == 'POST':
        try:
            start = int(request.POST.get('jump_number_offset', '1'))
            if start >= 1:
                profile = request.user.profile
                profile.jump_number_offset = start
                profile.save(update_fields=['jump_number_offset'])
        except (ValueError, Exception):
            pass
        return redirect('dashboard')

    # Suggest next jump number based on total_jumps entered at signup
    suggested_start = 1
    try:
        total = request.user.profile.total_jumps
        if total and total > 0:
            suggested_start = total + 1
    except Exception:
        pass

    return render(request, 'users/logbook_onboarding.html', {
        'suggested_start': suggested_start,
    })


@login_required
def profile_view(request):
    """User profile view and edit"""
    if request.method == 'POST':
        if '_jump_number_only' in request.POST:
            # Separate jump-number form in the sidebar
            try:
                new_offset = int(request.POST.get('jump_number_offset', ''))
                if new_offset >= 1:
                    profile = request.user.profile
                    old_offset = profile.jump_number_offset
                    if new_offset != old_offset:
                        profile.jump_number_offset = new_offset
                        profile.save(update_fields=['jump_number_offset'])
                        from logbook.models import Jump
                        Jump.renumber_for_user(request.user)
                    messages.success(request, 'Jump number updated.')
                else:
                    messages.error(request, 'Starting jump number must be at least 1.')
            except (ValueError, TypeError):
                messages.error(request, 'Invalid jump number.')
            return redirect('profile')

        form = UserProfileForm(request.POST, instance=request.user.profile)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully!')
            return redirect('profile')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserProfileForm(instance=request.user.profile)

    return render(request, 'users/profile.html', {'form': form})


@login_required
def add_canopy_view(request):
    """Add a new canopy"""
    if request.method == 'POST':
        form = CanopyForm(request.POST, user=request.user)
        if form.is_valid():
            canopy = form.save()
            messages.success(request, f'Canopy {canopy} added successfully!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CanopyForm(user=request.user)

    return render(request, 'users/add_canopy.html', {'form': form})


@login_required
def edit_canopy_view(request, canopy_id):
    """Edit an existing canopy"""
    canopy = get_object_or_404(Canopy, id=canopy_id, user=request.user)

    if request.method == 'POST':
        form = CanopyForm(request.POST, instance=canopy, user=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, f'Canopy {canopy} updated successfully!')
            return redirect('dashboard')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = CanopyForm(instance=canopy, user=request.user)

    return render(request, 'users/edit_canopy.html', {'form': form, 'canopy': canopy})


@login_required
def delete_canopy_view(request, canopy_id):
    """Delete a canopy"""
    canopy = get_object_or_404(Canopy, id=canopy_id, user=request.user)

    if request.method == 'POST':
        canopy_name = str(canopy)
        canopy.delete()
        messages.success(request, f'Canopy {canopy_name} deleted successfully!')
        return redirect('dashboard')

    return render(request, 'users/delete_canopy.html', {'canopy': canopy})


@login_required
def flights_view(request):
    """View user's flight history with filtering, sorting, and pagination"""
    from django.core.paginator import Paginator

    flights = Flight.objects.filter(pilot=request.user)

    # Filter options
    filter_type = request.GET.get('type', 'all')
    if filter_type == 'swoops':
        flights = flights.filter(is_swoop=True, swoop_rejected=False)
    elif filter_type == 'successful':
        flights = flights.filter(is_swoop=True, swoop_rejected=False, analysis_successful=True)
    elif filter_type == 'failed':
        flights = flights.filter(analysis_successful=False)

    # Canopy filter
    canopy_id = request.GET.get('canopy')
    if canopy_id:
        flights = flights.filter(canopy_id=canopy_id)

    # Sorting
    sort_by = request.GET.get('sort', '-created_at')
    valid_sorts = [
        'created_at', '-created_at',
        'session_id', '-session_id',
        'is_swoop', '-is_swoop',
        'turn_rotation', '-turn_rotation',
        'max_vertical_speed_mph', '-max_vertical_speed_mph',
        'max_ground_speed_mph', '-max_ground_speed_mph',
        'performance_grade', '-performance_grade'
    ]
    if sort_by in valid_sorts:
        flights = flights.order_by(sort_by)
    else:
        flights = flights.order_by('-created_at')

    # Pagination
    paginator = Paginator(flights, 20)  # 20 flights per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Get user's canopies for filter dropdown
    canopies = request.user.canopies.filter(is_active=True).order_by('manufacturer', 'model')

    context = {
        'page_obj': page_obj,
        'flights': page_obj,  # For template compatibility
        'canopies': canopies,
        'current_filter': filter_type,
        'current_canopy': canopy_id,
        'current_sort': sort_by,
    }

    return render(request, 'users/flights.html', context)


@login_required
def upload_flight_view(request):
    """Upload and process FlySight CSV files"""
    if request.method == 'POST':
        form = FlightUploadForm(request.POST, request.FILES, user=request.user)
        if form.is_valid():
            files = form.cleaned_data.get('file')
            canopy = form.cleaned_data.get('canopy')

            # Ensure files is a list
            if not isinstance(files, list):
                files = [files] if files else []

            # Handle single file upload - redirect to flight detail
            if len(files) == 1:
                return _process_single_file(request, files[0], canopy)

            # Handle multiple files - process all and redirect to summary
            else:
                return _process_multiple_files(request, files, canopy)

        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = FlightUploadForm(user=request.user)

    return render(request, 'users/upload_flight.html', {'form': form})


def _process_single_file(request, file, canopy):
    """Process a single uploaded file"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            for chunk in file.chunks():
                tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        # Process the file
        flight = process_flysight_file(tmp_file_path, pilot=request.user, canopy=canopy)

        # Clean up temporary file
        os.unlink(tmp_file_path)

        if flight.analysis_successful:
            if flight.is_swoop:
                messages.success(request, f'Flight uploaded and analyzed successfully! Swoop detected with {abs(flight.turn_rotation):.0f}° rotation.')
            else:
                messages.success(request, 'Flight uploaded successfully! No swoop detected in this flight.')
            return redirect('flight_detail', flight_id=flight.id)
        else:
            # Analysis failed - mark flight as unsuccessful and redirect to flight detail
            messages.warning(request, f'Flight uploaded but automatic swoop detection failed. Flight saved for later review.')
            return redirect('flight_detail', flight_id=flight.id)

    except Exception as e:
        # Clean up temp file if it exists
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        messages.error(request, f'Error processing flight: {str(e)}')
        return redirect('upload_flight')


def _process_multiple_files(request, files, canopy):
    """Process multiple uploaded files"""
    results = []
    successful_uploads = 0

    for file in files:
        result = {
            'filename': file.name,
            'success': False,
            'flight_id': None,
            'error': None,
            'swoop_detected': False,
            'turn_rotation': None
        }

        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                for chunk in file.chunks():
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            # Process the file
            flight = process_flysight_file(tmp_file_path, pilot=request.user, canopy=canopy)

            # Clean up temporary file
            os.unlink(tmp_file_path)

            result['success'] = True
            result['flight_id'] = flight.id
            result['analysis_successful'] = flight.analysis_successful
            result['swoop_detected'] = flight.is_swoop
            result['turn_rotation'] = flight.turn_rotation
            result['analysis_error'] = flight.analysis_error

            if flight.analysis_successful:
                successful_uploads += 1

        except Exception as e:
            # Clean up temp file if it exists
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            result['error'] = str(e)

        results.append(result)

    # Store results in session for the results page
    failed_uploads = len(files) - successful_uploads
    request.session['upload_results'] = {
        'results': results,
        'total_files': len(files),
        'successful_uploads': successful_uploads,
        'failed_uploads': failed_uploads
    }

    messages.success(request, f'Processed {len(files)} files. {successful_uploads} successful uploads.')
    return redirect('upload_results')


@login_required
def upload_results_view(request):
    """Display results of multiple file uploads"""
    upload_results = request.session.get('upload_results')

    if not upload_results:
        messages.error(request, 'No upload results found.')
        return redirect('upload_flight')

    # Clear the session data after retrieving it
    if 'upload_results' in request.session:
        del request.session['upload_results']

    return render(request, 'users/upload_results.html', {
        'upload_results': upload_results
    })


def flight_detail_view(request, flight_id):
    """View detailed flight analysis"""
    if request.user.is_authenticated:
        # For authenticated users, try to get their own flight first
        try:
            flight = Flight.objects.get(id=flight_id, pilot=request.user)
            is_owner = True
        except Flight.DoesNotExist:
            # If not their flight, check if it's a public flight they can view
            flight = get_object_or_404(Flight, id=flight_id, is_public=True)
            is_owner = False
    else:
        # For anonymous users, only allow public flights
        flight = get_object_or_404(Flight, id=flight_id, is_public=True)
        is_owner = False

    # Get chart data for visualization
    import json
    chart_data_raw = flight.get_chart_data()
    chart_data = json.dumps(chart_data_raw) if chart_data_raw else None

    # Get 3D visualization data
    viz_3d_raw = flight.get_3d_visualization_data()
    viz_3d_data = json.dumps(viz_3d_raw) if viz_3d_raw else None

    # Calculate additional metrics for display
    swoop_distance = None
    swoop_avg_altitude = None
    horizontal_speed_at_start = None
    rollout_start_altitude = None
    rollout_end_altitude = None

    if flight.is_swoop and flight.rollout_end_idx and flight.landing_idx:
        # Optimize GPS point queries by fetching only specific points we need
        try:
            # Get GPS points ordered by timestamp for direct indexing
            gps_points = list(flight.gps_points.order_by('timestamp'))

            if len(gps_points) > max(flight.rollout_end_idx, flight.landing_idx):
                rollout_end_point = gps_points[flight.rollout_end_idx]
                landing_point = gps_points[flight.landing_idx]

                # Calculate distance between rollout end and landing (touchdown)
                distance_m = rollout_end_point.location.distance(landing_point.location) * 111000  # Convert degrees to meters
                swoop_distance = distance_m * 3.28084  # Convert to feet
        except (IndexError, TypeError):
            swoop_distance = None

        # Calculate average altitude during rollout for swoop avg altitude
        if flight.rollout_start_idx and flight.rollout_end_idx:
            try:
                # Use more efficient query with timestamp bounds
                if 'gps_points' not in locals():
                    gps_points = list(flight.gps_points.order_by('timestamp'))

                if len(gps_points) > max(flight.rollout_start_idx, flight.rollout_end_idx):
                    start_time = gps_points[flight.rollout_start_idx].timestamp
                    end_time = gps_points[flight.rollout_end_idx].timestamp

                    # Use aggregate query instead of loading all points
                    avg_altitude_m = flight.gps_points.filter(
                        timestamp__gte=start_time,
                        timestamp__lte=end_time
                    ).aggregate(models.Avg('altitude_agl'))['altitude_agl__avg']

                    if avg_altitude_m:
                        swoop_avg_altitude = avg_altitude_m * 3.28084  # Convert to feet
            except (IndexError, TypeError):
                swoop_avg_altitude = None

        # Get horizontal speed at start of swoop
        if flight.flare_idx:
            try:
                if 'gps_points' not in locals():
                    gps_points = list(flight.gps_points.order_by('timestamp'))

                if len(gps_points) > flight.flare_idx:
                    flare_point = gps_points[flight.flare_idx]
                    horizontal_speed_at_start = flare_point.ground_speed * 2.23694 if flare_point.ground_speed else None  # Convert to mph
            except (IndexError, TypeError):
                pass

    # Get rollout start and end altitudes efficiently
    if flight.rollout_start_idx or flight.rollout_end_idx:
        try:
            if 'gps_points' not in locals():
                gps_points = list(flight.gps_points.order_by('timestamp'))

            if flight.rollout_start_idx and len(gps_points) > flight.rollout_start_idx:
                rollout_start_point = gps_points[flight.rollout_start_idx]
                rollout_start_altitude = rollout_start_point.altitude_agl * 3.28084  # Convert to feet

            if flight.rollout_end_idx and len(gps_points) > flight.rollout_end_idx:
                rollout_end_point = gps_points[flight.rollout_end_idx]
                rollout_end_altitude = rollout_end_point.altitude_agl * 3.28084  # Convert to feet
        except (IndexError, TypeError):
            pass

    # Get available competition gates for selection (only for owner)
    from flights.models import CompetitionGate
    available_gates = None
    if is_owner:
        available_gates = CompetitionGate.objects.filter(is_parsed=True).order_by('name')

    # Generate overhead view data if flight has swoop analysis
    overhead_data = None
    overhead_data_json = None
    if flight.is_swoop and flight.analysis_successful and flight.flare_idx is not None and flight.landing_idx is not None:
        try:
            from flights.utils.overhead_view import create_overhead_view_data
            import pandas as pd

            # Get GPS data for the flight (compressed format)
            gps_data = flight.get_gps_data()
            if gps_data and len(gps_data) > flight.flare_idx:
                # Narrow GPS data to just the swoop window (flare to landing)
                swoop_gps_data = gps_data[flight.flare_idx:flight.landing_idx + 1]

                # Get entry gate location (first point in swoop window = flare point)
                entry_point = swoop_gps_data[0]
                entry_lat = entry_point.get('lat', 0)
                entry_lon = entry_point.get('lon', 0)

                # Get entry heading (compass heading at flare)
                entry_heading = entry_point.get('heading', 0)

                # Create DataFrame for flight path (swoop window only)
                flight_df = pd.DataFrame([
                    {'lat': p.get('lat', 0), 'lon': p.get('lon', 0)}
                    for p in swoop_gps_data
                ])

                # Generate overhead view data
                gate_positions = None
                if flight.competition_gate and flight.competition_gate.gate_positions:
                    gate_positions = flight.competition_gate.gate_positions

                overhead_data = create_overhead_view_data(
                    flight_df,
                    entry_lat,
                    entry_lon,
                    entry_heading,
                    gate_positions
                )

                # Convert to JSON for template
                overhead_data_json = json.dumps(overhead_data)
        except Exception as e:
            logger.warning(f"Failed to generate overhead view data for flight {flight.id}: {e}")

    # Linked jump (may not exist for flights uploaded directly via GPS Flights)
    linked_jump = getattr(flight, 'jump', None)

    context = {
        'flight': flight,
        'linked_jump': linked_jump,
        'chart_data': chart_data,
        'viz_3d_data': viz_3d_data,
        'swoop_distance': swoop_distance,
        'swoop_avg_altitude': swoop_avg_altitude,
        'horizontal_speed_at_start': horizontal_speed_at_start,
        'rollout_start_altitude': rollout_start_altitude,
        'rollout_end_altitude': rollout_end_altitude,
        'available_gates': available_gates,
        'overhead_data': overhead_data_json,
        'is_owner': is_owner,
    }

    return render(request, 'users/flight_detail.html', context)


@login_required
def select_swoop_window_view(request, flight_id):
    """Disabled - redirects to flight detail"""
    # This functionality has been disabled
    # Redirect to flight detail view
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)
    messages.info(request, 'Manual swoop window selection has been disabled.')
    return redirect('flight_detail', flight_id=flight_id)


@login_required
@require_http_methods(["POST"])
def toggle_swoop_rejected_view(request, flight_id):
    """Toggle the swoop_rejected flag — marks a detected swoop as a normal landing."""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)
    flight.swoop_rejected = not flight.swoop_rejected
    flight.save(update_fields=['swoop_rejected'])
    action = "marked as a normal landing" if flight.swoop_rejected else "restored as a swoop"
    messages.success(request, f'Flight {action} and will {"not" if flight.swoop_rejected else "now"} appear in your swoop stats.')
    return redirect('flight_detail', flight_id=flight.id)


@login_required
@require_http_methods(["POST"])
def toggle_data_incorrect_view(request, flight_id):
    """Toggle the data_incorrect flag for a flight"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    # Get notes from the form if provided
    notes = request.POST.get('notes', '').strip()

    # Toggle the flag
    flight.data_incorrect = not flight.data_incorrect

    # Update notes if flight is being flagged as incorrect
    if flight.data_incorrect and notes:
        flight.notes = notes

    flight.save()

    action = "flagged as incorrect" if flight.data_incorrect else "unflagged"
    messages.success(request, f'Flight has been {action}.')

    return redirect('flight_detail', flight_id=flight.id)


@login_required
@require_http_methods(["POST"])
def update_flight_name_view(request, flight_id):
    """Update a flight's name via AJAX"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    import json
    try:
        data = json.loads(request.body)
        new_name = data.get('flight_name', '').strip()

        if not new_name:
            return JsonResponse({'success': False, 'error': 'Flight name cannot be empty'})

        if len(new_name) > 255:
            return JsonResponse({'success': False, 'error': 'Flight name is too long (max 255 characters)'})

        flight.flight_name = new_name
        flight.save(update_fields=['flight_name'])

        return JsonResponse({'success': True, 'flight_name': new_name})

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(["POST"])
def update_flight_gate_view(request, flight_id):
    """Update a flight's competition gate via AJAX"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    import json
    try:
        data = json.loads(request.body)
        gate_id = data.get('gate_id')

        # Import here to avoid circular imports
        from flights.models import CompetitionGate

        if gate_id:
            # Validate gate exists and is parsed
            gate = get_object_or_404(CompetitionGate, id=gate_id, is_parsed=True)
            flight.competition_gate = gate
        else:
            # Remove gate assignment
            flight.competition_gate = None
            # Clear gate metrics when removing gate
            flight.gate_crossing_idx = None
            flight.gate_speed_mps = None
            flight.gate_altitude_agl = None
            flight.passed_between_gates = None

        flight.save()

        return JsonResponse({'success': True})

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON data'})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(["POST"])
def calculate_flight_gate_metrics_view(request, flight_id):
    """Calculate gate metrics for a flight via AJAX"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    try:
        if not flight.competition_gate:
            return JsonResponse({'success': False, 'error': 'No competition gate assigned to this flight'})

        # Calculate gate metrics
        success = flight.calculate_gate_metrics()

        if success:
            return JsonResponse({'success': True})
        else:
            return JsonResponse({'success': False, 'error': 'Failed to calculate gate metrics. Check that the flight has GPS data and crosses the gates.'})

    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})


@login_required
@require_http_methods(["POST"])
def delete_flight_view(request, flight_id):
    """Delete a flight"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    flight_name = f"Flight {flight.session_id[:8]}"
    flight.delete()

    messages.success(request, f'{flight_name} has been deleted.')
    return redirect('flights')


@login_required
@require_http_methods(["POST"])
def bulk_delete_flights_view(request):
    """Delete multiple flights"""
    flight_ids = request.POST.getlist('flight_ids')

    if not flight_ids:
        messages.error(request, 'No flights selected for deletion.')
        return redirect('flights')

    # Delete selected flights belonging to the user
    deleted_count = Flight.objects.filter(
        id__in=flight_ids,
        pilot=request.user
    ).delete()[0]

    if deleted_count:
        messages.success(request, f'{deleted_count} flight(s) deleted successfully.')
    else:
        messages.warning(request, 'No flights were deleted.')

    return redirect('flights')


@login_required
def privacy_settings_view(request):
    """Privacy settings page for managing public profile and flights"""
    user = request.user

    # Get user's flights for bulk privacy management
    flights = Flight.objects.filter(pilot=user).order_by('-created_at')

    context = {
        'user': user,
        'flights': flights,
        'total_flights': flights.count(),
        'public_flights': flights.filter(is_public=True).count(),
    }

    return render(request, 'users/privacy_settings.html', context)


@login_required
@require_http_methods(["POST"])
def bulk_privacy_update_view(request):
    """Update privacy settings for multiple flights"""
    form = BulkPrivacyForm(request.POST)

    if form.is_valid():
        action = form.cleaned_data['action']
        flight_ids = form.cleaned_data['flight_ids'].split(',')

        # Filter to only user's flights
        flights = Flight.objects.filter(
            id__in=flight_ids,
            pilot=request.user
        )

        if action == 'make_public':
            updated_count = flights.update(is_public=True)
            messages.success(request, f'{updated_count} flight(s) made public.')
        elif action == 'make_private':
            updated_count = flights.update(is_public=False)
            messages.success(request, f'{updated_count} flight(s) made private.')

    return redirect('privacy_settings')


@login_required
@require_http_methods(["POST"])
def toggle_flight_privacy_view(request, flight_id):
    """Toggle privacy setting for a single flight"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

    flight.is_public = not flight.is_public
    flight.save()

    status = "public" if flight.is_public else "private"
    messages.success(request, f'Flight has been made {status}.')

    return redirect('flight_detail', flight_id=flight.id)


def user_search_view(request):
    """Search for public users"""
    form = UserSearchForm(request.GET)
    users = None

    if form.is_valid():
        query = form.cleaned_data.get('query', '').strip()
        license_level = form.cleaned_data.get('license_level')
        home_dz = form.cleaned_data.get('home_dz', '').strip()

        # Base queryset: only users with public profiles
        users = User.objects.filter(profile__public_profile=True).select_related('profile')

        # Apply search filters
        if query:
            users = users.filter(
                models.Q(username__icontains=query) |
                models.Q(first_name__icontains=query) |
                models.Q(last_name__icontains=query) |
                models.Q(profile__home_dz__icontains=query)
            )

        if license_level:
            users = users.filter(profile__license_level=license_level)

        if home_dz:
            users = users.filter(profile__home_dz__icontains=home_dz)

        users = users.order_by('username')

        # Add flight counts for each user
        if users:
            for user in users:
                user.public_flight_count = user.flights.filter(is_public=True).count()
                user.public_swoop_count = user.flights.filter(is_public=True, is_swoop=True, swoop_rejected=False).count()

    context = {
        'form': form,
        'users': users,
        'search_performed': bool(form.is_valid() and any(form.cleaned_data.values())),
    }

    return render(request, 'users/user_search.html', context)


def public_profile_view(request, username):
    """View a user's public profile"""
    user = get_object_or_404(User, username=username)

    # Check if profile is public
    if not user.profile.public_profile:
        messages.error(request, f"{user.username}'s profile is not public.")
        return redirect('user_search')

    # Get public flights
    public_flights = Flight.objects.filter(
        pilot=user,
        is_public=True
    ).order_by('-created_at')

    # Get public swoops only
    public_swoops = public_flights.filter(is_swoop=True, swoop_rejected=False, analysis_successful=True)

    # Calculate public stats
    stats = {
        'total_public_flights': public_flights.count(),
        'total_public_swoops': public_swoops.count(),
    }

    # Add swoop performance stats if there are public swoops
    if public_swoops.exists():
        from django.db.models import Avg, Max, Min, Count

        # Get rotation stats
        rotation_stats = public_swoops.filter(turn_rotation__isnull=False).aggregate(
            avg_rotation=Avg(models.Func(models.F('turn_rotation'), function='ABS')),
            max_rotation=Max(models.Func(models.F('turn_rotation'), function='ABS')),
            min_rotation=Min(models.Func(models.F('turn_rotation'), function='ABS')),
        )

        # Get speed stats
        speed_stats = public_swoops.filter(max_vertical_speed_mph__isnull=False).aggregate(
            avg_speed=Avg('max_vertical_speed_mph'),
            max_speed=Max('max_vertical_speed_mph'),
            min_speed=Min('max_vertical_speed_mph')
        )

        # Get ground speed stats
        ground_speed_stats = public_swoops.filter(max_ground_speed_mph__isnull=False).aggregate(
            max_ground_speed=Max('max_ground_speed_mph')
        )

        # Update stats with aggregated values
        if rotation_stats['max_rotation']:
            stats.update(rotation_stats)

        if speed_stats['max_speed']:
            stats.update(speed_stats)

        if ground_speed_stats['max_ground_speed']:
            stats['max_ground_speed'] = ground_speed_stats['max_ground_speed']

        # Get distance stats
        distance_stats = public_swoops.filter(
            swoop_distance_ft__isnull=False,
            swoop_avg_altitude_agl__lte=5.0
        ).aggregate(
            max_distance=Max('swoop_distance_ft')
        )

        if distance_stats['max_distance']:
            stats['max_swoop_distance'] = distance_stats['max_distance']

    # Recent public flights
    recent_public_flights = public_flights[:10]

    context = {
        'profile_user': user,
        'stats': stats,
        'recent_flights': recent_public_flights,
        'public_swoops': public_swoops[:20],  # Limit to recent swoops
    }

    return render(request, 'users/public_profile.html', context)


def public_swoops_view(request):
    """View all public swoops across the platform"""
    # Get all public swoops from users with public profiles
    public_swoops = Flight.objects.filter(
        is_public=True,
        is_swoop=True,
        analysis_successful=True,
        pilot__profile__public_profile=True
    ).select_related('pilot', 'pilot__profile', 'canopy').order_by('-created_at')

    # Pagination
    from django.core.paginator import Paginator
    paginator = Paginator(public_swoops, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'page_obj': page_obj,
        'total_count': public_swoops.count(),
    }

    return render(request, 'users/public_swoops.html', context)


@login_required
@require_http_methods(["POST"])
def upload_gate_file(request):
    """Handle gate file uploads and parse them into CompetitionGate objects"""
    try:
        from flights.models import CompetitionGate
        from flights.utils.gate_parser import GateFileParser

        gate_name = request.POST.get('gate_name', '').strip()
        gate_type = request.POST.get('gate_type', '').strip()
        gate_file = request.FILES.get('gate_file')

        # Validate inputs
        if not gate_name or not gate_type or not gate_file:
            return JsonResponse({
                'success': False,
                'error': 'Missing required fields'
            }, status=400)

        # Validate gate type
        valid_types = ['speed', 'standard', 'distance']
        if gate_type not in valid_types:
            return JsonResponse({
                'success': False,
                'error': f'Invalid gate type. Must be one of: {", ".join(valid_types)}'
            }, status=400)

        # Check file size (max 5MB)
        if gate_file.size > 5 * 1024 * 1024:
            return JsonResponse({
                'success': False,
                'error': 'File size exceeds 5MB limit'
            }, status=400)

        # Check file extension
        allowed_extensions = ['.csv', '.gsw']
        file_ext = os.path.splitext(gate_file.name)[1].lower()
        if file_ext not in allowed_extensions:
            return JsonResponse({
                'success': False,
                'error': f'Invalid file type. Must be CSV or GSW'
            }, status=400)

        # Save file temporarily and parse
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            for chunk in gate_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name

        try:
            # Parse the gate file
            parser = GateFileParser(tmp_path, gate_type=gate_type)
            gate_positions = parser.extract_gate_positions()

            if not gate_positions:
                return JsonResponse({
                    'success': False,
                    'error': 'No gate positions found in file'
                }, status=400)

            # Create or update CompetitionGate
            gate, created = CompetitionGate.objects.get_or_create(
                name=gate_name,
                defaults={
                    'gate_type': gate_type,
                    'gate_positions': gate_positions,
                    'is_parsed': True
                }
            )

            if not created:
                # Update existing gate
                gate.gate_type = gate_type
                gate.gate_positions = gate_positions
                gate.is_parsed = True
                gate.save()

            return JsonResponse({
                'success': True,
                'message': f'Gate course "{gate_name}" uploaded successfully',
                'gate_id': gate.id,
                'gate_count': len(gate_positions)
            })
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Error uploading gate file: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)