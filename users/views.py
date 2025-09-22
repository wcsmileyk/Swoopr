from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.contrib.auth.models import User
from django.db import transaction, models
from django.core.files.storage import default_storage
import os
import tempfile
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
                        return redirect('dashboard')

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
    """User dashboard showing flight statistics and recent activity"""
    user = request.user

    # Get user's flights and statistics
    flights = Flight.objects.filter(pilot=user).order_by('-created_at')
    total_flights = flights.count()
    swoops = flights.filter(is_swoop=True)
    total_swoops = swoops.count()

    # Calculate statistics
    stats = {
        'total_flights': total_flights,
        'total_swoops': total_swoops,
        'success_rate': (swoops.filter(analysis_successful=True).count() / total_swoops * 100) if total_swoops > 0 else 0,
    }

    # Add swoop performance stats using efficient database aggregation
    if total_swoops > 0:
        successful_swoops = swoops.filter(analysis_successful=True)
        if successful_swoops.exists():
            # Use database aggregation instead of Python loops
            from django.db.models import Avg, Max, Min, Count

            # Get rotation stats using database aggregation
            rotation_stats = successful_swoops.filter(turn_rotation__isnull=False).aggregate(
                avg_rotation=Avg(models.Func(models.F('turn_rotation'), function='ABS')),
                max_rotation=Max(models.Func(models.F('turn_rotation'), function='ABS')),
                min_rotation=Min(models.Func(models.F('turn_rotation'), function='ABS')),
                rotation_count=Count('turn_rotation')
            )

            # Get speed stats using database aggregation
            speed_stats = successful_swoops.filter(max_vertical_speed_mph__isnull=False).aggregate(
                avg_speed=Avg('max_vertical_speed_mph'),
                max_speed=Max('max_vertical_speed_mph'),
                min_speed=Min('max_vertical_speed_mph')
            )

            # Get ground speed stats
            ground_speed_stats = successful_swoops.filter(max_ground_speed_mph__isnull=False).aggregate(
                max_ground_speed=Max('max_ground_speed_mph')
            )

            # Update stats with aggregated values
            if rotation_stats['rotation_count'] > 0:
                stats.update({
                    'avg_rotation': rotation_stats['avg_rotation'],
                    'max_rotation': rotation_stats['max_rotation'],
                    'min_rotation': rotation_stats['min_rotation'],
                })

            if speed_stats['max_speed']:
                stats.update({
                    'avg_speed': speed_stats['avg_speed'],
                    'max_speed': speed_stats['max_speed'],
                    'min_speed': speed_stats['min_speed'],
                })

            if ground_speed_stats['max_ground_speed']:
                stats['max_ground_speed'] = ground_speed_stats['max_ground_speed']

            # Get swoop distance stats using the stored field (only for swoops with avg altitude ≤ 5m AGL)
            distance_stats = successful_swoops.filter(
                swoop_distance_ft__isnull=False,
                swoop_avg_altitude_agl__lte=5.0
            ).aggregate(
                max_distance=Max('swoop_distance_ft')
            )

            if distance_stats['max_distance']:
                stats['max_swoop_distance'] = distance_stats['max_distance']

            # Find flights that achieved personal bests (efficient queries)
            personal_bests = {}

            # Max rotation flight
            if rotation_stats['rotation_count'] > 0:
                max_rotation_flight = successful_swoops.filter(turn_rotation__isnull=False).annotate(
                    abs_rotation=models.Func(models.F('turn_rotation'), function='ABS')
                ).order_by('-abs_rotation').first()
                if max_rotation_flight:
                    personal_bests['max_rotation_flight_id'] = max_rotation_flight.id

            # Max vertical speed flight
            if speed_stats['max_speed']:
                max_speed_flight = successful_swoops.filter(
                    max_vertical_speed_mph=speed_stats['max_speed']
                ).first()
                if max_speed_flight:
                    personal_bests['max_speed_flight_id'] = max_speed_flight.id

            # Max ground speed flight
            if ground_speed_stats['max_ground_speed']:
                max_ground_speed_flight = successful_swoops.filter(
                    max_ground_speed_mph=ground_speed_stats['max_ground_speed']
                ).first()
                if max_ground_speed_flight:
                    personal_bests['max_ground_speed_flight_id'] = max_ground_speed_flight.id

            # Max swoop distance flight (only for swoops with avg altitude ≤ 5m AGL)
            if distance_stats['max_distance']:
                max_distance_flight = successful_swoops.filter(
                    swoop_distance_ft=distance_stats['max_distance'],
                    swoop_avg_altitude_agl__lte=5.0
                ).first()
                if max_distance_flight:
                    personal_bests['max_distance_flight_id'] = max_distance_flight.id

            stats['personal_bests'] = personal_bests

    # Recent flights
    recent_flights = flights[:5]

    # User's canopies
    canopies = user.canopies.filter(is_active=True).order_by('-is_primary', 'manufacturer', 'model')

    context = {
        'user': user,
        'stats': stats,
        'recent_flights': recent_flights,
        'canopies': canopies,
        'has_profile': hasattr(user, 'profile'),
    }

    return render(request, 'users/dashboard.html', context)


@login_required
def profile_view(request):
    """User profile view and edit"""
    if request.method == 'POST':
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
        flights = flights.filter(is_swoop=True)
    elif filter_type == 'successful':
        flights = flights.filter(is_swoop=True, analysis_successful=True)
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
        else:
            messages.warning(request, f'Flight uploaded but analysis failed: {flight.analysis_error}')

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


@login_required
def flight_detail_view(request, flight_id):
    """View detailed flight analysis"""
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)

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

    context = {
        'flight': flight,
        'chart_data': chart_data,
        'viz_3d_data': viz_3d_data,
        'swoop_distance': swoop_distance,
        'swoop_avg_altitude': swoop_avg_altitude,
        'horizontal_speed_at_start': horizontal_speed_at_start,
        'rollout_start_altitude': rollout_start_altitude,
        'rollout_end_altitude': rollout_end_altitude,
    }

    return render(request, 'users/flight_detail.html', context)


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
                user.public_swoop_count = user.flights.filter(is_public=True, is_swoop=True).count()

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
    public_swoops = public_flights.filter(is_swoop=True, analysis_successful=True)

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