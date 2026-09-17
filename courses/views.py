import json

from django.contrib.auth.decorators import login_required
from django.http import HttpResponseBadRequest, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_http_methods

from courses.models import Course, CourseImport, CourseSet, FlightCourseAnalysis, DISCIPLINE_CHOICES
from courses.services import (
    ApplyCourseError, CommitError, apply_course_to_flight, archive_course_set,
    commit_kml_import, run_kml_preview, set_visibility,
)
from flights.models import Flight


@login_required
def course_list_view(request):
    visible = CourseSet.objects.visible_to(request.user).prefetch_related('courses')
    my_sets = visible.filter(owner=request.user)
    public_sets = visible.exclude(owner=request.user)
    return render(request, 'courses/list.html', {
        'my_sets': my_sets,
        'public_sets': public_sets,
    })


@login_required
def course_import_upload_view(request):
    if request.method == 'POST':
        source_file = request.FILES.get('source_file')
        carve_direction = request.POST.get('carve_direction')
        disciplines = request.POST.getlist('disciplines') or None

        if not source_file:
            return render(request, 'courses/import_upload.html', {'error': 'Choose a KML file to upload'})
        if carve_direction not in ('left', 'right'):
            return render(request, 'courses/import_upload.html', {'error': 'Select a carve direction'})

        entry_heading_deg_true = None
        raw_heading = request.POST.get('entry_heading_deg_true', '').strip()
        if raw_heading:
            try:
                entry_heading_deg_true = float(raw_heading)
            except ValueError:
                return render(request, 'courses/import_upload.html', {'error': 'Entry heading must be a number'})

        course_width_m = None
        raw_width = request.POST.get('course_width_m', '').strip()
        if raw_width:
            try:
                course_width_m = float(raw_width)
            except ValueError:
                return render(request, 'courses/import_upload.html', {'error': 'Course width must be a number'})

        course_import = CourseImport.objects.create(
            owner=request.user,
            source_file=source_file,
            format='kml',
            input_mode='explicit_geometry',
        )
        run_kml_preview(
            course_import, carve_direction=carve_direction, disciplines=disciplines,
            entry_heading_deg_true=entry_heading_deg_true, course_width_m=course_width_m,
        )
        return redirect('courses:import_preview', import_id=course_import.id)

    return render(request, 'courses/import_upload.html', {'discipline_choices': DISCIPLINE_CHOICES})


@login_required
def course_import_preview_view(request, import_id):
    course_import = get_object_or_404(CourseImport, id=import_id, owner=request.user)

    preview = course_import.extracted or {}
    preview_geometry_json = json.dumps(preview.get('final_geometry') or preview.get('generated_geometry') or {})

    if request.method == 'POST':
        name = request.POST.get('name', '').strip()
        visibility = request.POST.get('visibility', 'private')
        site_name = request.POST.get('site_name', '').strip()
        if not name:
            return render(request, 'courses/import_preview.html', {
                'course_import': course_import, 'preview_geometry_json': preview_geometry_json,
                'error': 'Name is required',
            })
        try:
            course_set = commit_kml_import(
                course_import, name=name, visibility=visibility, site_name=site_name, actor=request.user,
            )
        except CommitError as exc:
            return render(request, 'courses/import_preview.html', {
                'course_import': course_import, 'preview_geometry_json': preview_geometry_json,
                'error': str(exc),
            })
        return redirect('courses:detail', course_set_id=course_set.id)

    return render(request, 'courses/import_preview.html', {
        'course_import': course_import, 'preview_geometry_json': preview_geometry_json,
    })


@login_required
def course_detail_view(request, course_set_id):
    course_set = get_object_or_404(CourseSet.objects.visible_to(request.user), id=course_set_id)
    courses = course_set.courses.select_related('current_revision').all()
    is_editable = CourseSet.objects.editable_by(request.user).filter(id=course_set.id).exists()
    return render(request, 'courses/detail.html', {
        'course_set': course_set,
        'courses': courses,
        'is_editable': is_editable,
        'geometry_json': json.dumps({
            c.discipline: c.current_revision.geometry for c in courses if c.current_revision
        }),
    })


@login_required
@require_http_methods(['POST'])
def course_visibility_view(request, course_set_id):
    course_set = get_object_or_404(CourseSet.objects.editable_by(request.user), id=course_set_id)
    visibility = request.POST.get('visibility')
    if visibility not in ('private', 'public'):
        return HttpResponseBadRequest('Invalid visibility')
    set_visibility(course_set, visibility, actor=request.user)
    return JsonResponse({'success': True, 'visibility': course_set.visibility})


@login_required
@require_http_methods(['POST'])
def course_archive_view(request, course_set_id):
    course_set = get_object_or_404(CourseSet.objects.editable_by(request.user), id=course_set_id)
    archive_course_set(course_set, actor=request.user)
    return JsonResponse({'success': True})


@login_required
def apply_course_to_flight_view(request, flight_id):
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)
    available_courses = (
        Course.objects.filter(course_set__in=CourseSet.objects.visible_to(request.user))
        .exclude(current_revision__isnull=True)
        .select_related('course_set', 'current_revision')
    )

    if request.method == 'POST':
        course = get_object_or_404(available_courses, id=request.POST.get('course_id'))
        try:
            analysis = apply_course_to_flight(flight, course, actor=request.user)
        except ApplyCourseError as exc:
            return render(request, 'courses/apply_to_flight.html', {
                'flight': flight, 'available_courses': available_courses, 'error': str(exc),
            })
        return redirect('courses:flight_analysis', flight_id=flight.id, analysis_id=analysis.id)

    return render(request, 'courses/apply_to_flight.html', {
        'flight': flight, 'available_courses': available_courses,
    })


@login_required
def flight_analysis_view(request, flight_id, analysis_id):
    flight = get_object_or_404(Flight, id=flight_id, pilot=request.user)
    analysis = get_object_or_404(
        FlightCourseAnalysis.objects.select_related('course', 'course_revision', 'course__course_set'),
        id=analysis_id, flight=flight,
    )
    return render(request, 'courses/flight_analysis.html', {
        'flight': flight,
        'analysis': analysis,
        'geometry_json': json.dumps(analysis.course_revision.geometry),
        'track_json': json.dumps(_track_for_map(flight, analysis)),
    })


def _track_for_map(flight, analysis):
    points = flight.get_gps_data() or []
    window = points[analysis.window_start_idx:analysis.window_end_idx]
    return [{'lat': p['lat'], 'lon': p['lon']} for p in window if 'lat' in p and 'lon' in p]
