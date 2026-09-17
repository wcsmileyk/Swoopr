"""
Service layer bridging CourseImport previews to committed CourseSet/Course/
CourseRevision rows. Kept out of views.py per the design doc's suggested
module layout, and out of models.py so models stay free of import/parsing
concerns.
"""

import dataclasses
import hashlib

from django.db import transaction

from courses.analysis.crossings import analyze_course_crossings
from courses.analysis.disciplines import compute_discipline_metrics, entry_gate_id_for
from courses.imports.kml import KmlImportError, build_preview, parse_kml_bytes
from courses.models import Course, CourseAuditEvent, CourseImport, CourseRevision, CourseSet, FlightCourseAnalysis

ANALYSIS_ENGINE_VERSION = 'course-analysis-v1'
DEFAULT_PRE_FLARE_BUFFER_SAMPLES = 120  # ~30s at 4Hz; generous enough to catch a pre-flare G1 drag entry


class CommitError(ValueError):
    """Raised when a CourseImport cannot be committed as requested."""


def run_kml_preview(course_import: CourseImport, carve_direction: str, disciplines=None,
                     entry_heading_deg_true=None, course_width_m=None) -> None:
    """Parse course_import.source_file and populate extracted/validation_report.
    Leaves the import in 'preview_ready' or 'failed' status. Idempotent --
    safe to call again (e.g. after the user changes carve_direction)."""
    course_import.source_file.open('rb')
    try:
        data = course_import.source_file.read()
    finally:
        course_import.source_file.close()

    course_import.file_hash = hashlib.sha256(data).hexdigest()

    try:
        parsed = parse_kml_bytes(data)
        # A file with only a G1 marker (no other gates) is necessarily a
        # seed, not explicit per-gate geometry.
        course_import.input_mode = 'g1_seed' if len(parsed.gates) <= 1 else 'explicit_geometry'
        preview = build_preview(
            parsed, carve_direction=carve_direction, disciplines=disciplines,
            entry_heading_deg_true=entry_heading_deg_true, course_width_m=course_width_m,
        )
    except KmlImportError as exc:
        course_import.status = 'failed'
        course_import.validation_report = {'errors': [str(exc)]}
        course_import.extracted = None
        course_import.save(update_fields=['file_hash', 'input_mode', 'status', 'validation_report', 'extracted'])
        return

    course_import.status = 'preview_ready'
    course_import.validation_report = {'warnings': preview['warnings']}
    course_import.extracted = preview
    course_import.save(update_fields=['file_hash', 'input_mode', 'status', 'validation_report', 'extracted'])


@transaction.atomic
def commit_kml_import(course_import: CourseImport, *, name: str, visibility: str = 'private',
                       site_name: str = '', actor=None) -> CourseSet:
    """Create a CourseSet with one Course+CourseRevision per previewed
    discipline. Fails clearly rather than partially committing."""
    if course_import.status != 'preview_ready':
        raise CommitError(f'Import is {course_import.status}, not preview_ready')
    if course_import.committed_course_set_id:
        raise CommitError('Import was already committed')
    if not course_import.extracted:
        raise CommitError('Import has no preview data to commit')

    preview = course_import.extracted
    entry_setup = preview['entry_setup']

    course_set = CourseSet.objects.create(
        name=name,
        owner=course_import.owner,
        site_name=site_name,
        visibility=visibility,
    )

    for discipline, geometry in preview['final_geometry'].items():
        course = Course.objects.create(course_set=course_set, discipline=discipline)
        revision = CourseRevision.objects.create(
            course=course,
            revision_number=1,
            geometry=geometry,
            algorithm_version=geometry.get('algorithm_version', ''),
            g1_lon=entry_setup['g1_lon'],
            g1_lat=entry_setup['g1_lat'],
            entry_heading_deg_true=entry_setup['entry_heading_deg_true'],
            course_width_m=entry_setup['course_width_m'],
            carve_direction=entry_setup['carve_direction'],
            source='kml_explicit' if discipline == preview.get('detected_discipline') else 'kml_seed',
            source_import=course_import,
            created_by=course_import.owner,
        )
        course.set_current_revision(revision)

    course_import.status = 'committed'
    course_import.committed_course_set = course_set
    course_import.save(update_fields=['status', 'committed_course_set'])

    CourseAuditEvent.objects.create(
        actor=actor or course_import.owner,
        action='import_committed',
        course_set=course_set,
        detail={'import_id': str(course_import.id), 'disciplines': list(preview['final_geometry'].keys())},
    )

    return course_set


def set_visibility(course_set: CourseSet, visibility: str, actor) -> None:
    if visibility not in ('private', 'public'):
        raise ValueError(f'Invalid visibility: {visibility!r}')
    old = course_set.visibility
    course_set.visibility = visibility
    course_set.save(update_fields=['visibility'])
    CourseAuditEvent.objects.create(
        actor=actor, action='visibility_changed', course_set=course_set,
        detail={'from': old, 'to': visibility},
    )


def archive_course_set(course_set: CourseSet, actor) -> None:
    course_set.archived = True
    course_set.save(update_fields=['archived'])
    CourseAuditEvent.objects.create(actor=actor, action='archived', course_set=course_set, detail={})


class ApplyCourseError(ValueError):
    """Raised when a course can't be applied to a flight as requested."""


def _json_safe(value):
    """Recursively convert dataclass/datetime values into something a
    JSONField can store. crossing_time may be a Unix epoch float (real
    flights) or a datetime (tests/other callers)."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def _default_analysis_window(flight):
    """Restrict to the terminal canopy approach, not the whole flight
    (design doc section 9) -- a generous buffer before the detected flare
    point through landing, so a pre-flare G1 drag entry is still included.
    Callers can override with explicit window indices when this default is
    wrong (e.g. automatic swoop classification failed)."""
    points = flight.get_gps_data()
    if not points:
        raise ApplyCourseError('Flight has no GPS data')

    if flight.flare_idx is not None:
        start = max(0, flight.flare_idx - DEFAULT_PRE_FLARE_BUFFER_SAMPLES)
    else:
        start = 0
    end = (flight.landing_idx + 1) if flight.landing_idx is not None else len(points)
    end = max(end, start + 2)
    return points, start, min(end, len(points))


@transaction.atomic
def apply_course_to_flight(flight, course, *, actor, window_start_idx=None, window_end_idx=None) -> FlightCourseAnalysis:
    """Evaluate flight's GPS track against course's current revision,
    store the result, and mark it primary for this flight+course pair.
    Never touches the flight's own legacy swoop/flare/turn analysis."""
    revision = course.current_revision
    if revision is None:
        raise ApplyCourseError(f'{course} has no revision to analyze against')

    all_points, default_start, default_end = _default_analysis_window(flight)
    start = window_start_idx if window_start_idx is not None else default_start
    end = window_end_idx if window_end_idx is not None else default_end
    window_points = all_points[start:end]

    if len(window_points) < 2:
        raise ApplyCourseError('Analysis window has fewer than 2 GPS points')

    gate_results = analyze_course_crossings(window_points, revision.geometry.get('gates', []))

    entry_gate_id = entry_gate_id_for(course.discipline)
    entry_result = gate_results.get(entry_gate_id)
    entry_status = 'crossed' if entry_result and entry_result.status == 'crossed' else 'entry_not_detected'

    metrics = (
        compute_discipline_metrics(course.discipline, gate_results, window_points, revision)
        if entry_status == 'crossed' else {}
    )

    warnings = []
    for gr in gate_results.values():
        warnings.extend(gr.warnings)
    if entry_status == 'entry_not_detected':
        warnings.append(
            f'No valid {entry_gate_id} crossing detected in the analysis window; '
            f'no course time/distance result'
        )

    v_accs = [p['v_acc'] for p in window_points if p.get('v_acc') is not None]
    avg_vertical_accuracy_m = round(sum(v_accs) / len(v_accs), 3) if v_accs else None

    FlightCourseAnalysis.objects.filter(flight=flight, course=course, is_primary=True).update(is_primary=False)

    return FlightCourseAnalysis.objects.create(
        flight=flight, course=course, course_revision=revision,
        engine_version=ANALYSIS_ENGINE_VERSION, is_primary=True,
        window_start_idx=start, window_end_idx=end,
        avg_vertical_accuracy_m=avg_vertical_accuracy_m,
        entry_status=entry_status,
        gate_results=_json_safe({gid: gr for gid, gr in gate_results.items()}),
        metrics=_json_safe(metrics), warnings=warnings, created_by=actor,
    )
