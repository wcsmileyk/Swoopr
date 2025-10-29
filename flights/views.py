from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from .models import CompetitionGate


@login_required
def gate_map_view(request, gate_id):
    """Display competition gate and course on map"""
    gate = get_object_or_404(CompetitionGate, id=gate_id)

    if not gate.is_parsed:
        return render(request, 'flights/gate_map_error.html', {
            'gate': gate,
            'error': 'Gate file has not been parsed yet.'
        })

    return render(request, 'flights/gate_map.html', {
        'gate': gate,
    })


@login_required
def gate_course_data(request, gate_id):
    """Return course data as JSON for map rendering"""
    gate = get_object_or_404(CompetitionGate, id=gate_id)

    if not gate.is_parsed:
        return JsonResponse({'error': 'Gate not parsed'}, status=400)

    return JsonResponse({
        'gate_positions': gate.gate_positions,
        'course_config': gate.course_config,
        'center': {
            'lat': gate.center_lat,
            'lon': gate.center_lon
        }
    })
