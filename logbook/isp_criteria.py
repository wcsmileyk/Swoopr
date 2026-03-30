"""
Helpers for loading and querying USPA ISP criteria from docs/uspa_isp.json.
"""
import json
from pathlib import Path

from django.conf import settings

_criteria_cache = None

# Maps our internal method choice values to JSON keys
_METHOD_KEY_MAP = {
    'AFF': 'AFF',
    'Tandem': 'Tandem',
    'IAD_SL': 'IAD/SL',
    'Coach': None,  # No method-specific criteria in the JSON
}

# Which profile flags authorize signing which methods
# Returns the rating label to snapshot on the signoff record
RATING_FOR_METHOD = {
    'AFF': [('affi', 'AFF-I')],
    'Tandem': [('ti', 'TI')],
    'IAD_SL': [('iad_sl', 'IAD-I')],
    # Coach jumps: coach flag first, then any instructor rating
    'Coach': [('coach', 'Coach'), ('affi', 'AFF-I'), ('ti', 'TI'), ('iad_sl', 'IAD-I')],
}

# Cat F criteria that must be signed off before Coach method is available
_COACH_UNLOCK_CRITERIA = {'Clear and pull at 5,500 feet', 'Clear and pull at 3,500 feet'}


def _load():
    global _criteria_cache
    if _criteria_cache is None:
        path = settings.BASE_DIR / 'docs' / 'uspa_isp.json'
        with open(path, encoding='utf-8') as f:
            _criteria_cache = json.load(f)
    return _criteria_cache


def get_criteria_for_jump(category: str, method: str) -> dict:
    """
    Returns the criteria the instructor should see for a given jump.
    {
      "ground": [...],
      "freefall": [...],
      "canopy": [...],
      "method": [...]   # method-specific criteria (may be empty)
    }
    """
    data = _load()
    cat = data.get(category.lower(), {})
    method_key = _METHOD_KEY_MAP.get(method)
    return {
        'ground': cat.get('ground', []),
        'freefall': cat.get('freefall', []),
        'canopy': cat.get('canopy', []),
        'method': cat.get(method_key, []) if method_key else [],
    }


def get_all_general_criteria(category: str) -> dict:
    """Returns just ground/freefall/canopy (no method-specific) for progress tracking."""
    data = _load()
    cat = data.get(category.lower(), {})
    return {
        'ground': cat.get('ground', []),
        'freefall': cat.get('freefall', []),
        'canopy': cat.get('canopy', []),
    }


def get_signing_rating(profile, method: str) -> str | None:
    """
    Returns the rating label to store on the signoff, or None if the
    instructor is not authorized to sign this jump method.
    """
    for attr, label in RATING_FOR_METHOD.get(method, []):
        if getattr(profile, attr, False):
            return label
    return None


def get_passed_criteria(user, category: str) -> set:
    """
    Returns the set of all criteria strings the user has had signed off
    across all jumps in a given category (general sections only).
    """
    from logbook.models import StudentSignoff
    passed = set()
    signoffs = StudentSignoff.objects.filter(
        jump__user=user,
        jump__student_category=category.lower(),
    ).values_list('criteria_passed', flat=True)
    for cp in signoffs:
        for section in ('ground', 'freefall', 'canopy'):
            passed.update(cp.get(section, []))
    return passed


def get_category_progress(user) -> dict:
    """
    Returns a dict of {category: {"total": N, "passed": M, "complete": bool}}
    for all A–H categories.
    """
    categories = list('abcdefgh')
    result = {}
    for cat in categories:
        all_criteria = get_all_general_criteria(cat)
        total = sum(len(v) for v in all_criteria.values())
        passed = get_passed_criteria(user, cat)
        passed_count = sum(
            1 for section_list in all_criteria.values() for c in section_list if c in passed
        )
        result[cat] = {
            'total': total,
            'passed': passed_count,
            'complete': total > 0 and passed_count >= total,
            'criteria': all_criteria,
            'passed_set': passed,
        }
    return result


def can_use_coach_method(user) -> bool:
    """Returns True if the Cat F clear & pull criteria have been signed off."""
    passed = get_passed_criteria(user, 'f')
    return _COACH_UNLOCK_CRITERIA.issubset(passed)
