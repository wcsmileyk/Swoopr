# Re-export from the canonical location so any import paths that weren't
# updated yet continue to work. Remove this file once all callers import
# directly from training.isp_criteria.
from training.isp_criteria import (  # noqa: F401
    RATING_FOR_METHOD,
    _COACH_UNLOCK_CRITERIA,
    _METHOD_KEY_MAP,
    can_use_coach_method,
    get_all_general_criteria,
    get_category_progress,
    get_criteria_for_jump,
    get_passed_criteria,
    get_signing_rating,
)
