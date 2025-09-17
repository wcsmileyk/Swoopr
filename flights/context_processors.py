"""
Context processors for flight-related data.
"""

def user_units(request):
    """
    Add user's preferred units to template context.
    Defaults to imperial if user is not authenticated or has no preference.
    """
    if request.user.is_authenticated and hasattr(request.user, 'profile'):
        units = request.user.profile.units
    else:
        units = 'imperial'

    return {
        'user_units': units,
    }