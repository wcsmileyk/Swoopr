import logging

from django.apps import AppConfig

_stats_logger = logging.getLogger('users.canopy_stats')


class UsersConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'users'

    def ready(self):
        from django.contrib.auth import get_user_model
        from django.db.models.signals import post_save
        from .models import UserProfile

        # Connect profile creation FIRST — nothing below should be able to break this.
        def create_user_profile(sender, instance, created, **kwargs):
            if created:
                UserProfile.objects.get_or_create(user=instance)

        post_save.connect(create_user_profile, sender=get_user_model(), dispatch_uid='users.create_user_profile')

        # Flight stats signal — isolated so any import failure here does not
        # prevent profile creation from working.
        try:
            from flights.models import Flight
            from .canopy_stats import update_canopy_stats

            def on_flight_saved(sender, instance, **kwargs):
                if instance.canopy_id and instance.analysis_successful:
                    try:
                        update_canopy_stats(instance.canopy)
                    except Exception as exc:
                        _stats_logger.warning(
                            "Failed to update canopy stats for canopy %s: %s",
                            instance.canopy_id, exc,
                        )

            post_save.connect(on_flight_saved, sender=Flight, dispatch_uid='users.update_canopy_stats')
        except Exception:
            _stats_logger.exception("Could not connect flight stats signal — canopy stats will not auto-update")