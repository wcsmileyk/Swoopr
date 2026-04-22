from django.contrib.auth.models import AbstractUser


class User(AbstractUser):
    """
    Custom User model that maps to the existing auth_user table. Setting
    AUTH_USER_MODEL to this now lets us add fields and extend the user in
    future phases without a disruptive data migration.

    No new table is created — db_table points at the existing auth_user table
    and the migration uses SeparateDatabaseAndState to register the model in
    Django's state without touching the database.
    """

    class Meta:
        db_table = 'auth_user'
