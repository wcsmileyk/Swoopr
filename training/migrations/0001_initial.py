import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    """
    Registers training.StudentSignoff and training.PendingInstructorRequest
    in the migration state. Both tables already exist (created by logbook/0004)
    — no database operations needed. logbook/0005 will subsequently remove
    the logbook.* state entries for these models.
    """

    initial = True

    dependencies = [
        ('logbook', '0004_jump_student_category_jump_student_jump_method_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.SeparateDatabaseAndState(
            database_operations=[],
            state_operations=[
                migrations.CreateModel(
                    name='PendingInstructorRequest',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('student_criteria', models.JSONField(default=dict, help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}')),
                        ('created_at', models.DateTimeField(auto_now_add=True)),
                        ('instructor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pending_sign_requests', to=settings.AUTH_USER_MODEL)),
                        ('jump', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pending_instructor_requests', to='logbook.jump')),
                    ],
                    options={
                        'ordering': ['-created_at'],
                        'db_table': 'logbook_pendinginstructorrequest',
                        'unique_together': {('jump', 'instructor')},
                    },
                ),
                migrations.CreateModel(
                    name='StudentSignoff',
                    fields=[
                        ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                        ('signed_at', models.DateTimeField(auto_now_add=True)),
                        ('instructor_name', models.CharField(max_length=100)),
                        ('instructor_license', models.CharField(blank=True, max_length=20)),
                        ('instructor_rating', models.CharField(choices=[('AFF-I', 'AFF Instructor'), ('TI', 'Tandem Instructor'), ('IAD-I', 'IAD/Static Line Instructor'), ('Coach', 'Coach')], max_length=10)),
                        ('criteria_passed', models.JSONField(default=dict, help_text='{"ground": [...], "freefall": [...], "canopy": [...], "method": [...]}')),
                        ('criteria_modified', models.BooleanField(default=False, help_text="True if the instructor changed the student's pre-filled criteria")),
                        ('student_notified', models.BooleanField(default=False)),
                        ('jump', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='student_signoff', to='logbook.jump')),
                        ('signed_by', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='student_signoffs_given', to=settings.AUTH_USER_MODEL)),
                    ],
                    options={
                        'db_table': 'logbook_studentsignoff',
                    },
                ),
            ],
        ),
    ]
