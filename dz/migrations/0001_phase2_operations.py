import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('organizations', '0003_waiver'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='OperatingDay',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('day_of_week', models.IntegerField(choices=[(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'), (4, 'Friday'), (5, 'Saturday'), (6, 'Sunday')])),
                ('is_open', models.BooleanField(default=True)),
                ('first_load_time', models.TimeField(blank=True, null=True)),
                ('reservation_cutoff_hours', models.IntegerField(default=24, help_text='How many hours before the jump time reservations close.')),
                ('slot_capacities', models.JSONField(blank=True, default=dict, help_text='Max bookings per jump type (and optionally per hour).')),
                ('notes', models.TextField(blank=True)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='operating_days', to='organizations.dropzone')),
            ],
            options={
                'ordering': ['dropzone', 'day_of_week'],
                'unique_together': {('dropzone', 'day_of_week')},
            },
        ),
        migrations.CreateModel(
            name='CheckIn',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('checked_in_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('notes', models.TextField(blank=True)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='check_ins', to='organizations.dropzone')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='check_ins', to=settings.AUTH_USER_MODEL)),
                ('checked_in_by', models.ForeignKey(blank=True, help_text='Staff member who processed the check-in. Null if self-check-in.', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='check_ins_processed', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-date', '-checked_in_at'],
                'unique_together': {('dropzone', 'user', 'date')},
            },
        ),
        migrations.CreateModel(
            name='DailyRoster',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('availability_state', models.CharField(choices=[('available', 'Available'), ('limited', 'Limited'), ('unavailable', 'Unavailable'), ('on_break', 'On Break')], default='available', max_length=15)),
                ('jump_count_today', models.IntegerField(default=0)),
                ('thirty_day_avg_at_checkin', models.FloatField(blank=True, help_text='Instructor 30-day jump average captured at check-in time.', null=True)),
                ('notes', models.TextField(blank=True)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='daily_rosters', to='organizations.dropzone')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='roster_entries', to=settings.AUTH_USER_MODEL)),
                ('check_in', models.OneToOneField(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='roster_entry', to='dz.checkin')),
            ],
            options={
                'ordering': ['-date', 'user__last_name'],
                'unique_together': {('dropzone', 'user', 'date')},
            },
        ),
        migrations.CreateModel(
            name='BreakRequest',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('reason', models.TextField(blank=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('denied', 'Denied')], default='pending', max_length=10)),
                ('reviewed_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='break_requests', to='organizations.dropzone')),
                ('instructor', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='break_requests', to=settings.AUTH_USER_MODEL)),
                ('roster', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='break_requests', to='dz.dailyroster')),
                ('reviewed_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='break_requests_reviewed', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        migrations.CreateModel(
            name='Reservation',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jump_type', models.CharField(choices=[('tandem', 'Tandem'), ('aff', 'AFF'), ('coach', 'Coach Jump')], max_length=10)),
                ('date', models.DateField()),
                ('time_slot', models.TimeField(blank=True, help_text='Requested jump time slot.', null=True)),
                ('guest_name', models.CharField(blank=True, max_length=100)),
                ('guest_email', models.EmailField(blank=True)),
                ('guest_phone', models.CharField(blank=True, max_length=20)),
                ('weight_lbs', models.IntegerField(blank=True, help_text='Customer weight in lbs. Used for tandem weight limit checks.', null=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('confirmed', 'Confirmed'), ('checked_in', 'Checked In'), ('jumped', 'Jumped'), ('cancelled', 'Cancelled'), ('no_show', 'No Show')], default='pending', max_length=15)),
                ('source', models.CharField(choices=[('online', 'Online'), ('phone', 'Phone'), ('walk_in', 'Walk-in')], default='online', max_length=10)),
                ('notes', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='reservations', to='organizations.dropzone')),
                ('user', models.ForeignKey(blank=True, help_text='Platform account if the customer has one.', null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='reservations', to=settings.AUTH_USER_MODEL)),
                ('assigned_instructor', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='instructor_reservations', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['date', 'time_slot'],
            },
        ),
    ]
