import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('dz', '0001_phase2_operations'),
        ('aircraft', '0002_aircraft_dz_fields'),
        ('training', '0003_training_progression'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='JumpRunConditions',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('exit_altitude', models.IntegerField(help_text='Exit altitude in feet.')),
                ('spot_notes', models.TextField(blank=True, help_text='Spot description, heading, landmarks, etc.')),
                ('set_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='jump_run_log', to='organizations.dropzone')),
                ('set_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='jump_run_entries', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-set_at'],
            },
        ),
        migrations.CreateModel(
            name='Load',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('load_number', models.PositiveIntegerField(help_text='Sequential load number for this DZ on this date.')),
                ('status', models.CharField(choices=[('building', 'Building'), ('on_call', 'On Call'), ('in_air', 'In Air'), ('landed', 'Landed')], default='building', max_length=10)),
                ('call_time', models.DateTimeField(blank=True, null=True, help_text='When this load was called (moved to on_call).')),
                ('took_off_at', models.DateTimeField(blank=True, null=True)),
                ('landed_at', models.DateTimeField(blank=True, null=True)),
                ('exit_altitude', models.IntegerField(blank=True, null=True, help_text='Planned exit altitude in feet. Defaults to aircraft default if set.')),
                ('reserved_student_slots', models.IntegerField(default=0, help_text='Number of slots held for students not yet assigned.')),
                ('notes', models.TextField(blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('aircraft', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='loads', to='aircraft.aircraft')),
                ('created_by', models.ForeignKey(null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='loads_created', to=settings.AUTH_USER_MODEL)),
                ('dropzone', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='loads', to='organizations.dropzone')),
            ],
            options={
                'ordering': ['date', 'load_number'],
                'unique_together': {('dropzone', 'date', 'load_number')},
            },
        ),
        migrations.CreateModel(
            name='LoadSlot',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('guest_name', models.CharField(blank=True, max_length=100)),
                ('role', models.CharField(choices=[('student', 'Student'), ('instructor', 'Instructor'), ('fun_jumper', 'Fun Jumper'), ('videographer', 'Videographer'), ('hop_n_pop', 'Hop & Pop')], max_length=15)),
                ('jump_type', models.CharField(choices=[('tandem', 'Tandem'), ('aff', 'AFF'), ('coach', 'Coach'), ('fun_jump', 'Fun Jump'), ('hop_n_pop', 'Hop & Pop'), ('tracking', 'Tracking'), ('wingsuit', 'Wingsuit')], max_length=15)),
                ('group_key', models.PositiveIntegerField(blank=True, null=True, help_text='Slots with the same group_key are a paired party (tandem pair, AFF group, etc.).')),
                ('exit_altitude', models.IntegerField(blank=True, null=True, help_text='Exit altitude for this slot. Overrides load default if set.')),
                ('weight_lbs', models.IntegerField(blank=True, null=True)),
                ('media_status', models.CharField(choices=[('none', 'None'), ('requested', 'Requested'), ('confirmed', 'Confirmed'), ('delivered', 'Delivered')], default='none', max_length=10)),
                ('added_at', models.DateTimeField(auto_now_add=True)),
                ('load', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='slots', to='dz.load')),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='load_slots', to=settings.AUTH_USER_MODEL)),
                ('reservation', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='load_slots', to='dz.reservation', help_text='Reservation this slot was created from, if any.')),
                ('enrollment', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='load_slots', to='training.studentenrollment', help_text='Student enrollment this slot is associated with, if any.')),
                ('added_by', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='load_slots_added', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['load', 'group_key', 'role'],
            },
        ),
        migrations.CreateModel(
            name='DailyAircraftStats',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date', models.DateField()),
                ('load_count', models.IntegerField(default=0)),
                ('avg_altitude_min', models.FloatField(blank=True, null=True, help_text='Average minutes from takeoff to exit altitude.')),
                ('avg_turn_min', models.FloatField(blank=True, null=True, help_text='Average minutes from landing to next takeoff.')),
                ('aircraft', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='daily_stats', to='aircraft.aircraft')),
            ],
            options={
                'ordering': ['-date', 'aircraft'],
                'unique_together': {('aircraft', 'date')},
            },
        ),
    ]
