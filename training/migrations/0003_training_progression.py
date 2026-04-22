import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0002_instructor_ratings'),
        ('organizations', '0002_dropzone_membership'),
        ('logbook', '0005_move_to_domain_apps'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='USPACategory',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('program_type', models.CharField(
                    choices=[
                        ('AFF', 'AFF'),
                        ('IAD_SL', 'IAD / Static Line'),
                        ('Coach', 'Coach'),
                        ('Tandem', 'Tandem'),
                    ],
                    max_length=10,
                )),
                ('code', models.CharField(max_length=10, help_text="'fjc', 'a'–'h' for AFF")),
                ('name', models.CharField(max_length=50)),
                ('order', models.PositiveIntegerField()),
                ('criteria', models.JSONField(default=dict)),
                ('recommended_jumps', models.JSONField(default=dict)),
            ],
            options={
                'verbose_name': 'USPA Category',
                'verbose_name_plural': 'USPA Categories',
                'ordering': ['program_type', 'order'],
                'unique_together': {('program_type', 'code')},
            },
        ),
        migrations.CreateModel(
            name='ProgramJump',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('method_group', models.CharField(
                    choices=[
                        ('all', 'All Methods'),
                        ('AFF_Tandem', 'AFF / Tandem'),
                        ('AFF', 'AFF Only'),
                        ('Tandem', 'Tandem Only'),
                        ('IAD_SL', 'IAD / Static Line'),
                        ('Coach', 'Coach'),
                    ],
                    default='all',
                    max_length=20,
                )),
                ('jump_number', models.PositiveIntegerField()),
                ('name', models.CharField(blank=True, max_length=100)),
                ('allowed_methods', models.JSONField(default=list)),
                ('dive_flow', models.JSONField(default=list)),
                ('assigned_criteria', models.JSONField(default=dict)),
                ('is_active', models.BooleanField(default=True)),
                ('dropzone', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='program_jumps',
                    to='organizations.dropzone',
                )),
                ('category', models.ForeignKey(
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='program_jumps',
                    to='training.uspacategory',
                )),
            ],
            options={
                'ordering': ['category__order', 'method_group', 'jump_number'],
                'unique_together': {('dropzone', 'category', 'method_group', 'jump_number')},
            },
        ),
        migrations.CreateModel(
            name='StudentEnrollment',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('program_type', models.CharField(
                    choices=[
                        ('AFF', 'AFF'),
                        ('IAD_SL', 'IAD / Static Line'),
                        ('Coach', 'Coach'),
                        ('Tandem', 'Tandem'),
                    ],
                    max_length=10,
                )),
                ('status', models.CharField(
                    choices=[
                        ('active', 'Active'),
                        ('paused', 'Paused'),
                        ('graduated', 'Graduated'),
                        ('withdrawn', 'Withdrawn'),
                    ],
                    default='active',
                    max_length=20,
                )),
                ('enrolled_date', models.DateField(auto_now_add=True)),
                ('notes', models.TextField(blank=True)),
                ('student', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='enrollments',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('dropzone', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='enrollments',
                    to='organizations.dropzone',
                )),
                ('enrolled_by', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='enrollments_created',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('current_jump', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='students_at_jump',
                    to='training.programjump',
                )),
            ],
            options={
                'ordering': ['-enrolled_date'],
                'unique_together': {('student', 'dropzone', 'program_type')},
            },
        ),
        migrations.CreateModel(
            name='StudentJump',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('jump_method', models.CharField(
                    choices=[
                        ('AFF_2I', 'AFF — 2 Instructor'),
                        ('AFF_solo', 'AFF — Solo JM'),
                        ('Tandem', 'Tandem'),
                        ('IAD', 'IAD'),
                        ('SL', 'Static Line'),
                        ('Coach', 'Coach'),
                    ],
                    max_length=10,
                )),
                ('jump_date', models.DateField()),
                ('outcome', models.CharField(
                    blank=True, null=True,
                    choices=[
                        ('pass', 'Pass'),
                        ('repeat', 'Repeat'),
                        ('partial', 'Partial Pass'),
                        ('incomplete', 'Incomplete'),
                    ],
                    max_length=20,
                )),
                ('criteria_passed', models.JSONField(default=dict)),
                ('notes', models.TextField(blank=True)),
                ('signed_off_at', models.DateTimeField(blank=True, null=True)),
                ('enrollment', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='student_jumps',
                    to='training.studentenrollment',
                )),
                ('jump', models.OneToOneField(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='student_jump',
                    to='logbook.jump',
                )),
                ('program_jump', models.ForeignKey(
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='student_jumps',
                    to='training.programjump',
                )),
                ('instructor', models.ForeignKey(
                    on_delete=django.db.models.deletion.PROTECT,
                    related_name='instructed_student_jumps',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('signed_off_by', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='student_jumps_signed',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['-jump_date'],
            },
        ),
    ]
