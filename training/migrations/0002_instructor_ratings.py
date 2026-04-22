import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('training', '0001_initial'),
        ('organizations', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='InstructorRating',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating_type', models.CharField(
                    choices=[
                        ('AFF-I', 'AFF Instructor'),
                        ('TI', 'Tandem Instructor'),
                        ('Coach', 'Coach'),
                        ('IAD-I', 'IAD/Static Line Instructor'),
                    ],
                    max_length=10,
                )),
                ('earned_date', models.DateField(blank=True, null=True)),
                ('notes', models.TextField(blank=True)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='instructor_ratings',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['user', 'rating_type'],
                'unique_together': {('user', 'rating_type')},
            },
        ),
        migrations.CreateModel(
            name='InstructorPersonalLimits',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('max_student_exit_weight', models.FloatField(
                    blank=True, null=True,
                    help_text='Maximum student exit weight in lbs',
                )),
                ('max_wind_speed', models.FloatField(
                    blank=True, null=True,
                    help_text='Personal wind limit in knots',
                )),
                ('notes', models.TextField(blank=True, help_text='Any other personal operating limits')),
                ('instructor_rating', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='personal_limits',
                    to='training.instructorrating',
                )),
            ],
            options={
                'verbose_name_plural': 'Instructor personal limits',
            },
        ),
        migrations.CreateModel(
            name='RatingQualification',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('rating_type', models.CharField(
                    choices=[
                        ('AFF-I', 'AFF Instructor'),
                        ('TI', 'Tandem Instructor'),
                        ('Coach', 'Coach'),
                        ('IAD-I', 'IAD/Static Line Instructor'),
                    ],
                    max_length=10,
                    help_text='Which rating type this qualification belongs to',
                )),
                ('name', models.CharField(max_length=100)),
                ('description', models.TextField(blank=True)),
                ('portable', models.BooleanField(
                    default=True,
                    help_text='If True, earned once and valid everywhere. If False, each DZ signs off independently.',
                )),
            ],
            options={
                'ordering': ['rating_type', 'name'],
                'unique_together': {('rating_type', 'name')},
            },
        ),
        migrations.CreateModel(
            name='InstructorQualificationEarned',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('signed_off_date', models.DateField(blank=True, null=True)),
                ('notes', models.TextField(blank=True)),
                ('instructor_rating', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='qualifications_earned',
                    to='training.instructorrating',
                )),
                ('qualification', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='earned_by',
                    to='training.ratingqualification',
                )),
                ('signed_off_by', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='qualifications_signed_off',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('dropzone', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='qualifications_issued',
                    to='organizations.dropzone',
                    help_text='DZ where this qualification was earned',
                )),
            ],
            options={
                'ordering': ['-signed_off_date'],
                'unique_together': {('instructor_rating', 'qualification')},
            },
        ),
        migrations.CreateModel(
            name='DropzoneAuthorization',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('authorization_type', models.CharField(
                    choices=[
                        ('AFF-I', 'AFF Instructor'),
                        ('TI', 'Tandem Instructor'),
                        ('Coach', 'Coach'),
                        ('IAD-I', 'IAD/Static Line Instructor'),
                        ('Videographer', 'Videographer'),
                    ],
                    max_length=20,
                )),
                ('stage', models.CharField(
                    choices=[
                        ('restricted', 'Restricted'),
                        ('provisional', 'Provisional'),
                        ('full', 'Full'),
                    ],
                    default='restricted',
                    max_length=20,
                )),
                ('cleared_date', models.DateField(blank=True, null=True)),
                ('notes', models.TextField(blank=True)),
                ('active', models.BooleanField(default=True)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='dz_authorizations',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('dropzone', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='authorizations',
                    to='organizations.dropzone',
                )),
                ('instructor_rating', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='dz_authorizations',
                    to='training.instructorrating',
                    help_text='Leave null for non-USPA roles like videographer',
                )),
                ('cleared_by', models.ForeignKey(
                    blank=True, null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='authorizations_granted',
                    to=settings.AUTH_USER_MODEL,
                )),
            ],
            options={
                'ordering': ['dropzone', 'authorization_type', 'user'],
                'unique_together': {('user', 'dropzone', 'authorization_type')},
            },
        ),
    ]
