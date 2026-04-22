import django.db.models.deletion
import django.utils.timezone
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('organizations', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DropzoneMembership',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('role', models.CharField(
                    choices=[
                        ('admin', 'DZ Admin'),
                        ('staff', 'Staff'),
                        ('member', 'Member'),
                    ],
                    default='member',
                    max_length=10,
                )),
                ('granted_date', models.DateField(auto_now_add=True)),
                ('active', models.BooleanField(default=True)),
                ('user', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='dz_memberships',
                    to=settings.AUTH_USER_MODEL,
                )),
                ('dropzone', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='memberships',
                    to='organizations.dropzone',
                )),
                ('granted_by', models.ForeignKey(
                    blank=True,
                    null=True,
                    on_delete=django.db.models.deletion.SET_NULL,
                    related_name='memberships_granted',
                    to=settings.AUTH_USER_MODEL,
                    help_text='User who assigned this role. Admin roles must be granted by a site admin.',
                )),
            ],
            options={
                'ordering': ['dropzone', 'role', 'user'],
                'unique_together': {('user', 'dropzone')},
            },
        ),
    ]
