"""
Seed realistic test data for DZ operations (Phase 2).

Creates:
  - 1 dropzone (Skydive Demo DZ)
  - 6 users: 1 DZ admin, 2 instructors (TI + AFF-I), 1 coach, 2 fun jumpers
  - DropzoneMemberships for all 6
  - InstructorRatings for the 3 rated staff
  - OperatingDay config for the full week
  - CheckIn + DailyRoster entries for today (instructors + coach)
  - 1 approved + 1 pending BreakRequest
  - 12 Reservations across today and the next 6 days
    (mix of tandem/AFF/coach, all statuses, guest + platform-user paths)

Safe to re-run — uses get_or_create throughout.

Usage:
  python manage.py seed_dz_data
  python manage.py seed_dz_data --dz "My DZ Name"   # target an existing DZ by name
"""

import datetime
import random

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.utils import timezone

from dz.models import BreakRequest, CheckIn, DailyRoster, OperatingDay, Reservation
from organizations.models import Dropzone, DropzoneMembership
from training.models import InstructorRating

User = get_user_model()

SEED_PASSWORD = 'testpass123'

USERS = [
    dict(username='dz_admin',    first_name='Alex',    last_name='Torres',   role='admin',  ratings=[]),
    dict(username='ti_sarah',    first_name='Sarah',   last_name='Kline',    role='staff',  ratings=['TI']),
    dict(username='affi_marcus', first_name='Marcus',  last_name='Bell',     role='staff',  ratings=['AFF-I']),
    dict(username='coach_dana',  first_name='Dana',    last_name='Osei',     role='staff',  ratings=['Coach']),
    dict(username='jumper_ryan', first_name='Ryan',    last_name='Park',     role='member', ratings=[]),
    dict(username='jumper_lexi', first_name='Lexi',    last_name='Navarro',  role='member', ratings=[]),
]

GUEST_RESERVATIONS = [
    dict(guest_name='Jordan Smith',   guest_email='jordan@example.com', guest_phone='555-0101', weight_lbs=165, jump_type='tandem'),
    dict(guest_name='Priya Patel',    guest_email='priya@example.com',  guest_phone='555-0102', weight_lbs=130, jump_type='tandem'),
    dict(guest_name='Chris Nguyen',   guest_email='chris@example.com',  guest_phone='555-0103', weight_lbs=180, jump_type='tandem'),
    dict(guest_name='Mia Johansson', guest_email='mia@example.com',    guest_phone='555-0104', weight_lbs=145, jump_type='aff'),
    dict(guest_name='Donte Williams', guest_email='donte@example.com',  guest_phone='555-0105', weight_lbs=195, jump_type='tandem'),
    dict(guest_name='Ava Chen',       guest_email='ava@example.com',    guest_phone='555-0106', weight_lbs=125, jump_type='coach'),
    dict(guest_name='Leo Morales',    guest_email='leo@example.com',    guest_phone='555-0107', weight_lbs=170, jump_type='tandem'),
    dict(guest_name='Zoe Fischer',    guest_email='zoe@example.com',    guest_phone='555-0108', weight_lbs=140, jump_type='aff'),
]


class Command(BaseCommand):
    help = 'Seed DZ Phase 2 test data (idempotent)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dz',
            type=str,
            default=None,
            help='Name of an existing dropzone to target. Creates "Skydive Demo DZ" if omitted.',
        )

    def handle(self, *args, **options):
        today = timezone.localdate()

        # ------------------------------------------------------------------
        # Dropzone
        # ------------------------------------------------------------------
        dz_name = options['dz'] or 'Skydive Demo DZ'
        dz, created = Dropzone.objects.get_or_create(
            name=dz_name,
            defaults=dict(city='Springfield', state='IL', country='USA', icao='KSPI'),
        )
        self.stdout.write(f"{'Created' if created else 'Found'} DZ: {dz}")

        # ------------------------------------------------------------------
        # Users + memberships + ratings
        # ------------------------------------------------------------------
        created_users = {}
        for spec in USERS:
            user, ucreated = User.objects.get_or_create(
                username=spec['username'],
                defaults=dict(
                    first_name=spec['first_name'],
                    last_name=spec['last_name'],
                    email=f"{spec['username']}@example.com",
                ),
            )
            if ucreated:
                user.set_password(SEED_PASSWORD)
                user.save()
            created_users[spec['username']] = user

            DropzoneMembership.objects.get_or_create(
                user=user, dropzone=dz,
                defaults=dict(role=spec['role'], active=True),
            )

            for rating_type in spec['ratings']:
                InstructorRating.objects.get_or_create(
                    user=user, rating_type=rating_type,
                    defaults=dict(earned_date=today - datetime.timedelta(days=365)),
                )

            self.stdout.write(f"  {'Created' if ucreated else 'Found'} user: {user.get_full_name()} ({spec['role']})")

        admin_user    = created_users['dz_admin']
        ti_user       = created_users['ti_sarah']
        affi_user     = created_users['affi_marcus']
        coach_user    = created_users['coach_dana']
        jumper_ryan   = created_users['jumper_ryan']
        jumper_lexi   = created_users['jumper_lexi']

        # ------------------------------------------------------------------
        # OperatingDay — full week config
        # ------------------------------------------------------------------
        day_configs = {
            0: dict(is_open=False),                                          # Monday
            1: dict(is_open=False),                                          # Tuesday
            2: dict(is_open=True,  first_load_time=datetime.time(9, 0),  reservation_cutoff_hours=24, slot_capacities={'tandem': 6, 'aff': 2, 'coach': 4}),
            3: dict(is_open=True,  first_load_time=datetime.time(9, 0),  reservation_cutoff_hours=24, slot_capacities={'tandem': 6, 'aff': 2, 'coach': 4}),
            4: dict(is_open=True,  first_load_time=datetime.time(8, 30), reservation_cutoff_hours=24, slot_capacities={'tandem': 8, 'aff': 3, 'coach': 4}),
            5: dict(is_open=True,  first_load_time=datetime.time(7, 30), reservation_cutoff_hours=48, slot_capacities={'tandem': 12, 'aff': 4, 'coach': 6}),
            6: dict(is_open=True,  first_load_time=datetime.time(7, 30), reservation_cutoff_hours=48, slot_capacities={'tandem': 12, 'aff': 4, 'coach': 6}),
        }
        for dow, kwargs in day_configs.items():
            OperatingDay.objects.update_or_create(
                dropzone=dz, day_of_week=dow, defaults=kwargs,
            )
        self.stdout.write('  OperatingDay config written for all 7 days')

        # ------------------------------------------------------------------
        # Today: CheckIn + DailyRoster for instructors
        # ------------------------------------------------------------------
        instructor_specs = [
            (ti_user,    28.0),
            (affi_user,  19.5),
            (coach_user, 11.0),
        ]
        for instructor, avg in instructor_specs:
            ci, _ = CheckIn.objects.get_or_create(
                dropzone=dz, user=instructor, date=today,
                defaults=dict(checked_in_by=admin_user),
            )
            DailyRoster.objects.get_or_create(
                dropzone=dz, user=instructor, date=today,
                defaults=dict(
                    check_in=ci,
                    availability_state='available',
                    jump_count_today=random.randint(0, 4),
                    thirty_day_avg_at_checkin=avg,
                ),
            )

        # Fun jumpers checked in but not on instructor roster
        for jumper in (jumper_ryan, jumper_lexi):
            CheckIn.objects.get_or_create(
                dropzone=dz, user=jumper, date=today,
                defaults=dict(checked_in_by=admin_user),
            )

        self.stdout.write(f'  CheckIn + DailyRoster seeded for today ({today})')

        # ------------------------------------------------------------------
        # Break requests
        # ------------------------------------------------------------------
        ti_roster = DailyRoster.objects.filter(dropzone=dz, user=ti_user, date=today).first()
        affi_roster = DailyRoster.objects.filter(dropzone=dz, user=affi_user, date=today).first()

        if ti_roster:
            br_approved, _ = BreakRequest.objects.get_or_create(
                dropzone=dz, instructor=ti_user, date=today, roster=ti_roster,
                defaults=dict(reason='Lunch break', status='pending'),
            )
            if br_approved.status == 'pending':
                br_approved.approve(admin_user)

        if affi_roster:
            BreakRequest.objects.get_or_create(
                dropzone=dz, instructor=affi_user, date=today, roster=affi_roster,
                defaults=dict(reason='Need a rest after 4 jumps', status='pending'),
            )

        self.stdout.write('  BreakRequests seeded (1 approved, 1 pending)')

        # ------------------------------------------------------------------
        # Reservations — spread across today + next 6 days
        # ------------------------------------------------------------------
        statuses_today = ['confirmed', 'checked_in', 'jumped', 'cancelled', 'pending', 'no_show']
        time_slots = [datetime.time(8, 0), datetime.time(9, 30), datetime.time(11, 0), datetime.time(13, 0), datetime.time(14, 30)]
        sources = ['online', 'phone', 'walk_in']

        res_count = 0
        for i, guest in enumerate(GUEST_RESERVATIONS):
            res_date = today + datetime.timedelta(days=i % 7)
            status = statuses_today[i % len(statuses_today)] if res_date == today else 'pending'
            time_slot = time_slots[i % len(time_slots)]
            source = sources[i % len(sources)]

            assigned = ti_user if guest['jump_type'] == 'tandem' else (
                affi_user if guest['jump_type'] == 'aff' else coach_user
            )

            _, created = Reservation.objects.get_or_create(
                dropzone=dz,
                guest_email=guest['guest_email'],
                date=res_date,
                defaults=dict(
                    jump_type=guest['jump_type'],
                    time_slot=time_slot,
                    guest_name=guest['guest_name'],
                    guest_phone=guest['guest_phone'],
                    weight_lbs=guest['weight_lbs'],
                    status=status,
                    assigned_instructor=assigned if status not in ('pending',) else None,
                    source=source,
                ),
            )
            if created:
                res_count += 1

        # Two platform-user reservations (jumper_ryan + jumper_lexi as coach students)
        for jumper, res_date_offset in ((jumper_ryan, 1), (jumper_lexi, 2)):
            _, created = Reservation.objects.get_or_create(
                dropzone=dz,
                user=jumper,
                date=today + datetime.timedelta(days=res_date_offset),
                jump_type='coach',
                defaults=dict(
                    time_slot=datetime.time(10, 0),
                    status='confirmed',
                    assigned_instructor=coach_user,
                    source='online',
                ),
            )
            if created:
                res_count += 1

        self.stdout.write(f'  {res_count} Reservations created')

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------
        self.stdout.write(self.style.SUCCESS(
            f'\nDone. DZ: "{dz.name}" | '
            f'Users: {len(USERS)} | '
            f'Roster today: {DailyRoster.objects.filter(dropzone=dz, date=today).count()} instructors | '
            f'Reservations total: {Reservation.objects.filter(dropzone=dz).count()}'
        ))
        self.stdout.write(
            f'\nTest accounts (password: {SEED_PASSWORD}):\n'
            + '\n'.join(f'  {u["username"]:20} {u["first_name"]} {u["last_name"]} ({u["role"]})' for u in USERS)
        )
