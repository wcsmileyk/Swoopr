"""
Seed USPACategory rows with criteria pools (from uspa_isp.json) and
USPA-recommended jump structures (from the SIM Chapter 1 dive flows).

Safe to run multiple times. Use --overwrite to refresh existing rows.

Usage:
    python manage.py seed_program_levels
    python manage.py seed_program_levels --overwrite
"""
import json

from django.conf import settings
from django.core.management.base import BaseCommand

from training.models import USPACategory

# ---------------------------------------------------------------------------
# USPA SIM Chapter 1 — recommended jump structure per category.
# Each entry under a method_group key is a list of jumps in order.
# dive_flow is the sequence of maneuvers from the SIM dive flow pages.
# criteria_keys lists which top-level criteria keys from the pool apply.
# ---------------------------------------------------------------------------
RECOMMENDED_JUMPS = {
    'fjc': {
        # FJC is ground training only — no in-air jumps
        'all': [],
    },
    'a': {
        'AFF_Tandem': [
            {
                'jump_number': 1,
                'name': 'Cat A',
                'allowed_methods': ['AFF_2I', 'Tandem'],
                'dive_flow': [
                    'Safe exit',
                    'Circle of Awareness (COA)',
                    'Practice deploy (3x)',
                    'COA',
                    '6,000\' lock on (6,500\' for tandem)',
                    '5,500\' wave off and pull (6,000\' for tandem)',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
        'IAD_SL': [
            {
                'jump_number': 1,
                'name': 'Cat A Jump 1',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Safe exit',
                    'Count 1–2–3–4–5',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Cat A Jump 2',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Safe exit',
                    'Count 1–2–3–4–5',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'b': {
        'AFF_Tandem': [
            {
                'jump_number': 1,
                'name': 'Cat B',
                'allowed_methods': ['AFF_2I', 'Tandem'],
                'dive_flow': [
                    'Relaxed exit',
                    'Circle of Awareness (COA)',
                    'Practice deploy',
                    'HAALR',
                    'Extend legs (3 seconds)',
                    'HAALR',
                    'Team turns (if trained)',
                    '6,000\' lock on (6,500\' for tandem)',
                    '5,500\' wave off and pull (6,000\' for tandem)',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
        'IAD_SL': [
            {
                'jump_number': 1,
                'name': 'Cat B Jump 1',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Practice deploy (with count): "Arch, Reach, Pull, Neutral, Check"',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Cat B Jump 2',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Practice deploy (with count): "Arch, Reach, Pull, Neutral, Check"',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 3,
                'name': 'Cat B Jump 3',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Practice deploy (with count): "Arch, Reach, Pull, Neutral, Check"',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'c': {
        # AFF: 2 jumps — first with 2 instructors, second with 1
        'AFF': [
            {
                'jump_number': 1,
                'name': 'Cat C — 2-Instructor',
                'allowed_methods': ['AFF_2I'],
                'dive_flow': [
                    'Relaxed exit',
                    'Circle of Awareness (COA)',
                    'Practice deploy',
                    'COA',
                    'Release',
                    'HAALR',
                    '6,000\' lock on',
                    '5,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Flight-cycle drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Cat C — Solo JM',
                'allowed_methods': ['AFF_solo'],
                'dive_flow': [
                    'Relaxed exit',
                    'Circle of Awareness (COA)',
                    'Practice deploy',
                    'COA',
                    'Release',
                    'HAALR',
                    '6,000\' lock on',
                    '5,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Flight-cycle drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
        # IAD/SL: Clear & Pull (1 jump) + 10-second freefall (2 jumps)
        'IAD_SL': [
            {
                'jump_number': 1,
                'name': 'Cat C #1 — Clear & Pull (4,500\')',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Pull (with count): "Arch, Reach, Pull, Neutral, Check"',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Cat C #2 — 10-Second Freefall (5,500\') Jump 1',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Relax into neutral',
                    'Count to 10',
                    '4,500\' wave off (at 7 seconds)',
                    '4,000\' pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Flight-cycle drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 3,
                'name': 'Cat C #2 — 10-Second Freefall (5,500\') Jump 2',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Exit with legs extended',
                    'Relax into neutral',
                    'Count to 10',
                    '4,500\' wave off (at 7 seconds)',
                    '4,000\' pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Flight-cycle drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'd': {
        'AFF': [
            {
                'jump_number': 1,
                'name': '90° Turns',
                'allowed_methods': ['AFF_solo', 'AFF_2I'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    'Find reference, ask to turn',
                    '90° turns SCS (repeat until 6,000\', no more)',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': '180° and 360° Turns',
                'allowed_methods': ['AFF_solo', 'AFF_2I'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    'Find reference, ask to turn',
                    '180° and 360° turns SCS (repeat until 6,000\', no more)',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
        # IAD/SL: 15-second freefall (2 jumps) + 30-second freefall (2 jumps)
        'IAD_SL': [
            {
                'jump_number': 1,
                'name': 'Cat D #1 — 15-Second Freefall (7,500\') Jump 1',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    '90° turns SCS',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Cat D #1 — 15-Second Freefall (7,500\') Jump 2',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    '90° turns SCS',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 3,
                'name': 'Cat D #2 — 30-Second Freefall (9,500\') Jump 1',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    '180° and 360° turns SCS',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 4,
                'name': 'Cat D #2 — 30-Second Freefall (9,500\') Jump 2',
                'allowed_methods': ['IAD', 'SL'],
                'dive_flow': [
                    'Observe spotting',
                    'Relaxed exit',
                    'COA',
                    'Practice deploy (optional)',
                    'HAALR',
                    '180° and 360° turns SCS',
                    '5,000\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Rear-riser turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'e': {
        'all': [
            {
                'jump_number': 1,
                'name': 'Barrel Roll / Recovery',
                'allowed_methods': ['AFF_solo'],
                'dive_flow': [
                    'Spot (with assistance)',
                    'Solo ungripped exit',
                    'HAALR',
                    'Barrel roll (recover within 5 sec)',
                    'HAALR',
                    'Repeat until 6,000\'',
                    '4,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Sweet-spot drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Front & Back Loops',
                'allowed_methods': ['AFF_solo'],
                'dive_flow': [
                    'Spot (with assistance)',
                    'Choice of exit',
                    'HAALR',
                    'Back loop (recover within 5 sec)',
                    'HAALR',
                    'Front loop (recover within 5 sec)',
                    'HAALR',
                    'Repeat until 6,000\'',
                    '4,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Stall-point drill',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'f': {
        'all': [
            {
                'jump_number': 1,
                'name': 'Flat Tracking — Heading & Pitch',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Choice of exit',
                    'Track for 5 seconds',
                    'Turn 180°',
                    'HAALR',
                    'Repeat until 6,000\'',
                    '4,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Braked turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Flat Tracking — Speed',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Choice of exit',
                    'Track for 5 seconds (focus on speed)',
                    'Turn 180°',
                    'HAALR',
                    'Repeat until 6,000\'',
                    '4,500\' wave off and pull',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Braked turns',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 3,
                'name': 'Clear & Pull (5,500\')',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Floater or poised exit',
                    'Pull within 5 seconds',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Half-braked flares',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 4,
                'name': 'Clear & Pull (no lower than 3,500\')',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Floater or poised exit',
                    'Pull within 5 seconds',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Half-braked flares',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'g': {
        'all': [
            {
                'jump_number': 1,
                'name': 'Forward Movement to Dock',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Rear-float exit (inside strut), coach does the count',
                    'Face direction of flight',
                    'Dock — coach backs up 5–10 feet',
                    'Forward movement to dock',
                    'COA',
                    'Repeat until breakoff',
                    '5,500\' breakoff sequence',
                    'Pull by 4,000\'',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Adjust glide path',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Levels — Up & Down',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Front-float exit (outside strut), student does the count',
                    'Face direction of flight',
                    'Dock — coach backs up 5–10 feet',
                    'Down to match level (5 ft)',
                    'COA',
                    'Up to match level (5 ft)',
                    'COA',
                    'Repeat until breakoff',
                    '5,500\' breakoff sequence',
                    'Pull by 4,000\'',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Turn reversals',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 3,
                'name': 'Level then Dock',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (with minimal assistance)',
                    'Choice of exit, student does the count',
                    'Face direction of flight',
                    'Dock — coach backs up 5–10 feet',
                    'Down to level and dock',
                    'COA',
                    'Dock — coach backs up 5–10 feet',
                    'Up to level and dock',
                    'COA',
                    'Repeat until breakoff',
                    '5,500\' breakoff sequence',
                    'Pull by 4,000\'',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Canopy drills as needed',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
    'h': {
        'all': [
            {
                'jump_number': 1,
                'name': 'Dive to Dock Jump 1 (100 feet)',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (without assistance)',
                    'Diving exit',
                    'Turn to coach',
                    'Dive to dock — use stairstep approach and FAZ',
                    'Repeat if time after coach repositions',
                    '5,500\' breakoff sequence',
                    'Pull by 4,000\'',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Low-turn-recovery practice',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
            {
                'jump_number': 2,
                'name': 'Dive to Dock Jump 2 (100 feet)',
                'allowed_methods': ['AFF_solo', 'Coach'],
                'dive_flow': [
                    'Spot (without assistance)',
                    'Diving exit',
                    'Turn to coach',
                    'Dive to dock — use stairstep approach and FAZ',
                    'Repeat if time after coach repositions',
                    '5,500\' breakoff sequence',
                    'Pull by 4,000\'',
                    'Canopy check — APT',
                    'Locate holding area',
                    'Low-turn-recovery practice',
                    'Fly pattern',
                    'Flare and PLF if needed',
                ],
            },
        ],
    },
}


AFF_LEVELS = [
    # (code, order, name)
    ('fjc', 0, 'First Jump Course'),
    ('a',   1, 'Category A'),
    ('b',   2, 'Category B'),
    ('c',   3, 'Category C'),
    ('d',   4, 'Category D'),
    ('e',   5, 'Category E'),
    ('f',   6, 'Category F'),
    ('g',   7, 'Category G'),
    ('h',   8, 'Category H'),
]


class Command(BaseCommand):
    help = 'Seed USPACategory rows with criteria and SIM dive flows'

    def add_arguments(self, parser):
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite criteria and recommended_jumps on existing rows',
        )

    def handle(self, *args, **options):
        overwrite = options['overwrite']

        path = settings.BASE_DIR / 'docs' / 'uspa_isp.json'
        with open(path, encoding='utf-8') as f:
            criteria_data = json.load(f)

        created = updated = skipped = 0

        for code, order, name in AFF_LEVELS:
            raw = criteria_data.get(code, {})
            criteria = {
                'ground':   raw.get('ground', []),
                'freefall': raw.get('freefall', []),
                'canopy':   raw.get('canopy', []),
                'AFF':      raw.get('AFF', []),
                'Tandem':   raw.get('Tandem', []),
                'IAD/SL':   raw.get('IAD/SL', []),
            }
            recommended = RECOMMENDED_JUMPS.get(code, {})

            obj, was_created = USPACategory.objects.get_or_create(
                program_type='AFF',
                code=code,
                defaults={
                    'name': name,
                    'order': order,
                    'criteria': criteria,
                    'recommended_jumps': recommended,
                },
            )

            if was_created:
                created += 1
            elif overwrite:
                obj.name = name
                obj.order = order
                obj.criteria = criteria
                obj.recommended_jumps = recommended
                obj.save()
                updated += 1
            else:
                skipped += 1

        self.stdout.write(self.style.SUCCESS(
            f'Done. Created {created}, updated {updated}, skipped {skipped} AFF categories.'
        ))
        if skipped:
            self.stdout.write('Use --overwrite to refresh existing rows.')
