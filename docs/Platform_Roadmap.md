# Flare Platform Roadmap

## Goal

Build a unified skydiving management platform connecting individual skydivers, instructors, and
dropzones — not "add manifest to a logbook" but a platform where identity, training, and operations
connect cleanly.

---

## Guiding Principles

- **Modular monolith** — single Django project, clearly separated domain apps
- **One source of truth per concept** — no duplicated models across domains
- **Preserve what works** — GPS analysis, personal logbook, and Tailwind UI are not changing
- **Import concepts, not code** — Manifester models are reference implementations, not copy-paste
- **No big-bang rewrites** — every phase must leave the app functional at the end

---

## Domain App Target Structure

| App | Owns |
|-----|------|
| `accounts` | User identity, auth |
| `organizations` | Dropzone, OrganizationMembership (model layer only, no views) |
| `aircraft` | Aircraft (shared pool, DZ-owned) |
| `logbook` | Jump, JumpType (personal log only) |
| `training` | StudentSignoff, StudentEnrollment, InstructorCertification, ProgramLevel, StudentJump |
| `ratings` | InstructorRate, Canopy, Rig, Waiver |
| `dz` | DZ-operator app space — OperatingDay, CheckIn, DailyRoster, BreakRequest, Reservation, Load, LoadSlot; service layer at `dz/services/` |
| `flights` | Flight, GPS analysis pipeline (unchanged) |
| `analysis` | Supporting analysis (unchanged) |

---

## Phase 0 — Structural Cleanup

**Goal:** Create the right app skeleton without breaking any existing functionality.
All DB tables keep their current names. Only Python-level reorganization.

### 0.1 — Custom Proxy User Model

**Why:** `AUTH_USER_MODEL` must be set before the platform grows further. Switching off
`auth.User` later with production data is very risky. A proxy model costs nothing now.

**What:**
- Create `accounts` app
- Define `User(AbstractUser)` with `Meta: proxy = True`
- Set `AUTH_USER_MODEL = 'accounts.User'` in settings
- Use `SeparateDatabaseAndState` migration — no new DB table created
- Update all existing FKs and imports to reference `accounts.User`

**Risk:** Low. Proxy model maps to the existing `auth_user` table. No data changes.

---

### 0.2 — New `organizations` App (move Dropzone)

**Why:** `Dropzone` in `logbook` is a shared reference object, not a personal log concept.
It will eventually own membership, operating config, and DZ-level settings.

**What:**
- Create `organizations` app
- Move `Dropzone` model there with `class Meta: db_table = 'logbook_dropzone'`
- Use `SeparateDatabaseAndState` in both apps to avoid touching the actual table
- Update all FKs (`logbook`, `users`, `flights`) and all imports
- Leave `home_dz` text field on `UserProfile` alone for now (legacy)

**Risk:** Low. DB table unchanged. Only Python imports and migration state update.

---

### 0.3 — New `aircraft` App (move Aircraft)

**Why:** `Aircraft` is an operational entity owned by a dropzone, not a personal log concept.
It belongs beside manifest and ops, not beside personal jumps.

**What:**
- Create `aircraft` app
- Move `Aircraft` model with `class Meta: db_table = 'logbook_aircraft'`
- Use `SeparateDatabaseAndState` in both apps
- Update all FKs and imports (`logbook`, etc.)

**Note:** Manifester's `Aircraft` model adds ADS-B ICAO hex, per-DZ ownership, weight limits,
avg turn times, and fuel capacity. These fields are deferred to Phase 3 when manifest comes in.

**Risk:** Low. Same pattern as Dropzone.

---

### 0.4 — New `training` App (move student/instructor models)

**Why:** `StudentSignoff`, `PendingInstructorRequest`, ISP criteria logic, and instructor rating
flags are training domain, not personal logbook. The `logbook` app is getting crowded.

**What:**
- Create `training` app
- Move `StudentSignoff` and `PendingInstructorRequest` with their `db_table` names preserved
- Move `isp_criteria.py` logic into `training/`
- Update imports in `logbook/views.py`, `logbook/models.py`, templates
- Instructor boolean flags (`coach`, `affi`, `ti`, `iad_sl`) stay on `UserProfile` for now
  (Phase 1 replaces them with `InstructorCertification`)

**Risk:** Low-medium. More files touch these models than Dropzone/Aircraft.

---

### Phase 0 Completion Criteria

- [ ] `AUTH_USER_MODEL = 'accounts.User'` set and all migrations clean
- [ ] `Dropzone` lives in `organizations`, all references updated
- [ ] `Aircraft` lives in `aircraft`, all references updated
- [ ] `StudentSignoff` and `PendingInstructorRequest` live in `training`, all references updated
- [ ] `logbook` contains only: `Jump`, `JumpType`, `InstructorRate`, `STUDENT_CATEGORY_CHOICES`, `JUMP_METHOD_CHOICES`
- [ ] All existing URLs, views, and templates continue to work without changes
- [ ] `python manage.py migrate` runs clean with no errors
- [ ] No fake or squashed migrations — full history preserved

---

## Phase 1 — Training Layer

**Goal:** Replace simplified ISP approximation with full training pipeline adapted from Manifester.

### Models to introduce (from Manifester, adapted)

- `InstructorCertification` — replaces `coach`/`affi`/`ti`/`iad_sl` booleans on UserProfile
  - role (tandem_instructor, aff_instructor, videographer, coach)
  - clearances (handcam, outside video, hot turns — TI-specific)
  - progression stage (AFF-specific)
  - expiry date, rating number
- `ProgramLevel` — replaces static `uspa_isp.json` approach
  - DZ-configurable levels with USPA criteria M2M
  - Per-DZ customization of advancement criteria
- `StudentEnrollment` — enrollment in a training program per DZ
  - student_type (tandem, aff, coach)
  - current_aff_level, current_coach_level
- `StudentJump` — replaces `student_category`/`student_jump_method` on `Jump`
  - Links StudentEnrollment + ProgramLevel + Instructor
  - Outcome tracking (pass, repeat, partial, incomplete)
  - criteria_passed M2M
  - sign-off tracking
- `Rig` — replaces `Canopy` (or extends alongside it)
  - Container make/model, serial number
  - Reserve last packed date + 180-day currency check
  - AAD type
  - is_primary flag
- `Waiver` — jump type waivers with signed date

### Migration path

- `InstructorCertification` replaces the 4 boolean flags, but booleans stay during transition
- `StudentJump` links to `Jump` via FK (non-breaking addition)
- Old `StudentSignoff`/`PendingInstructorRequest` flow deprecated gradually

---

## Phase 2 — Light Operations

**Goal:** DZ-facing instructor management and student check-in. No full manifest yet. All models go in `dz/`.

### Models to introduce (from Manifester)

- `OrganizationMembership` — user-DZ relationship with role
  - roles: admin, manifest, instructor, coach, fun_jumper, student
  - employment type: contractor, W2
- `OperatingDay` — per-DZ day-of-week schedule
  - Slot capacities per jump type per hour
  - First load time, reservation cutoff
- `CheckIn` — physical arrival log per day
- `DailyRoster` — instructor availability tracking
  - jump_count_today, availability_state
  - thirty_day_avg_at_checkin
- `BreakRequest` — break approval workflow
  - Tied to load landing events (load FK is stubbed until Phase 3)
- `Reservation` — online/phone reservation intake
  - type (tandem, aff, coach)
  - status state machine (pending → confirmed → checked_in → jumped)
  - Availability calculator (capacity per slot type per hour)

---

## Phase 3 — Full Manifest

**Goal:** Complete load manifesting for DZ operations. All models go in `dz/`.

### Models to introduce (from Manifester, with state machine logic)

- `Load` — a single jump load
  - Status state machine: building → on_call → took_off → dropped → descending → landed
  - VALID_TRANSITIONS enforced in model/service
  - Call time cascade logic (last queued load → in-air aircraft → DZ default)
  - reserved_student_slots
- `LoadSlot` — individual jumper assignment
  - jump_type, exit_altitude, weight_lbs
  - role (student, instructor, fun_jumper, video)
  - paired_with (student–instructor–video triplet)
  - media_status for video slots
- `DailyAircraftStats` — per-day performance averages
  - avg_altitude_min, avg_turn_min
  - Sample counts for rolling averages
- `DailyJumpRun` — append-only spot/altitude log

### Jump model update

`Jump` gets an optional FK to `LoadSlot` — personal log entry can now reference the
DZ-authoritative operational record. This completes the "personal vs operational" split
without destroying the existing `Jump` model.

### Frontend surface

Manifest board is the one surface where server-rendered + HTMX hits its limits.
Introduce a thin Vue/Alpine component for the manifest board only.
DRF API endpoints for: loads, slots, roster, check-ins. All under `dz/`.

---

## Phase 4 — Analysis + API Layer

**Goal:** Expose Flare's GPS analysis and operational data via clean APIs.

- DRF API for: flights, analysis results, personal logbook (read-only exports)
- Mobile-friendly endpoints for logbook and training progress
- USPA integration groundwork (credential verification)
- Analysis layer stays isolated — no coupling to manifest logic

---

## What Comes From Where

### Keep from Flare (unchanged)

- All of `flights/` and `analysis/` — GPS pipeline is a differentiator
- `logbook.Jump` — personal log record (thinned over time, not replaced)
- Tailwind dark theme, HTMX, Alpine.js patterns
- `UserProfile`, `Canopy` (adapted, not replaced)

### Adapt from Manifester (schema + business logic, not frontend)

- `Load`, `LoadSlot`, status state machine
- `DailyRoster`, `BreakRequest`, `CheckIn`
- `StudentEnrollment`, `InstructorCertification`, `ProgramLevel`, `StudentJump`
- `Rig`, `Waiver`
- Reservation availability calculator
- DRF patterns for operations API

### Discard from Manifester

- Vue.js frontend entirely
- `SkydiverProfile` model (merge needed fields into `UserProfile`)
- `DropZone` model (adopt schema, rename to `Organization`, place in `organizations`)
- Incomplete/unfinished views and serializers

---

## Technical Notes

### Moving models between apps safely

Use `SeparateDatabaseAndState` in Django migrations to tell the ORM the table already exists
under the old name. Pair with `class Meta: db_table = 'old_app_old_model'` on the new model.
This avoids any DROP/CREATE on production and makes rollback trivial.

### Proxy User model migration

```python
# In accounts/migrations/0001_initial.py
operations = [
    migrations.SeparateDatabaseAndState(
        database_operations=[],  # do nothing to the DB
        state_operations=[
            migrations.CreateModel(
                name='User',
                fields=[],
                options={'proxy': True, 'indexes': [], 'constraints': []},
                bases=('auth.user',),
            )
        ],
    )
]
```

### FK references to moved models

All `models.ForeignKey('logbook.Dropzone', ...)` become `models.ForeignKey('organizations.Dropzone', ...)`.
String references in migrations can stay as-is if the `db_table` is preserved — but app-label
string FKs in model files must be updated.

### Service layer

Not introduced until Phase 3 manifest logic demands it. First appearance:
- `dz/services/load_service.py` — call time cascade, status transitions
- `training/services/progression_service.py` — eligibility checks, advancement logic
