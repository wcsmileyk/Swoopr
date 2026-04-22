# Flare Platform Architecture & Direction

## Overview

Flare is evolving from a personal skydiving logbook and analysis tool into a broader platform that connects:

- Individual skydivers
- Instructors
- Dropzones (DZs)
- (Potentially) USPA and external systems

The long-term goal is to create a unified system where:

> Threat intelligence equivalent:  
> Skydiver identity, credentials, and activity → operational systems → actionable outcomes

In this context:
- Personal logbook = user-facing data
- DZ operations = organizational systems
- Training/signoffs = trust layer
- Future integrations = network layer

---

# Core Architectural Principles

## 1. Flare is a Modular Monolith

- Single Django project
- Multiple clearly separated domain apps
- No duplicated models across domains
- Shared core identity and reference data

---

## 2. Separate Domains Early

Avoid letting one app (especially `logbook`) become a dumping ground.

### Target Domain Structure

- `accounts` / `users`
- `organizations` (dropzones)
- `aircraft`
- `logbook`
- `training`
- `ratings`
- `manifest`
- `analysis`
- `integrations` (future)

---

## 3. One Source of Truth per Concept

There should only ever be one canonical model for:

- User / identity
- Dropzone
- Aircraft
- Instructor credentials
- Student progression

Avoid duplication across apps or legacy codebases.

---

# Critical Improvements (Do Before DZ Operations)

## 1. Implement a Custom User Model

### Problem
Currently using Django's default `User` + `UserProfile`.

### Risk
Becomes difficult to extend for:
- roles
- permissions
- verification
- external identity (USPA, etc.)

### Solution
Define a custom user model now:
- Keep it minimal
- Add UUID
- Add verification flags
- Keep domain-specific data outside

---

## 2. Introduce Organization & Membership Layer

### Problem
Dropzones are currently just reference objects.

### Needed Concepts

- `Organization` (Dropzone)
- `OrganizationMembership`
- `RoleAssignment`

### Why

A single user can be:
- Jumper at one DZ
- Instructor at another
- Admin at a third

This layer is required before manifest logic.

---

## 3. Split Logbook Responsibilities

### Problem
`logbook` currently contains:
- personal jumps
- dropzones
- aircraft
- instructor pay
- signoffs

### Solution

Move models into proper domains:

- `Dropzone` → `organizations`
- `Aircraft` → `aircraft`
- training/signoffs → `training`
- instructor rates → `ratings` or `operations`

### Rule

If a model exists without a personal logbook, it does not belong in `logbook`.

---

## 4. Separate Personal vs Operational Records

### Problem

`Jump` is currently both:
- personal log entry
- operational event

### Solution

Split into:

- **OperationalJump / LoadSlot / Flight**
  - DZ-controlled truth

- **PersonalLogEntry**
  - user-facing record
  - derived or manually entered

---

## 5. Separate Claimed vs Verified Data

### Problem

User profiles mix:
- self-claimed data
- authoritative data

### Solution

Introduce:

#### Claimed Data
- User-entered
- Not trusted

#### Verified Data
- DZ-verified
- Instructor-approved
- USPA-linked (future)

---

## 6. Normalize Reference Data

### Dropzone should include:
- timezone
- active status
- ownership
- contact info

### Aircraft should distinguish:
- aircraft type (Otter)
- aircraft instance (tail number, DZ-specific)

### JumpType should eventually split into:
- personal type
- training type
- manifest role
- compensation category

---

## 7. Introduce Service Layer

### Problem
Business logic risks spreading across:
- views
- models
- forms

### Solution

Add:
- `services/`
- `selectors/`
- `policies/`

Examples:
- instructor assignment logic
- eligibility checks
- progression validation

---

## 8. Add Auditability

Before DZ operations:

Track:
- signoffs
- credential changes
- manifest edits
- load assignments
- overrides

Goal:
> Who changed what, when, and why?

---

## 9. Move Toward API-First Backend

Not required immediately, but prepare for:

- DRF-based APIs
- Mobile-friendly interfaces
- Multiple frontends (jumper, instructor, DZ admin)

---

## 10. Protect the Analysis Layer

Flare’s analytics is a key differentiator.

### Rules:
- Keep `analysis` isolated
- Avoid coupling with manifest logic
- Use clean interfaces for data ingestion

---

# Migration Strategy (From Manifester)

## DO NOT merge repositories directly

Instead:

### Step 1
Freeze Manifester development

### Step 2
Map domains:
- loads
- roster
- check-in
- training
- certifications

### Step 3
Import concepts, not code

---

## Migration Order

### Phase 1 – Foundation
- Custom user model
- Organization & membership
- Clean domain separation

### Phase 2 – Training Layer
- ProgramLevel
- StudentJump
- InstructorCertification

### Phase 3 – Light Operations
- CheckIn
- DailyRoster
- BreakRequest

### Phase 4 – Full Manifest
- Load
- LoadSlot
- Assignment logic
- Timing/state transitions

---

# Product Surfaces

## Skydiver Portal
- logbook
- analytics
- ratings
- gear

## Instructor Portal
- signoffs
- availability
- student management

## DZ Portal
- manifest
- roster
- loads
- operations

---

# Long-Term Vision

Flare becomes a platform with:

## Identity Layer
- users
- credentials
- verification

## Personal Layer
- logbook
- analytics
- ratings

## Operational Layer
- manifest
- training
- staffing

## Integration Layer
- USPA
- DZ systems
- external APIs

---

# Guiding Philosophy

- One source of truth per concept
- Separate domains early
- Prefer structure over speed
- Build for evolution, not just MVP

---

# Final Note

The goal is not:

> "Add manifest to a logbook"

The goal is:

> "Build a platform where skydiver identity, training, and operations all connect cleanly"