# Swoopr course system: requirements and development plan

Status: design draft ready for implementation planning. Smiley has confirmed the Intermediate Distance entry/exit limits, gate-only height-check scope, and full gate-to-gate Accuracy contact requirement (D1–D3 below). Remaining scoring-profile details are explicitly provisional. No application changes, commits, issues, or deployments were made during this review.

Prepared for Smiley from the current Swoopr code and 15 supplied Deep & Steep tracks. Repository snapshot: [`a6a14f7e2ff63daa0f9640a931a843c013315542`](https://github.com/wcsmileyk/Swoopr/tree/a6a14f7e2ff63daa0f9640a931a843c013315542), default branch `master`. Sample GPS timestamps span September 5–16, 2026 UTC. Archive filesystem dates are not flight dates.

## Background and project origin

### The problem this needs to solve

Smiley trains canopy piloting at Mile-Hi Skydiving in Longmont, Colorado, and competes in Drag Distance, Carved Speed, and Zone Accuracy. He already uses Swoopr to analyze flights, but has not had a dependable way to relate those flights to a repeatable competition-course setup. The original question was practical: **if he establishes an entry gate, can the rest of the course be generated and his recorded swoop measured against it?**

This matters more as he works toward competing in Open. In his description of the class progression, Advanced Distance did not require the entry drag that Open Distance does. He wants to understand where his approach meets the entry and exit references, not just whether the flight produced good speed or distance in isolation.

Mile-Hi adds a real constraint. Smiley reports that dragging is not safely practical in most of the available landing area. The area he uses is an empty pond, lower than the surrounding landing area. It provides a training reference, but not water-contact evidence. He lacks reliable references for judging whether he is reaching the intended drag-entry and exit positions. The app must support analysis of this situation without treating the empty pond as a water surface or claiming that a modeled height establishes safe clearance.

There is also a setup constraint: he does not have enough cones, and does not want a field covered in markers. The goal is a **skeleton course** with a small physical marker set and complete virtual geometry for all three disciplines. This is a training arrangement, not a claim that a few cones satisfy the full competition-course setup requirements.

### Why the FAI rules and layouts are here

Smiley supplied the [2026 FAI/ISC Canopy Piloting Competition Rules](https://www.fai.org/sites/default/files/document/file/2026_ISC_CR%20Canopy%20Piloting.pdf) and screenshots of Annex F.1.1, F.2.2, and F.3. These provide the competition reference behind the geometry. They are not permission to hardcode every class-specific training assumption as an FAI rule.

The three layouts serve different purposes even when they share the same entry:

| Discipline | Layout reference used in this design | What the app needs to distinguish |
|---|---|---|
| Drag Distance | Annex F.2.2: straight, 10 m wide, G1 entry and G5 at 50 m. | G5 is a required exit reference, not the end of the distance measurement. Entry-drag evidence and measured distance are separate. |
| Carved Speed | Annex F.1.1: 10 m wide, 70 m centerline arc through 75°, with five gates. | It is a curved course, not a straight 70 m lane. Gate orientation changes around the arc. |
| Zone Accuracy | Annex F.3: four water gates at nominal 12 m spacing, followed by a detailed landing-zone layout. | Gate passage, water-contact credit, and landing-zone points are different checks. The central target is not automatically a complete score. |

These dimensions come from the [supplied FAI layouts](https://www.fai.org/sites/default/files/document/file/2026_ISC_CR%20Canopy%20Piloting.pdf). Section 8 contains the implementation geometry, including the corrected Accuracy center-zone location. The minimum pond dimensions and safety areas shown in the layouts provide site context, but a generated map is not a site-suitability assessment.

Keep three layers separate throughout implementation:

1. **Course geometry:** where the gates, boundaries, and zones are.
2. **Rules profile:** which class, organization, edition, height limits, contact requirements, and penalties apply.
3. **Training options:** deliberate extra checks, such as a continuous-height drill or analysis relative to a virtual surface.

The linked FAI document is the reference for its own rules. Smiley's Intermediate/Advanced/Open requirements are confirmed product inputs where stated, not independently verified rules for every league. In particular, his full gate-to-gate Accuracy contact requirement and optional continuous-height drill must retain their training-profile identity. The differences and remaining questions are documented in §3.

### How we got here

1. **Start with an entry gate and generate the course.** Smiley initially asked how to use his Deep & Steep GPS tracker without a FlySight. Swoopr already had gate-related code, but its workflow and results were unclear. The first goal was to derive usable course coordinates from a small number of known inputs.

2. **Use gSwoop as an interim workflow.** Smiley then realized he could manually record tracks with his tracker. That made the [gSwoop survey methodology](https://gswoop.com/gps.htm)—45-second stops at its prescribed survey points—a useful import model while Swoopr's course system is developed. This is a requirement to understand that measurement method and the Deep & Steep records, not a claim that every Deep & Steep file is already natively compatible with gSwoop. Survey control points, gate endpoints, the center of an exit gate, and the geometric center of a Speed arc must have distinct roles.

3. **Work through two Mile-Hi entry setups.** Google Earth coordinates and rough entry headings were provided and corrected. The eastern-heading setup is approximately 130° with a left carve; the western-heading setup is approximately 303° with a right carve. These are labels for two distinct entry setups, not exact east/west compass bearings or permission to reverse one course automatically. The heading describes the swooper's direction of travel at G1, not the bearing across the gate. The coordinate revisions exposed the need to validate marker roles, width, forward direction, and carve direction before generation. Map-derived coordinates remain estimates, regardless of how many decimal places an export contains.

4. **Ask for maps, then reduce the physical setup.** The requested map coverage was all three disciplines from the eastern setup and Speed from the western setup. Discussion then moved to a skeleton arrangement: selected G1 and Speed markers, the Distance 50 m reference, and a small Accuracy target marker. The exact physical marker selection must remain configurable. Hiding unneeded cones on the setup view must never remove gates from analysis. Likewise, the Accuracy center marker is a target reference, not proof of a 100-point result.

5. **Make KML a first-class input.** Requests for individual coordinates, DMS output for an iPhone Compass, and then separate eastern Distance and Speed KMLs led to the current import requirements. Those two existing KMLs are compatibility inputs for the first release. The workflow must also accept a minimal G1 seed plus heading and generate one, several, or all three disciplines. The Google Earth/iPhone workflow explains why role confirmation, true-heading labels, and honest placement uncertainty are important UI requirements.

6. **Ground the plan in Swoopr and real flight records.** Smiley supplied the GitHub repository and 15 sample Deep & Steep flights, asking for a development plan rather than immediate implementation. The review found problems in parsing, course construction, crossing detection, permissions, and stale results (§2). The flight files establish the input schema and realistic sampling/quality behavior (§4), but they are not examples of completed 45-second survey walks.

7. **Clarify class and contact requirements.** Smiley confirmed Intermediate Distance at 3 m for G1 and 1.5 m for G5 at 50 m; height validity at the gates themselves, with continuous enforcement allowed as a separate training option; and uninterrupted Accuracy contact over the entire gate-to-gate interval. Those decisions are incorporated in the requirements and tests. Unspecified point mappings and governing class rules remain explicit questions, not guessed defaults.

An earlier Accuracy target calculation was corrected during the layout review: the center-zone center is 71 m from G1, not 67 m. Earlier Accuracy coordinates must not be promoted to canonical fixtures. This is one reason generated geometry needs source references, validation, and immutable revisions. The eastern Distance and Speed KMLs are unaffected by that Accuracy correction.

### Goals translated into product requirements

| Smiley's goal | Required behavior | Detail |
|---|---|---|
| Set up a useful course from minimal measurements. | Accept complete KML geometry, a G1 seed with sufficient orientation information, or a gSwoop-style survey track. Preview and resolve ambiguity before saving. | §§5–7 |
| Train any of the three disciplines without rebuilding everything. | Generate separate discipline courses from one shared entry setup. Offer a skeleton display without changing analytical geometry. | §§5, 8 |
| Use Deep & Steep instead of requiring a FlySight purchase. | Decode the actual supplied record format and timestamps. Detect survey stops by elapsed duration, not a device-specific point count. | §§4, 7 |
| Manage courses beyond a one-off personal upload. | Personal public/private course management, usable public courses, and authorized global administration, all with map previews and ownership checks. | §5 |
| Measure the swoop against the course he actually intended to fly. | Applying a course uses the detected G1 crossing as course entry. Preserve existing turn/flare analysis under its own labels. | §§5, 9 |
| Compare Intermediate, Advanced, and Open. | Select a versioned class/rules profile without moving or duplicating course geometry. Honor per-gate exceptions. | §§3, 8–9 |
| Understand height and drag performance without misleading results. | Show calibrated height estimates and uncertainty. Keep observed contact evidence separate from GPS estimates, and unknown distinct from failure. | §§4, 9 |
| Build this into the current app in manageable steps. | Reuse the existing stack, deliver geometry and G1-based analysis first, then survey import and evidence-backed scoring, with regression tests and non-destructive migration. | §§10–12 |

### What success looks like—and what it does not

Smiley can establish or import an entry setup, review the generated courses on a map, place a practical subset of physical markers, and reuse that setup across training jumps. After a flight, he selects the course and class, sees where and when he crossed each required gate, and gets discipline-specific performance with clear evidence and uncertainty labels. A later course correction must not silently rewrite the meaning of an earlier result.

The limiting factor for height is not just writing a better gate-intersection function. The supplied low/fast samples report median vertical accuracy of about 0.82 m, with a median sample interval of 0.25 seconds (§4). Those are receiver-reported quality and timing statistics, not verified error bounds. They do not establish the position of a foot relative to the water throughout a 1.5 m gate or contact interval. Tracker mounting, surface calibration, body position, and independent observations matter.

The intended product is therefore a useful training-analysis system, not an automated competition judge. Horizontal course analysis can be valuable before every scoring question is resolved. It must never manufacture water contact, hide uncertainty to award points, or encourage lower flight to satisfy an unreliable height estimate.

## 1. Recommendation

Build a versioned course subsystem beside the existing flight analysis, not another patch to the current gate JSON. Reuse Django/PostGIS, compressed flight records, the existing Leaflet map, and flight-detail controls. Replace the gate parser, geometry builder, and crossing evaluator behind a compatibility layer.

This file can be placed at `docs/design/course-system.md` in the repository when implementation begins. The requested GitHub connection was used read-only for this review.

The product has four distinct responsibilities:

1. Import measurements: KML geometry, gSwoop-style survey tracks, or manually entered G1 coordinates and heading.
2. Generate and manage courses: Distance, Carved Speed, and Zone Accuracy, sharing an entry setup where requested.
3. Apply a course and class/rules profile to a flight, using the actual G1 crossing as the course-entry event.
4. Present course-specific performance, rule checks, uncertainty, and independently supplied contact observations.

The essential boundary: GPS can estimate a path and height. It cannot, from these files alone, certify water contact, first body contact with land, a stand-up landing, or canopy-down. Missing evidence must remain unknown, not silently earn points or cause a zero.

### First useful release

Import the two eastern KMLs already created, preview them on a map, save private/public courses, select one on a flight, and show interpolated G1/G5 crossings plus Speed G2–G4. Include class selection, heading, lateral offset, speed, elapsed time, and honest height uncertainty. Support manual water-contact/landing observations before attempting any automated scoring claims. gSwoop survey import is the next independently testable slice.

### Non-goals

- Official electronic judging or competition certification.
- Real-time low-altitude flight guidance or prompts to fly lower.
- Inferring safe terrain clearance from satellite imagery or an empty pond.
- Replacing canopy/turn/rollout analysis, redesigning the entire site, or introducing an unrelated frontend stack.
- Meet leaderboards or normalized competition points in the initial release.

## 2. What exists, how it works, and why it fails

The existing flow is: upload CSV/GSW from flight detail or admin → `GateFileParser` → `CompetitionGate.gate_positions` → select course on a flight → explicitly calculate metrics. Admin parsing also calls `CourseBuilder`; the user upload route does not. Documentation saying upload UI is a future enhancement is stale: controls and endpoints exist in this snapshot.

| Area | Observed implementation | Required change |
|---|---|---|
| Data model | `CompetitionGate` combines source upload, entry markers, optional exit point, and mutable generated JSON; types are only `standard` and `speed`. | Separate entry setup, discipline, geometry revision, import provenance, and analysis result. |
| Survey parsing | Filters out all rows starting with `$` before testing for `$GNSS`. | Parse record type before ignoring metadata; shared normalized track decoder. |
| Dwell detection | Spatial DBSCAN in degrees; sample-count thresholds; clusters sorted by number of points. | Time-contiguous dwell intervals in meters, elapsed duration, chronological role mapping, repeat-A check. |
| Survey geometry | Averages four cluster positions as an inside G1 marker and guesses bearing from sorted bearings. | Intersect surveyed longitudinal/transverse axes; explicitly resolve G1 center, travel direction, and marker roles. Unequal survey arms must work. |
| Course geometry | Speed is a straight quadrilateral with a 4 m exit; no intermediate curved gates. Accuracy is five 2 m strips starting at entry. | Replace with discipline templates and complete numbered gate pairs/zones. |
| Crossing | Tests the infinite gate line, not the finite gate segment; no forward direction or final-approach restriction. | Directed, bounded, time-interpolated crossing events within the selected approach. |
| False entry fallback | If no crossing exists, nearest approach is returned as a crossing result with no maximum-distance cutoff. | Return `not_crossed` or `unresolved`; show nearest approach only as a separate diagnostic. |
| Width test | Requires distance to both markers to be strictly less than 5 m. For an exact 10 m gate this is impossible by the triangle inequality. | Test the interpolated position along the finite gate span, with uncertainty. |
| Metrics | Uses the sample after the intersection; legacy `entry_gate_speed_mps` actually contains flare speed. | Separate flare speed and G1 crossing speed, including fractional crossing time. |
| Stale results | Changing the selected gate does not clear existing gate metrics; removal clears them. | Invalidate results on any course, class, calibration, observation, or track change. |
| User upload | `get_or_create(name=gate_name)` can update another user's same-named object; source file, owner, center, and generated config are not populated by that route. | Owner-scoped IDs, source retention, preview/commit workflow, transactions; no global name-based overwrite. |
| Access | Map/data views require login but have no object-level course visibility checks; available gates query includes every parsed gate. | Central read/use/edit/publish permission policy. |
| Tests | `flights/tests.py` and `api/tests.py` are stubs; archived parser script is not a comprehensive regression suite. | Automated geometry, parser, authorization, integration, and uncertainty tests. |

Code evidence: [parser](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/utils/gate_parser.py), [course builder](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/utils/course_builder.py), [calculator](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/utils/gate_calculator.py), [models](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/models.py), [user views](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/users/views.py), [map access](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/views.py).

Read-only reproductions against the isolated utility modules confirmed: a clean center crossing returns `passed_between_gates=False`; an outside crossing still returns crossing metrics; a track roughly 100 m away that never crosses still returns metrics; one syntactically valid `$GNSS` row yields zero parsed points. Full Django integration tests were not run: this was a source review, not a configured database/application deployment.

## 3. Rules: preserve the request, make ambiguities explicit

Course geometry, competition class, and rule edition are separate selections. Changing class must not duplicate or move the course.

### Requested training profiles

| Event/check | Intermediate | Advanced | Open |
|---|---|---|---|
| Default gate-height limit | 3.0 m, with the Distance G5 exception below. | 1.5 m | 1.5 m |
| Speed | Check each of G1–G5 against class height; no valid performance if a required gate is failed. | Same, 1.5 m. | Same, 1.5 m. |
| Distance water drag | Not required, inferred from the requested class progression; confirm. | Not required per request. | Contact at/before G1 required; exact boundary semantics selected by profile. |
| Distance height | G1 entry: 3.0 m. G5 exit at 50 m: 1.5 m. Check at the required gates; optional continuous-height drill is separate. | G1/G5: 1.5 m, checked at the required gates; optional continuous-height drill is separate. | G1/G5: 1.5 m, checked at the required gates; optional continuous-height drill is separate. |
| Accuracy gate passage | 3 m default; class-specific drag/scoring details need confirmation. | 1.5 m gate-passage test, separate from water-contact points. | Same separation as Advanced. |
| Accuracy water contact | Class-specific scoring remains profile-controlled; do not invent missing Intermediate rules. | Uninterrupted water contact throughout each required gate-to-gate interval for drag credit; passing the height check alone does not earn drag points. | Same full-interval contact requirement as Advanced. |

These are Smiley's training requirements, not verified USCPA class rules. Profiles must record organization, edition, class, verification status, and source. A custom training profile can support a deliberate continuous ceiling without labeling it an official rule.

Confirmed on September 16, 2026: Intermediate Distance uses 3 m at entry and 1.5 m at the 50 m exit; height validity is checked at the gates themselves; Smiley also permits a stricter continuous training check. Accuracy contact means the entire interval between gates, not just contact at the two endpoints. These confirmations resolve D1–D3 as product requirements, without upgrading them to independently verified competition rules.

### Important differences from the linked FAI document

FAI 2026 checks Drag Distance VE at G5, not a continuous 1.5 m ceiling throughout G1–G5. Drag may occur at or before G1's leading edge. A water-gate score refers to uninterrupted contact across its leading-edge plane, not automatically the entire interval between successive gates. Penalties after a scored G1 may produce a default result rather than zero. Therefore retain raw performance, penalty reason, and scoring outcome separately. See [FAI 2026 §§2, 6.1, 6.5, 6.8](https://www.fai.org/sites/default/files/document/file/2026_ISC_CR%20Canopy%20Piloting.pdf).

### Confirmed decisions and remaining scoring questions

| ID | Decision / question | Implementation and status |
|---|---|---|
| D1 — confirmed | Intermediate Distance: G1 3.0 m; G5 at 50 m 1.5 m. | Store per-event, per-gate overrides. Do not apply the 3 m class default to Distance G5. |
| D2 — confirmed | Height validity is evaluated at required gates. Continuous enforcement is allowed as a training option. | Gate-only validity is the default; keep an optional, explicitly configured continuous-height drill and its result separate. Intermediate has mixed gate limits, not an implicit continuous 1.5 m competition ceiling. |
| D3 — confirmed | Accuracy drag credit requires uninterrupted contact over the full gate-to-gate interval. | Represent named intervals with start/end gate references. For the standard layout these are G1→G2, G2→G3, and G3→G4 (12 m each); use the actual imported geometry. Endpoint-only contact does not satisfy the interval requirement. |
| D4 | Which competition/league and edition define Intermediate and Advanced? What are Intermediate Accuracy points? | Ship unverified training presets; accept the actual meet rules before publishing verified scoring presets. |
| D5 | Does “no score” mean no valid measured performance, zero, or a default result? | Preserve separate result categories and nullable score. Never equate `unknown` with zero or disqualification. |
| D6 | Where is the tracker mounted, and is a measured water/surface reference available? | Horizontal analysis works without this. Foot-height and contact assessments remain unknown until calibration/evidence is supplied. |
| D7 | How do the three Accuracy intervals map to four water-gate point awards, and what is the separate G1 contact requirement? | Store interval compliance now. Require an explicit class-profile mapping and G1 entry-contact policy before computing a complete water-point subtotal; do not invent an upstream interval or silently assign points to either end. |

## 4. Track-data findings and what height analysis can support

The archive contains 15 CSVs and 11,524 valid timestamped records. Header:

```text
time,lat,lon,hMSL,velN,velE,velD,hAcc,vAcc,sAcc,gpsFix,numSV,heading,headAcc
```

There is a units row after the header. Timestamps include variable fractional precision. Altitudes are MSL meters, velocities are meters/second, and positive `velD` indicates descent. No barometric-altitude or body-pose stream is present. The `sAcc` units row says `(m)`, despite the velocity-accuracy field convention and Swoopr interpreting it as m/s: retain the raw units and verify the device schema rather than silently assuming every units label is correct.

### Measurement method

These are screening windows, not video-verified swoop/gate intervals. For each file:

1. Estimate a provisional sensor-height reference from the median `hMSL` among final-30-second samples with horizontal speed below 1 m/s and absolute `velD` below 0.5 m/s. If fewer than five samples exist, fallback to the last-five-second median and label it. All 15 files supplied enough samples for the first method.
2. Descent window: final 180 seconds, relative height −10 to 609.6 m, speed above 5 m/s, downward velocity positive, and preceding sample interval in (0, 1] seconds.
3. Low/fast window: final 120 seconds, relative height −10 to 10 m, speed above 10 m/s, same timestamp constraint; allow small climbs because plane-out should not be excluded.
4. Pool samples across files; report medians and empirical 95th percentiles of the reported fields. These percentiles describe the dataset, NOT confidence bounds on actual positional error.

The 2,000 ft ceiling means approximately 609.6 m above the provisional local reference, not MSL; Mile-Hi's MSL elevation itself exceeds 2,000 ft. Short terminal anchors span only 1.49–6.00 seconds, so they are weak calibration evidence, not surveys. Slow samples can also be noncontiguous or include movement; production calibration needs an explicit interval and known posture/location.

| Statistic | Final descent: 2,270 samples | Low/fast: 314 samples |
|---|---:|---:|
| Reported hAcc median / P95 | 0.569 / 0.908 m | 0.670 / 0.898 m |
| Reported vAcc median / P95 | 0.784 / 1.125 m | 0.822 / 1.100 m |
| Reported sAcc median / P95, numeric field | 0.120 / 0.331 | 0.145 / 0.348 |
| Timestamp spacing median / P95 | 0.25 / 0.50 s | 0.25 / 0.50 s |
| Distance between samples median / P95 | 5.14 / 9.70 m | 5.64 / 10.73 m |

All screened low/fast samples report `gpsFix=3`. No non-increasing timestamps were found in these files. That does not eliminate missed samples, bias, or poor height calibration. At 30 m/s, 0.25 s is 7.5 m and 0.5 s is 15 m: interpolation is mandatory, but it does not recreate unobserved motion.

Correction to the earlier conversational advice: these supplied files predominantly contain 4 Hz high-rate timestamps, not the 3 Hz described in the manual previously consulted. Parse real timestamps, never assume a device-wide fixed rate.

### Conclusions

- Horizontal crossing location, heading, and speed are useful training estimates, with uncertainty near a boundary.
- Many 1.5 m height decisions will overlap the error budget; even 3 m gates cannot be classified reliably when calibration or pose is unknown.
- `vAcc` is a receiver estimate, not measured truth or a guaranteed bound. There is no independent ground truth in these CSVs to validate actual accuracy.
- Toe contact needs centimeter-scale contact evidence and body position. A GNSS height curve crossing zero cannot prove a water drag.
- Existing AGL uses a low quantile of the final 90 seconds, without proving those samples represent the local surface. In this dataset that baseline differs from the provisional terminal anchor by −0.048 to +0.590 m; neither baseline is ground truth. That difference alone matters near a 1.5 m threshold.
- A 45-second stand at 4 Hz supplies roughly 180 samples, not necessarily 180 independent observations. Averaging reduces some noise, not common bias or inter-session drift.

### Height model

Store the following independently:

1. Raw device MSL height and reported accuracy, unchanged.
2. Surface model: actual water plane, land/terrain profile, or virtual training plane; datum, reference source, date, and uncertainty.
3. Per-flight vertical bias calibration, its measured anchor, and uncertainty.
4. Device mount and device-to-lowest-body-point/foot offset model with a range, not just a universal constant.
5. Gate-top reference: water/ground elevation at that gate plus the selected class limit.

Conceptually:

```text
estimated_foot_height_above_water(t)
  = device_hMSL(t) - track_height_bias(t)
    - device_to_foot_vertical_offset(t) - water_surface_msl
```

If using a known standing anchor, compute bias from measured sensor elevation, not bare ground elevation. Do not subtract a foot offset twice after zeroing the sensor to a standing baseline. A bent knee, wrist-mounted tracker, leaned harness, and seated posture invalidate a fixed offset. Gate passage can involve any qualifying body part; estimated foot passage is a labeled proxy, not a full body model.

Two supported calibration routes: surveyed surface elevation with datum compatibility; or same-session stationary measurements at a known height relative to a reference surface, with recorded vertical difference to the water/virtual plane. A separate day's survey and flight may have different GNSS bias. Use water level per session; a static course revision alone does not establish today's water elevation.

Missing calibration or unknown vertical datum means `height_status=unknown`. KML `clampToGround` zeros are not zero MSL and are not water elevation. This applies to both supplied eastern KMLs.

For calibrated estimates, use a documented engineering uncertainty interval including receiver estimate, calibration, surface, mount/pose, interpolation, and drift. Until field validated, do not call this a 95% confidence interval. A conservative initial budget may add bounded components; quadrature is allowed only when independence/variance assumptions are justified. Smoothing must not claim to remove bias or narrow uncertainty through artificial sample multiplication.

| Condition | Height result |
|---|---|
| Entire supported height interval below gate top | `estimated_clear` |
| Entire interval above gate top | `estimated_ve` |
| Interval overlaps gate top | `uncertain` |
| Missing reference, offset, adequate samples, or quality | `unknown` |

Water-contact state is a separate enum: `confirmed_contact`, `confirmed_no_contact`, `unknown`, with provenance `observer`, `video`, or `manual_unverified`. GPS may add `geometrically_plausible_contact`; that cannot automatically become confirmed contact. Manual observations record author, time/window, source, confidence, and revision. Below-surface estimates should trigger a calibration review, not a successful drag badge.

For Mile-Hi, virtual-water training is supported as a separate mode. It evaluates geometry relative to a chosen plane and never awards actual water-contact points. Choosing a plane does not establish safe clearance over an empty pond, berms, or surrounding terrain; site suitability belongs to an on-site assessment.

## 5. Functional requirements and workflows

### R1 — Import wizard

1. Upload KML or survey CSV; display filename, format, and parsed features/records.
2. Choose input mode: explicit course geometry, G1 seed, or gSwoop survey.
3. Select one discipline, any subset, or all three. Select carve direction whenever Speed is generated.
4. Resolve marker roles, forward heading, source units, and missing values; show confidence/errors.
5. Preview measured/imported features versus generated features on a map, with entry arrow and measured dimensions.
6. Name the course set, set visibility (private by default), and commit atomically. Failed preview creates no active course.

All-three generation means one entry setup with three separate discipline geometries, not one ambiguous combined polygon. An explicit Speed KML can seed Distance/Accuracy from its G1; its other gates remain Speed gates. Choosing one discipline must create only that discipline. Multiple unrelated G1 setups in a file require a setup selection; never mix them by nearest name alone.

### R2 — Personal management

“My Courses” and “Public Courses” views, searchable by location, name, discipline, and owner. Users can create, preview, duplicate, revise, archive, import/export, and change visibility of their own sets. Public courses are usable/forkable, not editable by other users. Copies retain attribution but receive new ownership and IDs. Identical names across owners are legal.

Map controls: discipline toggles, gate labels, centers, endpoints, entry arrow, centerline, boundaries, accuracy zones, measured/source uncertainty, and optional skeleton-marker display. Skeleton display only hides visual features; it must not delete analytical gates. Coordinate editor accepts decimal degrees and DMS, stores WGS84, and labels heading as true north. Smartphone rounded compass coordinates must not be presented as survey precision.

### R3 — Global management

Staff with an explicit `manage_all_courses` permission can search all owners/visibility states, review failed imports, moderate public entries, revise official templates/profiles, and archive problematic courses. Every privileged mutation is audited. `is_staff` alone should not grant arbitrary course-administration authority. Template geometry and rules profiles have a review/publish action distinct from making a user's course public.

### R4 — Permissions and retention

Enforce read/use/edit/export/source-download permissions server-side for every page, JSON endpoint, import, assignment, and background job. A private course is visible only to owner and authorized admins. Source survey tracks remain private even when derived course geometry is public. Public flights must not leak private course names, source files, calibration notes, or geometry through hidden JSON, export, or URLs; render only permitted course metadata. Course publication does not publish any flights.

Public-to-private changes revoke discovery and fresh assignments. Existing third-party assignments retain a private, immutable historical snapshot for that flight owner's reproducibility; warn the publisher that prior copies cannot be recalled. Other public viewers lose the now-private overlay. Decide this retention policy before launch and test it explicitly. Archive referenced revisions; do not cascade-delete historical analysis.

### R5 — Apply course to flight

Owner selects course revision, event, class/profile, calibration, and mode (`training_estimate` or `virtual_water`). Apply/recalculate returns an analysis ID and state, then shows result when ready. Allow comparisons of the same flight under multiple disciplines/classes without overwriting the underlying track; designate one primary analysis for the main display.

For course-based metrics, G1 crossing is entry time and distance zero. Preserve original flare, turn, rollout, landing detection, and their metric labels. If there is no validly detected G1 crossing, show `entry_not_detected` or `entry_unresolved`; do not substitute flare or closest approach. The pre-entry approach remains visible for drag-at/before-G1 evidence. A user-adjusted approach window is an auditable input, not an edit to raw GPS.

Course analysis may run even if automatic swoop classification failed, when the user explicitly selects an approach. Respect `swoop_rejected` for dashboard inclusion. Do not let an unverified course result silently become a personal best or public leaderboard entry.

## 6. KML import contract

Use standard KML 2.2: namespace-aware parsing, nested Document/Folder elements, Point and LineString geometries, and relevant MultiGeometry containers. KML coordinates are longitude, latitude, altitude; keep a single internal convention and convert explicitly for Leaflet. Preserve altitude mode and ExtendedData. Google documents these semantics in its [KML reference](https://developers.google.com/kml/documentation/kmlreference).

### Existing files must import unchanged

- `mile_hi_eastern_distance.kml`: G1/G5 Inside, Outside, and Center; gate lines; boundaries and centerline for the first 50 m only.
- `mile_hi_eastern_speed.kml`: G1–G5 Inside, Outside, and Center; five gate lines; curved boundaries and centerline.

Recognize those names case-insensitively, scoped to course/folder. Do not mistake reference lines, centers, or duplicate gate-line geometry for additional gates. Points and lines describing the same gate must reconcile within a configured tolerance (initial 0.20 m, an import-consistency tolerance, not survey accuracy). Conflicts require resolution. Three coordinates in a centerline are not three gates. For arbitrary names, provide a role-mapping preview.

An explicit geometry import preserves uploaded positions. Validate 10 m width, expected gates, forward direction, plausible separation, and curve shape; deviations produce warnings or a custom-course label. “Normalize to template” is a separate consented action producing a new revision, never silent coordinate movement. Missing Accuracy zones may be generated from the confirmed G1/template and labeled generated.

### G1 seed modes

| Supplied seed | Additional required inputs |
|---|---|
| G1 endpoint pair | Forward choice/heading if ambiguous; Speed carve direction. Derive midpoint and measured width. |
| G1 center Point | True entry heading; width default 10 m with confirmation; Speed carve direction. |
| A single side marker | Explicit left/right or inside/outside role, heading, width, and carve direction where inside/outside is used. Never assume it is the center. |

Inside/outside only has an unambiguous curved-course meaning after carve direction is known. Internally keep left/right relative to forward travel; display inside/outside from handedness. A provided heading inconsistent with an endpoint pair's normal triggers a warning rather than rotating the pair. Legacy eastern labels imply inside=left for the left carve.

Proposed Swoopr ExtendedData contract (all optional for legacy imports, required as appropriate for new exports):

```xml
<ExtendedData>
  <Data name="swoopr_schema_version"><value>1</value></Data>
  <Data name="course_key"><value>mile-hi-east</value></Data>
  <Data name="geometry_mode"><value>seed</value></Data>
  <Data name="entry_heading_deg_true"><value>129.6456265</value></Data>
  <Data name="course_width_m"><value>10</value></Data>
  <Data name="carve_direction"><value>left</value></Data>
</ExtendedData>
```

Gate features add `discipline`, `gate_number`, `feature_role` (`left_endpoint`, `right_endpoint`, `center`, `gate_line`, `boundary`, `zone`), and measurement reference (`marker_center` versus `leading_edge`). Metadata takes precedence over names, but contradictions are reported. If heading is absent, collect it in the wizard or derive it only when geometry supplies an unambiguous direction.

Reject external entities/DTD and remote NetworkLink fetching. Do not download icons/assets referenced by KML. Strip active HTML from descriptions; bound file size, element count, nesting, and coordinates. Proposed initial limits: 10 MiB, 100,000 coordinate tuples, depth 32. Reject NaN/infinity, out-of-range values, duplicate contradictory gate IDs, degenerate gates, and impossible zone polygons. KMZ support is deferred; reject it clearly rather than accidentally parsing compressed bytes as XML.

## 7. gSwoop survey import

The source procedure records timed stops at A, B, C, D, A again, then optionally Speed G5 center; each stop is 45 seconds. A is repeated intentionally. The diagram places the gate at the crossing of A–C and B–D; down-course points from C toward A. See [gSwoop procedure](https://gswoop.com/gps.htm) and [survey diagram](https://gswoop.com/images/pondwalk.jpg). Support this methodology with Deep & Steep CSVs without claiming native gSwoop compatibility for that device.

### Pipeline

1. Decode timestamps/coordinates/quality using the same normalization layer as flight imports. Preserve source-row IDs and raw units. Never synthesize 5 Hz timestamps for survey evaluation.
2. Work in local metric coordinates. Detect contiguous dwell candidates from measured velocity when available, displacement, spatial spread, and timestamp continuity. Use elapsed seconds, not 50/100/225-sample constants.
3. Suggested initial detector tuning: target 45 s; flag candidates shorter than 35 s; trim approximately 3 s of arrival/departure movement; use speed around 0.5 m/s as one signal, not the only test. Break on substantial time gaps. These thresholds are tunable and require real-survey validation.
4. Preserve all stops chronologically. Assign A1, B, C, D, A2, optional E (G5 center) in the preview. Extra stops, missing stops, and merged noisy intervals require explicit role/time-window editing. Do not reject everything merely because the count is not exact.
5. Estimate robust centroids and per-dwell dispersion/quality. Reject or flag jumps and bad fixes. Keep A1/A2 distinct long enough to quantify repeatability and drift. Agreement supports a weighted combined A; disagreement above the quality budget requires user review, not automatic correction of the entire survey.
6. Intersect the A–C longitudinal line and B–D transverse line in meters. Use the intersection as the proposed G1 center, C→A as the forward orientation, and the transverse line as a perpendicularity check. Display and confirm roles. Unequal arms are valid; arithmetic mean of the four stops is not the intersection in general. Near-parallel axes or very poor perpendicularity are errors.
7. Construct G1 endpoints using the confirmed width and direction. Ground stops are axis-control points, not necessarily cone locations. Let the user explicitly override center/side interpretation if they walked a different protocol.
8. Optional E validates the generated Speed G5 center and can help resolve handedness. Under standard-generation mode, residuals create a warning; do not distort radius or move G1 to force a fit. An explicit custom-fit mode is separately labeled nonstandard. Without E, generate Speed from G1 + heading + selected carve.
9. Survey heights record where the GPS was held, not the water surface. Capture hold-height and surface relationship only when actually measured; otherwise use the survey horizontally.

The supplied archive contains flight tracks, not known 45-second survey recordings. It validates decoding, timing, and realistic field quality—not dwell-role detection or survey accuracy. Do not interpret post-landing stops from these flights as a completed gate survey. Before calling this feature ready, obtain at least one real A–B–C–D–A survey with independently checked points; test with and without E and with both carve directions.

## 8. Canonical geometry and schema

Use local meters for geometry, WGS84 lon/lat for persistence/export, true-north bearings, meters for distances/heights, seconds for time, and m/s for speeds. Use a tested projection/geodesic implementation, not several inconsistent degrees-to-meters formulas. Keep projection origin and algorithm version in the geometry snapshot.

### Local coordinates and generation

Let G1 center be origin, x forward, y left. For true heading h:

```text
east  = x*sin(h) - y*cos(h)
north = x*cos(h) + y*sin(h)
```

For Speed, theta=75 degrees, s=+1 for left or −1 for right, and R=70/radians(75). R rounds to 53.48 m. The existing KMLs used rounded R=53.48, making the arc approximately 70.004 m; preserve imports and treat this millimeter-scale template difference as insignificant. Generate a canonical exact-70 m arc using:

```text
center(phi)  = (R*sin(phi), s*R*(1-cos(phi)))
forward(phi) = (cos(phi), s*sin(phi))
left(phi)    = (-s*sin(phi), cos(phi))
gate sides  = center(phi) +/- (width/2)*left(phi)
phi_i       = (i-1)*75 degrees/4, for i=1..5
```

Left/right endpoints are converted to inside/outside using s. Build the entire curved corridor, not the convex hull or chord rectangle. Centerline stations are 0, 17.5, 35, 52.5, and 70 m. The gate normals follow the local tangent; G5 is not perpendicular to the entry heading.

Distance uses G1 at x=0 and G5 at x=50, width 10 m. Continue the analytical sideline corridor beyond G5. Keep a separately configured landing-area extent, or report terminal-boundary status unknown. A KML whose lines end at 50 m must not automatically invalidate longer distances.

Accuracy geometry was checked against the user's layout and the full-size PDF rendering. Water gates are nominally x=0,12,24,36; waterline 44; Z1/Z2 at 50; example end 78. The waterline and gate spacing are configurable measured features, not a reason to shift the fixed landing-zone reference.

| Zone | Longitudinal extent from G1 (m) | Lateral extent y (m, left positive) | Diagram points |
|---|---:|---|---:|
| Z1 | 44–50 nominal | −5 to 5 | 3 |
| Z2 | 50–56 | −5 to 5 | 11 |
| Z3 | 56–61 | −5 to 5 | 19 |
| Z4 | 61–65 | −5 to 5 | 27 |
| Z5 | 65–68 | −5 to 5 | 34 |
| Z6 | 68–70 | −5 to 5 | 41 |
| Z7 | 70–72 | −5 to−1.5 and 1.5 to 5 | 46 |
| Z8 | 70–72 | −1.5 to−0.5 and 0.5 to 1.5 | 48 |
| CZ | 70–72 | −0.5 to 0.5 | 50 |
| Z9 | 72–74 | −5 to 5 | 25 |
| Z10 | 74–78 | −5 to 5 | 5 |

Water-gate diagram points are 21,5,8,16. Keep point tables in the selected rule profile; class variations must not be inferred from geometry. Encode each disconnected Z7/Z8 region as a MultiPolygon or explicit polygon list. Boundary ownership and exact contact on shared edges need tests against the chosen profile, not incidental polygon iteration order.

Correction to earlier messages: CZ is a 2 m longitudinal by 1 m lateral box, centered at **71 m**, not 67 m. The earlier Accuracy target coordinate must not be used as a canonical regression fixture. Existing eastern Speed and Distance KMLs contain no Accuracy target and are unaffected. This correction follows the full-resolution visual review of [Annex F.3](https://www.fai.org/sites/default/files/document/file/2026_ISC_CR%20Canopy%20Piloting.pdf).

### Proposed Django models

| Model | Responsibility / key fields |
|---|---|
| `CourseSet` | UUID, name, owner, optional site, visibility, status, created/updated; current entry-setup revision and discipline list. |
| `CourseImport` | Owner, private source FileField, SHA-256, format, parser version, input metadata, extracted features/dwells, validation report, status, expiration for abandoned previews. |
| `Course` | Stable discipline identity within set; current revision; archive flag. |
| `CourseRevision` | Immutable revision number, canonical geometry JSON, entry origin/heading/width/handedness, generation parameters, source/import reference, surveyed/estimated flags, geometry hash, creator/change reason. |
| `RuleProfileRevision` | Immutable organization/edition/class, discipline-specific checks, per-gate height overrides, gate-only validity policy, optional continuous training-check configuration, water-contact intervals and award mapping, penalty mapping, points table, source references, verification state. |
| `FlightCourseAnalysis` | Flight + exact course/profile revisions + calibration/observation revisions; engine version, input hashes, status, metrics, per-gate checks, per-interval contact evidence, separate training-check results, warnings and result. Multiple analyses per flight, one primary. |
| `FlightCourseEvidence` | Versioned calibration and manual/video observations; author, source, times/positions, validity, confidence; private by default. |
| `CourseAuditEvent` | Actor, action, target, prior/new version references, timestamp, moderation/change reason. |

For MVP, store gates and zones inside schema-validated revision JSON to avoid maintaining duplicate JSON and relational geometry. PostGIS Point/bounding geometry on Course/Revision supports location filtering. If relational Gate/Zone models are introduced later, designate one authoritative representation and derive the other.

Canonical gate record:

```json
{
  "id": "speed:G4",
  "gate_number": 4,
  "station_m": 52.5,
  "left_endpoint": {"lon": -105.16477991, "lat": 40.16166218},
  "right_endpoint": {"lon": -105.16474636, "lat": 40.16157588},
  "measurement_reference": "marker_center",
  "longitudinal_marker_depth_m": null,
  "source": "generated_from_g1",
  "horizontal_uncertainty_m": null,
  "surface_reference_id": null
}
```

These are the eastern left-carve G4 endpoints previously calculated. Height belongs to the profile plus surface calibration, not the permanent coordinates. Unknown marker depth prevents pretending a cone-center line is an exact leading-edge contact boundary. If measured marker-center spacing versus clear inner-edge spacing matters, record marker footprint/reference explicitly.

## 9. Crossing and analysis engine

### Input normalization

Preserve original sample IDs, UTC timestamps, coordinates, hMSL, velocity components, hAcc/vAcc/sAcc, fix, satellites, heading/headAcc, units, and provenance. Store absent quality as unknown, never fabricated “1 m accuracy” or six satellites. Retain map from original row → normalized sample ID; filtered DataFrame positions must not be reused as compressed-list indexes.

Current flight normalization retains most values but drops `gpsFix` and `headAcc` from stored compact points. Add them for new imports; legacy records without them remain unknown, not rejected retroactively. Invalid timestamps and duplicate/conflicting times get explicit diagnostics. Keep the raw source private for reproducible reprocessing where storage permits.

### Directed gate crossing

For each candidate segment p0→p1 and gate center c with forward unit normal n, signed distances are d0=(p0−c)·n and d1=(p1−c)·n. A forward crossing goes upstream→downstream. For a nondegenerate crossing:

```text
alpha = -d0 / (d1-d0)
t_cross = t0 + alpha*(t1-t0)
p_cross = p0 + alpha*(p1-p0)
```

Test the intersection against the gate's finite width. Interpolate velocity components and hMSL at that timestamp, then derive speed/heading. Preserve sample IDs, alpha, time, signed lateral offset, approach angle, and uncertainty. Handle exact-on-plane points once, endpoint touches, repeated crossings, backwards travel, collinear travel, and zero-length gates explicitly.

Restrict to a terminal canopy approach, not all aircraft/freefall/post-landing points. Use flight phase/window evidence and chronology; “under 2,000 ft” alone is not sufficient. When there are multiple candidate approaches, ask for a selection rather than choosing the one with the best score. Walk-back must never create another run.

Enforce gate order G1→G5 as applicable. A geometrical G1 crossing can still fail height; retain it as diagnostic entry but mark required eligibility separately. Missing G1 produces no valid course time or distance result. Crossing the infinite line outside the gate is an out-of-span observation, not a successful gate entry.

### Sampling and uncertainty

Initial release uses transparent linear interpolation. Do not silently use splines that can overshoot or invent a boundary pass. For gaps over 0.5 s flag low confidence; over 1.0 s do not auto-classify a crossing as valid. Also inspect travel distance, turn rate, and segment/gate geometry: a short interval at high speed or through a tight carve may still be ambiguous. These are engineering defaults to validate, not competition rules.

Propagate course-placement error along with track error, including heading uncertainty that grows down-course. Keep shared survey bias distinct from independent per-gate errors. Missing source accuracy in a Google Earth-derived KML is not zero error. An uncertainty region overlapping a sideline yields an uncertain containment check. Device location outside a gate does not by itself prove no body part was inside it.

Timing uncertainty should include along-normal position uncertainty divided by normal crossing speed plus interpolation/model error; near-parallel entry is ill-conditioned. Display approximate times with appropriate precision, not competition-grade thousandths from a 4 Hz stream. A measured receiver velocity can be more useful than finite differences, but still carries uncertainty.

### Discipline outputs

| Event | Outputs | Requires independent evidence / limitations |
|---|---|---|
| Speed | G1→G5 elapsed time; each gate's time, speed, heading, lateral offset and height status; entry/exit speed; sector times; centerline/actual-path length; containment checks. | Vertical status depends on calibration/body proxy; path between sparse samples may be uncertain; canopy-down is not known from GNSS. |
| Distance | Entry speed; G5 time/height; pre-entry drag evidence; distance down the course from G1 to observed touchdown/contact; lateral position; separately reported detected landing estimate and trajectory length. | Use along-course projection, not distance flown or final resting position. Earliest/nearest relevant land contact needs observation; landing estimate is not confirmed touchdown. |
| Accuracy | G1–G4 passage; full-interval contact status for G1→G2, G2→G3, and G3→G4; separate G1 contact evidence; provisional points only where the profile defines the award mapping; landing zone polygon(s); complete landing/contact interval; stand-up evidence; raw point subtotal/status. | Cannot use final GPS position alone. Contact at both gate planes does not establish contact throughout the interval. Unobserved touched zones, missing interval evidence, unresolved award mapping, or unresolved pose must remain incomplete. |

The scoring engine consumes checks/evidence after geometry, not inside the parser. Return `eligible_estimate`, `rule_failure`, `insufficient_evidence`, or `not_evaluable`, with reason codes and raw diagnostics. Required unknown checks prevent a definitive success result. Keep score kind/value separate from metric kind/value; no-data is null, not numeric zero. Human/video-confirmed evidence never becomes an unlabeled sensor measurement.

### Gate height versus continuous training checks

The default Distance validity checks are at G1 and G5: Intermediate uses 3.0 m / 1.5 m respectively; Advanced and Open use 1.5 m / 1.5 m. Speed checks all five gates against its class limit. A path above a limit between gates must not automatically become a failed gate.

The optional Distance continuous-height drill evaluates the G1→G5 interval against an explicitly selected ceiling, with its own `training_check` result and label. It is off by default. For Intermediate, selecting a 1.5 m continuous ceiling deliberately creates a stricter drill: an otherwise compliant 2 m G1 entry can fail that drill while retaining its passing 3 m G1 result. Selecting a 3 m continuous ceiling does not relax the separate 1.5 m G5 check. Do not infer a linear taper from 3 m to 1.5 m or call either ceiling an official continuous rule. Record the chosen threshold and profile revision so results are reproducible.

Continuous checks evaluate the modeled path with uncertainty and observability gaps, not every actual unsampled instant. A missing or ambiguous observation yields `unknown`/`uncertain`, not an assumed compliant segment.

### Full-interval Accuracy contact

Use `contact_interval` for Smiley's requested training profile, with start/end gate IDs and crossing timestamps; preserve `leading_edge_plane` as a distinct policy for profiles that require it. Derive interval length from course geometry rather than hardcoding 12 m. Record evidence coverage over the entire interval, including interruptions and obscured/unobserved portions.

- `confirmed_contact`: independent evidence supports uninterrupted contact throughout the interval; retain its source and confidence.
- `confirmed_no_contact`: independent evidence establishes a break anywhere in the interval; interval drag credit is not earned under this profile.
- `unknown`: only endpoint contact, GPS proximity, incomplete video, or otherwise insufficient coverage. No automatic credit or zero from missing evidence.

Keep gate-height passage independent from interval drag credit. No amount of interpolation can certify uninterrupted water contact between GPS samples. Four water-gate point awards cannot be silently mapped onto three intervals: until D7 is specified, show interval compliance and any independently resolved components, but leave the complete water-point subtotal unresolved.

Example result shape:

```json
{
  "discipline": "distance",
  "class": "open",
  "course_revision": "immutable-id",
  "rules_revision": "training-open-v1",
  "engine_version": "course-analysis-v1",
  "entry_source": "g1_intersection",
  "geometry_status": "crossed",
  "height_status": "uncertain",
  "water_contact_status": "unknown",
  "result_status": "insufficient_evidence",
  "score_value": null,
  "reason_codes": ["WATER_CONTACT_UNVERIFIED", "HEIGHT_INTERVAL_OVERLAPS_LIMIT"],
  "raw_metrics": {"entry_speed_mps": 30.0},
  "observations": [],
  "warnings": ["Training estimate; not an official judging result"]
}
```

Values above are illustrative, not an assessment of any uploaded jump.

## 10. Application integration and interfaces

Keep the existing Django pages with Leaflet and current chart components. Add course management as a focused `courses` app (preferred boundary), leaving original flight analysis in `flights`. No need to add a full REST framework or task queue solely for this feature; bounded single-flight analysis can run synchronously initially. Use the existing deployment's job mechanism when available, or add a small explicit worker only for bulk work. Do not use unmanaged web-request threads for long imports.

| Existing file | Planned work |
|---|---|
| `flights/models.py` | Retain legacy gate FK/metrics temporarily; add association to primary course analysis and stable track revision/hash; stop presenting flare-derived speed as measured G1 speed. |
| `flights/flight_manager.py` | Factor shared track decoding into an independent module; retain missing-quality provenance, fix/headAcc, stable sample IDs. Preserve old AGL for legacy views while adding separate calibrated course heights. |
| `flights/utils/gate_parser.py` | Deprecate; compatibility wrapper to new import service only after preview/schema validation. |
| `flights/utils/course_builder.py` | Replace with tested pure geometry templates; no new use of legacy Speed/Accuracy outputs. |
| `flights/utils/gate_calculator.py` | Replace with directed finite crossing evaluator and per-gate event results. |
| `users/views.py` | Route gate upload/assignment/calculation to services; apply object permissions; replace global name overwrite; invalidate stale results. |
| `templates/users/flight_detail.html` | Course/profile selector, primary analysis status, G1-based metrics, evidence/uncertainty panel, map overlays, per-gate table. Preserve original phase plots. |
| `flights/views.py`, `flights/urls.py`, gate map template | Compatibility redirects/read adapters; authorization on every old URL; reuse map capabilities. |
| `flights/admin.py` | Read-only legacy state; new global management with audited permissions and POST-only mutations. |
| `flights/management/commands/calculate_gate_metrics.py` | Explicit legacy mode or adapter to new analysis command; dry-run and scoped batches. |
| `GATE_SYSTEM_README.md`, `GATE_USAGE_GUIDE.md` | Replace misleading numbering/workflow and document imports, uncertainty, privacy, and profile status. |

Suggested new modules: `courses/models.py`, `permissions.py`, `schemas.py`, `imports/kml.py`, `imports/survey.py`, `geometry.py`, `templates_registry.py`, `rules.py`, `services.py`, `analysis/crossings.py`, `analysis/heights.py`, `analysis/disciplines.py`, and `tests/`. Shared track decoder belongs under `flights` or a small neutral module rather than importing the whole FlightManager into survey parsing.

### Endpoint contract (proposed)

| Method / path | Behavior |
|---|---|
| `GET /courses/` | Permission-filtered personal/public list, pagination, map bounds filters. |
| `POST /courses/imports/` | Private source ingest and preview report; return import ID, detected features, warnings, missing inputs. |
| `PATCH /courses/imports/{id}/mapping/` | Owner edits roles, timing windows, generation options; regenerates preview. |
| `POST /courses/imports/{id}/commit/` | Validate and atomically create selected discipline revisions; idempotency token prevents duplicate commits. |
| `GET /courses/{id}/` and `/geometry/` | Permitted metadata and GeoJSON/render data. |
| `POST /courses/{id}/revisions/` | Owner/admin creates new immutable revision; optimistic version check. |
| `PATCH /courses/{id}/visibility/` | Permission-checked public/private change and audit. |
| `POST /courses/{id}/fork/` or `/archive/` | Copy or archive without mutating historical references. |
| `GET /courses/{id}/export.kml` | Export permitted selected revision with Swoopr metadata. |
| `POST /flights/{id}/course-analyses/` | Owner applies exact revisions, calibration and event/class. |
| `GET /flights/{id}/course-analyses/{analysis_id}/` | Authorized result and evidence provenance. |
| `POST /flights/{id}/course-evidence/` | Versioned manual/video contact or calibration annotation. |

Authenticated session + CSRF for writes; 400/422 for invalid input, 403 or non-disclosing 404 for unauthorized objects, 409 for stale revisions, and no parse errors containing sensitive paths. Source files are read through Django storage APIs, not hardcoded `.path`, so remote storage remains possible. Background tasks re-check ownership/visibility and exact input revisions before publishing results.

Result cache key includes track content/parser version, approach-window selection, course geometry hash, rule-profile revision, height calibration, evidence revision, and analysis-engine version. Changes mark old results stale and create new results. Detaching a course cannot leave its speed/height badge visible. Recalculation must not silently rewrite a historical result's inputs.

## 11. Migration and delivery plan

### Phase 0 — Characterize and contain defects

- Add regression tests reproducing current false entry and parser defects.
- Immediately prevent same-name cross-user updates and unscoped access in existing routes; make parsing mutations POST/CSRF protected.
- Mark legacy gate results as legacy/unverified; separate legacy flare speed label from actual gate speed.
- Audit existing CompetitionGate rows: source availability, owner, parse status, coordinate plausibility, assigned flights. No automatic deletion or trusting `is_parsed=True` as validation.

Exit: existing workflows remain usable, known unsafe permissions/fallbacks have tests, and migration inventory is reviewed.

### Phase 1 — Geometry, revisions, KML, management

- Add models, permissions, import preview/commit, schema validation, one/all-three generation, and Leaflet management views.
- Implement explicit and seeded KML with the existing eastern files as compatibility fixtures.
- Implement precise Speed and Distance templates and the visually verified Accuracy polygons above; class-specific scoring remains gated on profile review.
- Add export round-trip and admin global management.

Exit: upload eastern KMLs unchanged, see correctly named gates, create independent private/public courses, generate all three from a confirmed G1 without unintended duplicates or unauthorized access.

### Phase 2 — Course-aware horizontal flight analysis

- Normalize stable samples and quality; implement directed crossings and ordered course traversal.
- Wire assignment/reanalysis, actual G1 entry metrics, Speed times, Distance projection, and uncertainty-aware lateral checks into flight detail.
- Keep legacy turn/flare analysis intact; expose missing entry rather than fallback.

Exit: synthetic crossing ground truth and field map/video comparisons pass; no closest-approach success; historical metrics unchanged unless intentionally relabeled.

### Phase 3 — gSwoop survey import

- Time-based dwell detector, A-repeat validation, axis intersection, optional G5 check, manual mapping, and source-quality preview.
- Synthetic tests at 1/3/4/5/10 Hz, missing samples, unequal dwells, extra stops, unequal cross arms, and both directions.
- Validate at least one real survey, then field-check multiple recordings before labeling placement accuracy.

Exit: repeatable surveyed entry orientation and visible uncertainty; flight CSVs do not masquerade as completed surveys.

### Phase 4 — Height, evidence, and class-specific training results

- Implement surface/calibration/mount models and uncertainty statuses.
- Versioned class profiles and rule checks using the confirmed D1–D3 requirements; resolve remaining class/award/outcome details (D4, D5, D7) before complete scoring, and verify the governing rules before labeling any profile official.
- Implement per-gate Distance height overrides, separate optional continuous-height training results, and full-interval Accuracy contact coverage without inferring contact from GPS.
- Add water-contact/landing/stand-up annotations, provisional scoring, and virtual-water mode.
- Test independent calibration/video cases, including systematic offsets and ambiguous threshold crossings. Do not fit calibration to make a selected jump pass.

Exit: a well-separated height case gets an estimate, a marginal case is uncertain, absent water evidence stays unknown, and independently reviewed observations reproduce expected profile outcomes.

### Phase 5 — Migration, batch use, hardening

- Migrate eligible legacy rows into draft course sets with provenance; legacy source geometry must be revalidated because its construction is flawed.
- Ownerless rows go to admin quarantine, not public by default. Legacy `standard` does not automatically prove both Distance and Accuracy were measured.
- Keep legacy IDs/FKs and old metrics as read-only history. Recompute only selected flights, compare results, and switch feature flags after review.
- Add scoped batch application, monitoring, export QA, performance tuning, documented support workflow, and rollback tests.

Exit: no destructive migration; old and new analyses are distinguishable; rollback can disable new UI/analysis without losing original tracks or revisions.

Suggested PR slices: (1) authorization/regressions, (2) shared decoder and quality schema, (3) revision models and permissions, (4) pure geometry/templates, (5) KML wizard, (6) course management/map/export, (7) crossings and flight integration, (8) survey import, (9) calibration/evidence, (10) rule profiles/scoring, (11) migration and cleanup. Each slice must be independently reviewable and leave the app usable. These are planned changes, not work already performed.

## 12. Acceptance tests

| ID | Scenario | Required result |
|---|---|---|
| A01 | Import existing eastern Speed KML | Five gates, ten endpoints; center/line duplicates reconciled; matching left curve. |
| A02 | Import eastern Distance KML | Two gates at 0/50; no fabricated intermediate gates or landing cutoff at 50. |
| A03 | G1 pair + confirmed forward heading, select all three | Three discipline revisions sharing the entry setup; no duplicate physical entry shift. |
| A04 | Single point without center/side role or heading | Useful missing-input error, not a guessed course. |
| A05 | Heading0/90/180/270 and left/right carve | Correct mirrored local geometry, widths, gate normals, station order. |
| A06 | Explicit nonstandard gates | Coordinates preserved, deviations visible, custom/normalize choice. |
| A07 | KML clampToGround altitude0 | Unknown measured elevation, never sea-level water. |
| A08 | KML point/line disagreement, bad coordinate, XML entity or NetworkLink | Conflict/error; no remote fetching or entity expansion. |
| A09 | A1/B/C/D/A2 at 45 s and 4 Hz | Roles retain time order and repeat-A; no size-based sorting. |
| A10 | Unequal survey arms or dwell durations; optional E | Correct line intersection; G5 residual shown; duration independent of sample count. |
| A11 | Extra stop, repeat-A drift, missing C, long dropout | Review/error; never silently reassign roles. |
| A12 | Synthetic center crossing halfway between samples | Interpolated time/position/speed, successful horizontal span check. |
| A13 | Outside span, reverse crossing, near miss, far-away track | No successful entry or substituted closest approach. |
| A14 | Aircraft crosses first, swoop later, walk-back after | Only selected terminal approach considered; no score-maximizing selection. |
| A15 | Exact-on-line point and multi-segment repeat | Deduplicated deterministic event. |
| A16 | Height interval straddles1.5 or3 m | `uncertain`; no automatic zero/pass. |
| A17 | Missing vAcc, body offset, water reference, or actual contact | Unknown fields propagate; no fabricated accuracy or water points. |
| A18 | Same raw flight, Intermediate versus Open Speed | Same geometry/crossings, different height limit and profile result. |
| A19 | Gate-only Distance validity and optional continuous-height drill | A between-gate excursion can fail the drill without fabricating a gate failure; threshold, mode, and uncertainty are explicit. |
| A20 | No open-distance drag evidence versus confirmed no drag | Unknown result in first case; configured penalty in second; raw metrics preserved. |
| A21 | Accuracy stop ends in CZ after touching lower-value zone | Use complete observed landing contacts, not final GPS position. |
| A22 | Same course name, two users | Independent objects; no overwrite or source access. |
| A23 | Guess private ID through map/API/export/flight assignment | Denied; no hidden JSON/source leak. |
| A24 | Change course/class/calibration or detach | Stale result cleared/recomputed; previous immutable result remains auditable. |
| A25 | Re-import exact file or retry commit | Idempotent or explicit duplicate option; no silent global deduplication by name. |
| A26 | Course revision while another user views old flight | Historical pinned geometry/result remains unchanged. |
| A27 | Legacy ownerless gate and malformed stored config | Admin quarantine; no automatic publication or scoring migration. |
| A28 | Two core decoders normalize same track | Stable sample IDs, units, missing fields, and timestamps agree. |
| A29 | Visibility change, admin moderation, anonymous public flight | Consistent permission/retention rules, audited changes, no private overlays. |
| A30 | Accuracy complete template | All zones non-overlapping, correct point values, exact boundary membership, validated against diagram and class profile. |
| A31 | Intermediate Distance mixed entry/exit limits | With unambiguous height evidence, G1 at 2 m passes its 3 m check; G5 at 2 m fails its 1.5 m check. A passing G5 cannot be obtained by applying the class default. |
| A32 | Intermediate G1 at 2 m, G5 at 1 m; optional continuous ceiling | With sufficient evidence, both gate-height checks pass; a selected 1.5 m continuous drill fails independently. A 3 m drill never overrides the separate G5 limit. |
| A33 | Accuracy contact at both gates but an observed interruption between | Height passage can still pass; the full-interval contact requirement fails and associated drag credit is not earned where its mapping is defined. |
| A34 | Accuracy endpoint-only evidence or obscured interval | Contact remains unknown; no interpolated drag credit, fabricated confirmed break, or definitive zero from missing evidence. |
| A35 | Accuracy interval compliance known but point mapping/G1 policy absent | Show the three interval results; leave complete water points unresolved, with a profile-configuration diagnostic. |
| A36 | Nonstandard imported Accuracy gate spacing | Contact windows use the actual corresponding crossing times and gate geometry, not fixed 12 m segments. |

Suggested pure-geometry regression tolerance: 1 cm for constructed local-coordinate fixtures, not claimed physical accuracy. Existing rounded KML coordinates may use a 5 cm computational compatibility tolerance. Performance target: analyze a normalized 10,000-point flight against one course in under 1 s on the documented test machine; benchmark before declaring this met. Limit parsing work, paginate course lists, and cache revision geometry without bypassing authorization.

### Field validation before trustworthy scoring estimates

Use measured gate locations and independent video/observer labels from ordinary planned training, not special attempts to skim lower. Validate horizontal crossing time against video with synchronization uncertainty recorded. Test survey repeatability on separate walks and retain systematic disagreement rather than averaging it away. Validate height calibration on known static reference heights first, then compare estimates with independently labeled existing flights and body positions.

Use different flights to calibrate and evaluate the height model. Report false-clear, false-VE, unresolved rate, timing residuals, and sensitivity to reference bias. A model that marks everything unknown is not useful, but a model that hides uncertainty to boost coverage is worse. Publication of numerical confidence levels requires these validation data; the current sample archive provides no such ground truth.

## 13. Evidence appendix

### Per-file low/fast screening results

All distances below are meters. hAcc/vAcc are reported estimates. No row is a verified competition-gate assessment.

| File | Records | Low/fast samples | hAcc median | vAcc median | vAcc P95 | dt median / P95 (s) |
|---|---:|---:|---:|---:|---:|---:|
| gps_02376.csv | 1491 | 25 | 0.393 | 0.513 | 0.609 | 0.25 / 0.45 |
| gps_02377.csv | 507 | 23 | 0.539 | 0.660 | 0.675 | 0.25 / 0.25 |
| gps_02378.csv | 477 | 21 | 0.700 | 0.828 | 0.839 | 0.25 / 0.49 |
| gps_02379.csv | 525 | 23 | 0.671 | 0.807 | 0.818 | 0.25 / 0.25 |
| gps_02380.csv | 545 | 23 | 0.762 | 1.037 | 1.129 | 0.25 / 0.50 |
| gps_02381.csv | 530 | 23 | 0.804 | 1.048 | 1.062 | 0.25 / 0.48 |
| gps_02382.csv | 555 | 26 | 0.901 | 1.105 | 1.115 | 0.25 / 0.43 |
| gps_02383.csv | 610 | 22 | 0.450 | 0.736 | 0.755 | 0.25 / 0.50 |
| gps_02384.csv | 1330 | 11 | 0.625 | 0.904 | 0.934 | 0.25 / 0.50 |
| gps_02385.csv | 2383 | 12 | 0.331 | 0.677 | 0.679 | 0.25 / 0.37 |
| gps_02386.csv | 638 | 14 | 0.639 | 0.726 | 0.736 | 0.25 / 0.50 |
| gps_02387.csv | 462 | 23 | 0.689 | 0.900 | 0.992 | 0.25 / 0.48 |
| gps_02388.csv | 451 | 20 | 0.675 | 0.894 | 0.954 | 0.25 / 0.49 |
| gps_02389.csv | 518 | 26 | 0.316 | 0.460 | 0.520 | 0.25 / 0.50 |
| gps_02390.csv | 502 | 22 | 0.722 | 1.016 | 1.045 | 0.25 / 0.48 |

### Reproducibility and source inventory

Track screening used the formulas and masks in §4, with standard CSV/ZIP parsing and NumPy percentiles. It read the archive in place without modifying uploads. Dataset SHA-256 values are recorded below so later code can confirm identical test inputs. Store real flight fixtures privately; do not commit personal GPS tracks to the public repository without separate approval. Use small synthetic fixtures in public tests.

```text
c09c5f14b5657383deea6a6edf2b0ee6e23b17061d321de39ffdcba8859fe763  gps_02376.csv
a2a6dad322c1247e5f07faddffeeb6b069a38ab747094fb2f6d98c665f1e3109  gps_02377.csv
d772a787864134c67145a870e2e045b683755a0b884a5ad599b66876cf25a5c0  gps_02378.csv
deae3ccdaf3b52a5bcaac1ce6c3a9834aefb7327d331859a911098cfb407e448  gps_02379.csv
4ec651b6f96559f7314783728e84896dac44ae6f5881d01f22788b2d2195b9a0  gps_02380.csv
d710ce423870f62d172a0c2f8ca982ee8d30e10a63014672f030cdd1fcd1e334  gps_02381.csv
cc37b8228da447f6197065cef4ed8207f03acccfe8bb5f0092ed35ab10af05e5  gps_02382.csv
c8bac23c35b3bb0f3cd29e354e867478f6a4501bb15638d59b3ffd2bd3f8248e  gps_02383.csv
9e5991f35e69d060f26ae44d2274c9c63ce60de2fa7d1d792a301ccda8e1c55e  gps_02384.csv
fe0667d06ab06194627c0669743f772558a241459b10a6a9e11802ad8abc6fd9  gps_02385.csv
6f51a6991a0aac8501eaf4592cad72790c2a94d0690578c6fcbba81d5f8e2315  gps_02386.csv
162370c0b644e13fda66bab73d503786c668f505dc3d6e4e9d011a79ddc248ee  gps_02387.csv
22683bf75e29de12078673b4fd85d0630e85218546352768fcd495e65678e191  gps_02388.csv
d26ac18606ddf595c29317754696492413d1a3ce2ea1c69bfdb77bff21365ae1  gps_02389.csv
c3e246ba8999f1c658d44b64a33d2ada14cd9f6a60cc0411bf7015175b7bd717  gps_02390.csv
```

Relevant implementation references beyond §2: [FlightManager ingestion/AGL/metrics](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/flight_manager.py), [flight-detail template](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/templates/users/flight_detail.html), [admin](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/flights/admin.py), [current usage guide](https://github.com/wcsmileyk/Swoopr/blob/a6a14f7e2ff63daa0f9640a931a843c013315542/GATE_USAGE_GUIDE.md).

Review boundaries: inspected source at the pinned commit, parsed all supplied tracks, inspected both earlier KMLs and the gSwoop diagram, and ran isolated reproductions of selected pure-Python bugs. Did not run the deployed app, access production data, execute stored ML pickle files, or verify receiver accuracy against surveyed/video ground truth.

Bottom line: this is a tractable staged feature. The first deliverable should be trustworthy course geometry and actual G1-based analysis. Reliable contact scoring requires evidence and validated profiles, not just a more elaborate GPS-height threshold.
