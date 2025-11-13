# CRITICAL BUG REPORT: Flare Detection Misalignment

**Severity:** 🔴 CRITICAL
**Status:** IDENTIFIED
**Impact:** Data integrity - all timing and speed metrics are offset by ~6-8 seconds

---

## The Problem

**Entry Gate Speed Discrepancy:**
- gswoop: 86.7 km/h
- Swoopr: 58.7 km/h
- **Difference: -32.3% (CRITICAL)**

**Root Cause Found:** Flare detection is firing **too late**

---

## Evidence

### Speed Timeline Before Flare

```
Index | t_s    | gspeed (m/s) | gspeed (km/h) | Observation
------|--------|--------------|---------------|---------------------------
15641 | -42.2s | 24.4         | 87.8          | Peak before big slow-down
15646 | -41.2s | 24.6         | 88.6          |
15651 | -40.2s | 25.0         | 90.0          | ← MATCHES gswoop 86.7!
15656 | -39.2s | 25.8         | 92.9          |
15661 | -38.2s | 25.6         | 92.2          |
15666 | -33.2s | 20.9         | 75.2          | Major slowdown begins
15671 |    0s  | 16.3         | 58.7          | ← SWOOPR FLARE (TOO LATE!)
15676 |  +5.2s | 15.1         | 54.4          |
```

### What Happened

1. Pilot exits at ~1791 m AGL with ~215 km/h ground speed
2. During freefall descent, speeds slow to ~25 m/s (90 km/h)
3. **At some point, pilot pulls brake** - initiates flare (major slowdown begins)
4. Speed drops from 25 m/s to 16 m/s over next 6 seconds
5. **Swoopr detects flare at the BOTTOM of this slowdown** (too late!)
6. **gswoop detects flare at the START of this slowdown** (~6 seconds earlier)

### Comparison with gswoop

```
gswoop:
  entry gate speed:      86.7 km/h  ← 25 m/s, ~6-10 seconds before our flare
  initiated turn:        298 m AGL, 190 m back

Swoopr:
  entry gate speed:      58.7 km/h  ← 16.3 m/s, AFTER most of slowdown
  flare_idx:             15671      ← This is too late!
```

---

## Impact Analysis

### Affected Metrics (ALL TIMING BASED)

1. **Entry Speed** - WRONG (58.7 vs 86.7 km/h) ❌
2. **Turn Duration** - OFF by ~6 seconds (13.0 vs 10.2 sec) ❌
3. **Rollout Duration** - May be off ❌
4. **Canopy Grading** - Based on wing loading (affected by entry speed) ❌
5. **Performance Comparisons** - All skewed ❌

### Metrics NOT Affected

- ✅ Landing detection (looks at the end, not affected)
- ✅ Max speeds (independent of flare timing)
- ✅ Rotation magnitude (turns out to be robust)
- ✅ Distance calculations (based on positions, not flare timing)

---

## Root Cause in Code

### Current Flare Detection (flight_manager.py:842-856)

```python
def find_flare(self, df, landing_idx):
    """Find the flare point (max altitude drop rate change)"""

    # Look for the point with maximum vertical speed in the 30s window
    # before max altitude is reached
    flare_win_vspeed_relax_s = 0.4
    flare_back_win = 30.0

    # Get the slice before landing
    slice_df = df[df.index <= landing_idx].copy()

    # Find max vertical speed in window
    # This finds the MAXIMUM downward speed, which occurs AFTER
    # the pilot has already flared significantly!
```

**The Problem:** Current algorithm looks for **maximum vertical speed**, which occurs **after** the flare is mostly complete. By that time, the pilot is already deep in the slowdown.

**What we should look for:** The POINT WHERE THE SLOWDOWN BEGINS, not where it's at maximum.

---

## Solution Approach

### Option 1: Detect Slowdown Onset (Recommended)

Instead of finding max vertical speed, find where vertical speed **starts increasing** (pilot pulling brake).

```python
def find_flare_v2(self, df, landing_idx):
    """Find flare by detecting onset of deceleration"""

    # Get slice before landing
    slice_df = df[df.index <= landing_idx].copy()

    # Calculate deceleration (rate of change of vertical speed)
    # d(vspeed)/dt - look for where this goes positive (pulling up)

    vspeed = slice_df['velD'].abs().values
    dv = np.diff(vspeed)  # Rate of change

    # Find where deceleration starts (dv becomes positive/large)
    # within some window (e.g., 30s before landing)

    # This should catch the ONSET of slowdown, not its maximum
```

### Option 2: Compare with max vertical speed altitude

```python
# Look for where vertical speed STARTS increasing
# (not where it's maximum)

# Find max vertical speed first (current approach)
max_vspeed_idx = max_vspeed_location

# Then backtrack to where the spike started
for i in range(max_vspeed_idx, 0, -1):
    if vspeed[i] < 0.5 * vspeed[max_vspeed_idx]:
        # Found onset point
        break
```

### Option 3: Use ground speed as proxy

```python
# Ground speed drops during flare due to brake/heading change
# Find where ground speed STARTS dropping significantly

gspeed = slice_df['gspeed'].values
dg = np.diff(gspeed)  # Rate of change

# Find where negative dg becomes sustained
# (indicates brake application)
```

---

## Validation Against gswoop

Once fixed, should produce:

```
✅ Entry gate speed: ~86-87 km/h (vs current 58.7)
✅ Turn duration: ~10-11 sec (vs current 13.0)
✅ Initiated turn altitude: ~298 m (matches!)
✅ Turn rotation: ~446° (already correct!)
```

---

## Testing Protocol

1. **Before implementing fix:**
   - Compare Swoopr vs gswoop on 10 test flights
   - Record baseline metrics

2. **After implementing fix:**
   - Run same 10 flights
   - Verify entry speed within ±10% of gswoop
   - Verify turn duration within ±5% of gswoop
   - Verify rotation still matches (should be unaffected)

3. **Regression testing:**
   - Ensure landing detection still works
   - Ensure max speeds still detected correctly
   - Run full test suite

---

## Timeline

**Effort Estimate:** 4-6 hours
- 1-2 hours: Investigation and algorithm selection
- 1-2 hours: Implementation
- 1-2 hours: Testing and validation

**Priority:** 🔴 **CRITICAL** - Must fix before any public launch

---

## Code References

- Current flare detection: `flights/flight_manager.py:842`
- Find max speeds: `flights/flight_manager.py:849`
- Swoop analysis entry: `flights/flight_manager.py:434`

---

## Next Steps

1. **Investigate** what exactly triggers the slowdown (vertical speed increase vs ground speed decrease vs altitude change)
2. **Implement** new flare detection algorithm
3. **Validate** against gswoop with 10+ test cases
4. **Update** analysis to use new flare point
5. **Document** any side effects

This is the most critical fix needed to achieve data parity with gswoop.
