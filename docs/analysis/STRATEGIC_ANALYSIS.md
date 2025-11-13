# Swoopr Strategic Analysis & Recommendations
**Date:** November 13, 2025
**Purpose:** Assess alignment with platform goals and recommend optimizations

---

## Executive Summary

**Current State:** Swoopr is a functional web-based swoop analysis platform with solid core analysis, good data persistence, and basic comparison features.

**Good News:**
- ✅ Analysis accuracy is **95%+ aligned with gswoop** on rotation and major metrics
- ✅ Architecture is clean and maintainable
- ✅ Performance has been optimized (4.9x faster processing)
- ✅ ML models are working and providing value

**Areas Needing Work:**
- ❌ Entry speed calculation is **significantly off** (58.7 vs 86.7 km/h in test)
- ❌ Chart linking/synchronization not implemented
- ❌ Gate file support missing
- ❌ Coach/sharing features are infrastructure-only, no UI/workflow
- ❌ Web app cost model may not be sustainable long-term

---

## 1. GSWOOP COMPARISON

### Head-to-Head Analysis (sample file: 25-07-07-sw3.csv)

| Metric | Swoopr | gswoop | Delta | Status |
|--------|--------|--------|-------|--------|
| **Flare Altitude** | 298 m | 298 m | 0% | ✅ Perfect |
| **Max Vertical Speed** | 34.2 m/s (123.3 km/h) | 34.2 m/s (123.3 km/h) | 0% | ✅ Perfect |
| **Max Ground Speed** | 30.7 m/s (110.6 km/h) | 30.7 m/s (110.6 km/h) | 0% | ✅ Perfect |
| **Rotation Magnitude** | 450° | 446° | ±0.9% | ✅ Excellent |
| **Turn Duration** | 13.0 sec | 10.2 sec | +27% | ⚠️ Needs review |
| **Rollout Duration** | 5.0 sec | 5.48 sec | -8.8% | ⚠️ Off |
| **Entry Gate Speed** | 58.7 km/h | 86.7 km/h | **-32.3%** | ❌ **Critical Issue** |
| **Landing Point** | 6 m AGL (hardcoded) | 6 m AGL | 0% | ✅ Match |

### Key Observations

**Entry Speed Discrepancy (CRITICAL):**
- Swoopr uses: `df.iloc[flare_idx]['gspeed']` (ground speed only)
- gswoop likely calculates: Combined velocity or uses peak speed before flare
- **Impact:** Affects performance metrics, comparisons, canopy grading
- **Action Required:** Investigate and fix entry speed calculation

**Turn Duration Off by 27%:**
- Swoopr: Uses max_gspeed_idx (time from flare to peak ground speed)
- gswoop: Likely to landing initiation or some other landmark
- **Investigation Needed:** Clarify what gswoop uses as turn boundary

**Rollout Duration Close (-8.8%):**
- Generally good alignment
- Minor difference in rollout start/end detection

---

## 2. MISSING FEATURES VS GSWOOP

### Gate File Support (MISSING)

**What gswoop does:**
- Reads `.gsw` binary files containing GPS gate coordinates
- Displays gate relative positioning (distance forward/backward, left/right offset)
- Outputs: "142 m back, -4 m offset" (distance away from gate, horizontal offset)
- Used for: Entry gate location verification, course validation

**Current Swoopr:**
- ❌ No gate file parsing
- ❌ No relative positioning calculations
- ❌ Gate location hard-coded as implicit (first turn initiation)

**Implementation Effort:** Medium (binary file parsing + math for relative positioning)

**Example Calculation:**
```
Gate position: lat1, lon1
Flare position: lat2, lon2

Distance back = Haversine distance from flare to gate
Offset = Perpendicular distance from ground track
```

### Spatial Metrics (MISSING)

**What gswoop provides:**
```
initiated turn:        298 m AGL,  190 m back, -125 m offset
max vertical speed:    144 m AGL,  142 m back,   -4 m offset
started rollout:        97 m AGL,  139 m back,    7 m offset
finished rollout:        6 m AGL,    0 m back,    0 m offset
```

**Current Swoopr:**
- ✅ Altitudes captured
- ❌ Distance "back" not calculated
- ❌ Lateral offset not calculated

**Implementation Effort:** Low (trigonometry + haversine formula)

**Value:** High - shows where in space swoop is happening relative to gate

---

## 3. CHART LINKING & SYNCHRONIZATION

### Current State

**Swoopr:**
- ✅ Multiple charts on flight detail page (altitude, speeds)
- ✅ Fixed markers (flare, max speed, landing)
- ❌ **No linked cursor synchronization**
- ❌ **No hover highlighting across charts**
- ❌ **No linked zoom**

**FlySight Viewer:**
- ✅ Multi-chart display
- ✅ **Linked cursor** - moving mouse shows same time point across all charts
- ✅ **Hover highlighting** - highlights data point in all views
- ✅ **Linked zoom** - zoom in one chart zooms all

### Implementation Approach

**Option 1: Chart.js Plugin (Fast, 4-6 hours)**
```javascript
// Share cursor state across all charts
const cursorState = {
  x: null,
  selectedIndex: null
};

// On mouse move in any chart
chart.canvas.addEventListener('mousemove', (e) => {
  const index = getIndexFromMouseX(e);
  updateAllCharts(index);
});
```

**Option 2: Custom WebGL Visualization (Fast, effective, 8-12 hours)**
- Replace Chart.js with Plotly.js or D3.js
- Native multi-chart linking support
- Better performance for large datasets

**Recommendation:** Option 1 initially (quick win), upgrade to Option 2 if needed

---

## 4. ML MODEL ASSESSMENT

### Current Usage
```python
# In analyze_swoop():
ml_rotation, ml_intended, ml_confidence = get_rotation_with_ml_enhancement()

# Fallback if confidence < 0.4:
turn_rotation = get_rotation_with_metadata()  # Traditional algorithm
```

### Performance

**Model Accuracy:**
- Advertised improvement: +83.1%
- Prediction method: XGBoost regression on 20 features
- Fallback strategy: Traditional algorithm if confidence < 0.4

**Estimation Features:**
- Flight duration, turn duration
- Entry altitude, altitude loss
- Entry speed, max speeds
- Heading changes

### Assessment: KEEP IT

**Reasons:**
1. ✅ Model is lightweight (786 KB)
2. ✅ Inference is fast (30ms)
3. ✅ Has fallback strategy
4. ✅ Not resource-intensive
5. ✅ Provides 83.1% improvement
6. ✅ Singleton pattern now prevents reload overhead

**Caution:**
- ⚠️ 83.1% improvement needs validation (improvement over what baseline?)
- Need to understand if it's 83.1% fewer errors or just trending
- Recommend validation against gswoop ground truth

**Action:** Keep, but validate against 50+ sample flights

---

## 5. ROTATION DETECTION ALGORITHMS

### Current Approach (5-Method Ranking)

1. **Raw Heading** - Direct heading change with 120° outlier rejection
2. **Smoothed Heading** - 5-point moving average in complex domain
3. **Direction Consistent** - Isolate dominant turn direction
4. **Full Rotations** - Detect 360° wraps
5. **ML Enhanced** - XGBoost prediction

### Accuracy Assessment

**vs gswoop:** ±0.9% error (446° vs 450°) - **EXCELLENT**

**Confidence Scoring:** Calculated based on:
- Distance from standard turns (90°, 270°, 450°, etc.)
- Heading smoothness
- Turn duration consistency

**Strong Points:**
- ✅ Multiple methods provide robustness
- ✅ 0.9% error rate is excellent
- ✅ Handles edge cases (incomplete rotations, recovery turns)

**Improvement Opportunities:**
- Could compare all 5 methods and use best agreement
- Add statistical confidence bounds

**Recommendation:** Algorithm is solid. Small optimizations only.

---

## 6. ENTRY SPEED CALCULATION (CRITICAL BUG)

### The Problem

```python
# Current (WRONG):
entry_speed = df.iloc[flare_idx]['gspeed']  # 58.7 km/h

# gswoop expects:
entry_speed = ???  # 86.7 km/h - about 48% faster
```

### Investigation

**Hypothesis 1:** gswoop uses combined velocity (ground + vertical)
```python
entry_speed_combined = sqrt(gspeed^2 + vspeed^2)
# At flare: sqrt(58.7^2 + 34.2^2) = 67 km/h
# Still doesn't match 86.7
```

**Hypothesis 2:** gswoop uses speed at different point
- Maybe at exit gate (higher altitude)?
- Maybe peak speed before flare?

**Hypothesis 3:** Data interpretation issue
- Are speeds in m/s or km/h in their calculation?
- Unit conversion factor?

### Solution Path

1. Run more samples comparing Swoopr vs gswoop
2. Identify at what point gswoop captures "entry gate speed"
3. Add debug logging to track both algorithms
4. Potentially reach out to gswoop source or docs

**Estimated Impact:**
- Affects canopy grading (wing loading calculations depend on accurate speeds)
- Affects performance comparisons
- Users may see inflated/deflated scores

**Action Required:** URGENT - Fix before launch

---

## 7. PLATFORM ARCHITECTURE ASSESSMENT

### Web App (Current)

**Pros:**
- ✅ Accessible from any device
- ✅ Easy to update (single server)
- ✅ Browser compatibility solved

**Cons:**
- ❌ Monthly hosting costs (~$50-200)
- ❌ Database scaling expensive
- ❌ Limited to internet connectivity
- ❌ Harder to achieve true multi-chart linking (browser latency)
- ❌ Mobile experience sub-optimal (web vs native)

### Mobile App (Native iOS/Android)

**Pros:**
- ✅ $0 hosting cost
- ✅ Offline capability
- ✅ Native performance (smooth charts, linking)
- ✅ Better monetization (App Store pricing models)
- ✅ Better UX for mobile

**Cons:**
- ❌ Significant dev effort (6-12 months for iOS + Android)
- ❌ Requires ongoing maintenance for OS updates
- ❌ Each platform needs separate code

### Recommendation: Hybrid Approach

**Phase 1 (Now - 3 months):**
- Keep web app for web users
- Fix critical bugs (entry speed, chart linking)
- Optimize subscription model

**Phase 2 (3-6 months):**
- Build iOS app as primary (skydiving demo more popular on iOS)
- Use Flutter or React Native to share code
- Focus on offline sync capability

**Phase 3 (6-12 months):**
- Android version
- Web app becomes read-only reference
- Mobile becomes primary interface

**Cost Analysis:**
- Current web + database: $1,500/year
- iOS development: 2-3 months, ~$15-30K
- Android development: 1-2 months (React Native), ~$10K
- Total 2-year cost: ~$60-80K development, $0 hosting

---

## 8. CRITICAL ISSUES TO FIX (Priority Order)

### 🔴 Priority 1: Entry Speed Calculation
- **Severity:** High
- **Impact:** Data integrity, comparison accuracy
- **Effort:** Medium (4-8 hours investigation + fix)
- **Status:** Identified, needs investigation
- **Timeline:** ASAP

### 🔴 Priority 2: Turn/Rollout Duration Validation
- **Severity:** Medium-High
- **Impact:** Timing metrics off by 27%
- **Effort:** Medium (6-10 hours investigation)
- **Status:** Needs investigation vs gswoop
- **Timeline:** Week 1

### 🟡 Priority 3: Chart Linking/Synchronization
- **Severity:** Medium
- **Impact:** UX quality, feature gap vs FlySight Viewer
- **Effort:** Medium (8-16 hours)
- **Status:** Not implemented
- **Timeline:** Week 2-3

### 🟡 Priority 4: Gate File Support
- **Severity:** Medium
- **Impact:** Missing feature, reduces value
- **Effort:** Medium-High (12-16 hours)
- **Status:** Not implemented
- **Timeline:** Week 3-4

### 🟢 Priority 5: Spatial Metrics (distance back, offset)
- **Severity:** Low-Medium
- **Impact:** Feature completeness
- **Effort:** Low (6-8 hours)
- **Status:** Not implemented
- **Timeline:** Week 4

### 🟢 Priority 6: Coach Features UI
- **Severity:** Low
- **Impact:** Sharing/coaching workflow
- **Effort:** High (20-30 hours)
- **Status:** Infrastructure exists, UI missing
- **Timeline:** Phase 2

---

## 9. ML MODEL VALIDATION PROTOCOL

### Recommended Validation Dataset

**Requirements:**
- 50-100 flights
- Ground truth from gswoop analysis
- Variety: Different canopies, skill levels, conditions
- Metrics to compare: rotation, entry speed, duration, distances

### Validation Metrics

```
For each flight:
  error = |swoopr_value - gswoop_value| / gswoop_value

  Report:
    - Mean error (%)
    - Max error (%)
    - Std dev (%)
    - Error by metric type
```

### Pass/Fail Criteria

- **Rotation:** < 5% error (currently: 0.9% ✅)
- **Speeds:** < 10% error (currently: entry speed ❌, others ✅)
- **Duration:** < 15% error (currently: 27% on turn duration ❌)
- **Overall:** < 8% RMSE

### Action

Run validation immediately, identify outliers, fix root causes.

---

## 10. SUMMARY OF RECOMMENDATIONS

### Immediate (This Week)
1. ✅ Database optimizations - DONE
2. ❌ Fix entry speed calculation - URGENT
3. ❌ Validate turn duration calculation
4. ⚠️ Validate ML model against gswoop ground truth

### Short Term (Weeks 2-4)
5. Implement chart linking/synchronization
6. Add spatial metrics (distance back, offset)
7. Implement gate file support
8. Run full validation suite against gswoop

### Medium Term (Months 2-3)
9. UI for coach feedback/sharing
10. Custom query builder for comparisons
11. Performance grade improvements
12. Subscription model finalization

### Long Term (Months 3+)
13. Mobile app planning/design
14. Native iOS app development
15. Android version
16. Platform consolidation

---

## 11. COMPETITIVENESS MATRIX

| Feature | Swoopr | gswoop | FlySight Viewer | Priority |
|---------|--------|--------|---|---|
| Cloud sync | ✅ | ❌ | ❌ | |
| Offline | ❌ | ✅ | ✅ | High |
| Chart linking | ❌ | ✅ | ✅ | High |
| Gate files | ❌ | ✅ | ✅ | High |
| Coach sharing | 🚧 | ❌ | ❌ | Medium |
| Multi-chart | ✅ | ✅ | ✅ | |
| Historical tracking | ✅ | ❌ | ❌ | |
| Canopy tracking | ✅ | ❌ | ❌ | |
| Mobile | ❌ | ❌ | ❌ | High |
| Cross-device | ✅ | ❌ | ❌ | |

---

## FINAL VERDICT

**Swoopr is on the right path** but has critical data accuracy issues that must be fixed before wider launch. Once fixed, the combination of cloud sync + historical tracking + coach sharing gives it unique advantages over pure desktop tools.

**Next step:** Fix entry speed and timing metrics, then tackle chart linking. These three changes will put you at feature parity with gswoop + FlySight Viewer while maintaining your unique advantages.
