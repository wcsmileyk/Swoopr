# Swoopr Implementation Roadmap
## Priority-Based Action Plan (Next 8 Weeks)

---

## CRITICAL (Fix Before Public Launch)

### Week 1: Flare Detection Fix

**Issue:** Flare detection fires 6-8 seconds too late, causing all timing metrics to be wrong

**Deliverables:**
- [ ] Investigate exact signal (vspeed change vs gspeed change vs altitude rate)
- [ ] Implement new flare detection algorithm (detect slowdown onset, not maximum)
- [ ] Validate against 10+ test flights vs gswoop
- [ ] Verify rotation still accurate (should be unaffected)

**Files to modify:**
- `flights/flight_manager.py` - `find_flare()` method

**Effort:** 4-6 hours
**Status:** Identified, needs investigation

**Expected Impact:**
- Entry speed: 58.7 → ~86.7 km/h (48% improvement)
- Turn duration: 13.0 → ~10.2 sec (21% improvement)
- Data parity with gswoop achieved

---

### Week 1-2: Rollout Duration Validation

**Issue:** Rollout duration off by 8-10%, needs investigation

**Deliverables:**
- [ ] Compare rollout detection algorithm with gswoop method
- [ ] Identify if it's rollout_start or rollout_end that's off
- [ ] Fix whichever point is misaligned
- [ ] Validate against test fleet

**Files to modify:**
- `flights/flight_manager.py` - `get_roll_out()` method

**Effort:** 4-6 hours
**Status:** Needs investigation

---

## HIGH PRIORITY (Complete by Week 3)

### Week 2-3: Chart Linking Implementation

**Issue:** Multi-chart cursor linking not implemented (feature gap vs FlySight Viewer)

**Deliverables:**
- [ ] Implement shared cursor state across all charts
- [ ] Add mouse move listeners to Chart.js canvases
- [ ] Update all charts to highlight same time point
- [ ] Add linked zoom (optional)

**Files to create/modify:**
- `flights/templates/flights/flight_detail.html` - Chart section
- `flights/static/js/chart_linking.js` - New file

**Effort:** 8-12 hours
**Status:** Not implemented

**Expected Impact:**
- Improves UX significantly
- Achieves feature parity with FlySight Viewer on charting
- Makes multi-chart analysis much more usable

**Code Skeleton:**
```javascript
// New file: static/js/chart_linking.js

class ChartLinker {
  constructor(charts) {
    this.charts = charts;
    this.currentIndex = null;
    this.attachListeners();
  }

  attachListeners() {
    this.charts.forEach(chart => {
      chart.canvas.addEventListener('mousemove', (e) => {
        const index = this.getIndexFromEvent(e, chart);
        this.updateAllCharts(index);
      });
    });
  }

  updateAllCharts(index) {
    this.currentIndex = index;
    this.charts.forEach(chart => {
      // Highlight data point at index
      // Show vertical line at x position
    });
  }
}
```

---

### Week 3: Gate File Support Implementation

**Issue:** Missing .gsw gate file support (feature gap vs gswoop)

**Deliverables:**
- [ ] Implement .gsw binary file parser
- [ ] Extract gate GPS coordinates from file
- [ ] Calculate distance-back and offset metrics
- [ ] Display in flight analysis

**Files to create/modify:**
- `flights/gate_parser.py` - Binary file parsing (may already exist)
- `flights/flight_manager.py` - Add spatial metrics calculation
- `flights/models.py` - Add gate-related fields

**Effort:** 12-16 hours
**Status:** Not implemented

**Expected Impact:**
- Feature parity with gswoop on gate support
- Enables course validation
- Adds spatial metrics to analysis

**Formula for Distance Back & Offset:**
```python
from math import radians, sin, cos, asin, sqrt

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def calculate_spatial_metrics(flare_lat, flare_lon, flare_heading,
                             gate_lat, gate_lon):
    """Calculate distance back and offset from gate"""

    # Distance from flare to gate
    distance_back = haversine(flare_lat, flare_lon, gate_lat, gate_lon)

    # Offset (perpendicular distance)
    # This requires converting to flat coordinates relative to flare heading
    # ... trigonometry here ...

    return distance_back, offset
```

---

## MEDIUM PRIORITY (Complete by Week 5)

### Week 4: ML Model Validation

**Issue:** ML model claims 83.1% improvement, needs validation

**Deliverables:**
- [ ] Prepare validation dataset (50-100 flights)
- [ ] Compare ML predictions vs gswoop ground truth
- [ ] Calculate error metrics (MAE, RMSE, %error)
- [ ] Identify any systematic biases
- [ ] Document findings

**Files to create:**
- `ml_validation.py` - Validation script

**Effort:** 6-8 hours
**Status:** Analysis ready

**Success Criteria:**
- Rotation error < 5% (currently: 0.9% ✅)
- Overall RMSE < 8%

---

### Week 4-5: Spatial Metrics Addition

**Issue:** Missing "distance back" and "offset" metrics shown by gswoop

**Deliverables:**
- [ ] Add spatial metrics calculation for all detected points
- [ ] Store in Flight model
- [ ] Display in flight analysis
- [ ] Update database schema

**Files to modify:**
- `flights/models.py` - Add fields
- `flights/flight_manager.py` - Calculate metrics
- Migrations

**Effort:** 6-8 hours
**Status:** Blocked on gate file support

---

### Week 5: Coach Features UI/UX

**Issue:** Coach infrastructure exists but no UI/workflow implemented

**Deliverables:**
- [ ] Design coach feedback interface
- [ ] Create flight sharing UI (invite coaches)
- [ ] Implement coach view/comment section
- [ ] Add notification system for shared flights

**Files to create/modify:**
- `flights/templates/` - New templates for sharing
- `flights/views.py` - Coach views
- `flights/models.py` - Add comment/feedback models

**Effort:** 20-30 hours
**Status:** Planning phase

---

## OPTIMIZATION (Complete by Week 8)

### Week 6: Algorithm Fine-Tuning

**Issue:** Some metrics off by > 5%, need refinement

**Deliverables:**
- [ ] Analyze edge cases (partial rotations, recovery turns, etc.)
- [ ] Adjust confidence scoring algorithm
- [ ] Test against diverse flight types

**Effort:** 8-12 hours

---

### Week 7: Performance & UX Polish

**Deliverables:**
- [ ] Chart rendering optimization
- [ ] Mobile UX improvements
- [ ] Loading state improvements
- [ ] Error message clarity

**Effort:** 8-10 hours

---

### Week 8: Documentation & Launch Prep

**Deliverables:**
- [ ] API documentation
- [ ] User guide
- [ ] Feature comparison matrix
- [ ] Coach user guide

**Effort:** 6-8 hours

---

## Timeline Summary

```
Week 1  │ Flare fix, Rollout validation
Week 2  │ Continue chart linking
Week 3  │ Finish chart linking, Gate files start
Week 4  │ Gate files, ML validation
Week 5  │ Spatial metrics, Coach UI starts
Week 6  │ Algorithm tuning
Week 7  │ Polish & optimization
Week 8  │ Documentation, Launch prep
        └─────────────────────────────────
Total: 8 weeks → Production ready
```

---

## Testing at Each Stage

### Week 1 (After Flare Fix)
```
Compare 10 test flights:
✅ Entry speed within ±10% of gswoop
✅ Turn duration within ±5% of gswoop
✅ Rotation still ±1% of gswoop
```

### Week 3 (After Chart Linking)
```
Manual testing:
✅ Mouse move updates all charts
✅ Charts highlight same time point
✅ Performance acceptable (no lag)
```

### Week 4 (After Gate Files)
```
With example .gsw files:
✅ Parse file successfully
✅ Calculate spatial metrics correctly
✅ Display in UI matches gswoop
```

### Week 5 (Final Validation)
```
Full integration test:
✅ All metrics within ±5% of gswoop
✅ Charts link smoothly
✅ Gate overlay works
✅ Coach sharing UI functional
```

---

## Resource Estimates

| Phase | Backend | Frontend | Testing | Total |
|-------|---------|----------|---------|-------|
| Critical (Wk 1-2) | 8-12 hrs | 0 hrs | 2 hrs | 10-14 hrs |
| High (Wk 2-3) | 4 hrs | 8-12 hrs | 2 hrs | 14-18 hrs |
| Medium (Wk 4-5) | 12-16 hrs | 6-8 hrs | 3 hrs | 21-27 hrs |
| Polish (Wk 6-8) | 12 hrs | 8-10 hrs | 4 hrs | 24-26 hrs |
| **TOTAL** | **36-44 hrs** | **22-30 hrs** | **11 hrs** | **69-85 hrs** |

**Equivalent to:** 2-2.5 weeks of full-time development (or 8-10 weeks part-time)

---

## Success Criteria for Launch

- [ ] All metrics within ±5% of gswoop
- [ ] Chart linking working smoothly
- [ ] Gate file support functional
- [ ] ML model validated
- [ ] Coach sharing implemented
- [ ] No critical bugs in validation suite
- [ ] Performance acceptable on slow networks
- [ ] Mobile UX acceptable

---

## Post-Launch Roadmap (Months 3+)

### Phase 2: Mobile App
- iOS app development (Flutter/React Native)
- Offline sync capability
- App Store deployment

### Phase 3: Advanced Features
- Custom query builder
- Advanced analytics dashboard
- Video integration (GoPro, etc.)
- Wearable data integration

### Phase 4: Community
- Flight sharing social features
- Leaderboards (opt-in)
- Coach marketplace

---

## Decision Points

**Decision 1: Mobile App Priority?**
- iPhone first (skydiving market)
- Android second
- Or web-only for now?

**Decision 2: Subscription Model?**
- Basic (free): 50 flights, community features
- Pro ($5/mo): Unlimited flights, coach access, analytics
- Team ($20/mo): Multiple users, team analytics

**Decision 3: Server Hosting?**
- Current: Render ($50-200/month)
- Upgrade database or scale horizontally?
- Switch to AWS/GCP for more control?

---

## Risks & Mitigation

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Flare fix breaks rotation | High | Extensive validation before merge |
| Chart linking causes perf issues | Medium | Test with large datasets (10K+ points) |
| Gate file format issues | Medium | Get .gsw spec or reverse engineer |
| ML model not generalizable | Medium | Validate on diverse flights |
| Database scaling | Medium | Horizontal scaling ready from start |

---

## Summary

**Bottom Line:** Follow this roadmap strictly, and Swoopr will be production-ready in 8 weeks with feature parity or superiority to both gswoop and FlySight Viewer, while maintaining unique advantages (cloud sync, historical tracking, coach sharing).

The critical flare detection fix is the blocker—everything else can proceed in parallel.
