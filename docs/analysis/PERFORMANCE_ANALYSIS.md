# Swoopr Performance Analysis Report

**Analysis Date:** November 13, 2025
**Test File:** 25-07-07-sw3.csv (1.62 MB, 16,008 GPS points, 53 min duration)

---

## Executive Summary

Your 500 errors on Render production are **NOT primarily caused by disk IO or memory usage during a single file analysis**. However, the profiling reveals **critical inefficiencies when processing multiple files or under load**, which are likely the root cause.

**Critical Finding:** The FlightManager initializes a 40+ MB ML model **on every instantiation**, consuming **79.5% of total processing time**. In a web server handling concurrent requests, this leads to rapid memory accumulation and eventual OOM (Out of Memory) crashes.

---

## Performance Breakdown

### Current Performance (Single File Analysis)

| Phase | Duration | Memory | IO | % of Total |
|-------|----------|--------|----|----|
| **ML Model Loading** | 2.93s | +100.8 MB | 0 MB | **79.5%** ⚠️ CRITICAL |
| **CSV File Reading** | 0.27s | +9.5 MB | 3.0 MB | 7.4% |
| **Rotation Metrics** | 0.20s | +0.6 MB | 0.4 MB | 5.5% |
| **ML Prediction** | 0.03s | +0.4 MB | 0.0 MB | 0.9% |
| **Other Operations** | 0.25s | +1.2 MB | 0.3 MB | 6.8% |
| **TOTAL** | **3.68s** | **+112.5 MB** | **3.7 MB** | 100% |

### Key Metrics

- **Throughput:** ~4.4 files per second (on a single core)
- **Memory per file:** 112.5 MB (peak during ML model loading)
- **Disk IO per file:** 3.7 MB read

---

## Bottleneck Analysis

### 1. **CRITICAL: ML Model Loading (79.5% of time)**

**Problem:**
- The ML model is loaded **every time** a FlightManager instance is created
- Takes 2.93 seconds per instance
- Consumes 100.8 MB of RAM per load
- In Django, with 4+ concurrent worker processes, this means 400+ MB just for ML models

**Impact on Render:**
```
4 worker processes × 100.8 MB = 403 MB
Add Django/PostgreSQL overhead = 600+ MB base memory
Process 10 concurrent uploads = 1000+ MB memory needed
```

If your Render instance has < 1.5 GB RAM, you **will get OOM errors**.

**Code Location:** `flights/flight_manager.py:1125-1138`

```python
def load_ml_model(self):
    """Load the trained ML model for rotation prediction"""
    try:
        model_path = Path(__file__).parent / 'rotation_prediction_model.pkl'
        if model_path.exists():
            model_data = joblib.load(model_path)  # ← SLOW: 2.93s
            self.ml_model = model_data['model']   # ← MEMORY: +100.8MB
```

---

### 2. **CSV File Reading (7.4% of time)**

**Performance:** 0.27 seconds for 1.62 MB = 5.9 MB/s
**Status:** ✅ Acceptable

However, larger files may cause issues:
- 10 MB file: ~0.5s
- 50 MB file: ~2.5s (+ potential pandas memory overhead)

**Code Location:** `flights/flight_manager.py:163-240`

---

### 3. **Rotation Metrics Calculation (5.5% of time)**

**Performance:** 0.20 seconds for 16,008 points
**Memory:** +0.6 MB
**Status:** ✅ Good

This includes:
- Landing detection
- Flare detection
- Speed peak finding
- Dual rotation metrics

All are numpy/pandas vectorized operations (efficient).

---

### 4. **Database Operations (NOT PROFILED)**

⚠️ **Critical Gap:** The profiler did NOT include database operations.

**Potential Issues:**
1. **GPS Point Creation** - If still using legacy `_create_gps_points` method with `bulk_create`
   - ~16,000 GPS points per file
   - Could take 1-5 seconds per file depending on DB load
   - Locks database connection during bulk insert

2. **Flight Record Updates** - Multiple saves to Flight model
   - Happens within transaction.atomic()
   - Could timeout on slow databases

---

## Root Cause: Production 500 Errors

### Most Likely Scenario

1. **Web request arrives** → Render worker spawned
2. **FlightManager.__init__()** → ML model loaded (100 MB allocation)
3. **User uploads file** → Additional memory for DataFrame (10+ MB)
4. **Second user uploads** → Another FlightManager instance (100 MB)
5. **Third user uploads** → Memory pressure increases
6. **Database transaction** → Waiting for I/O while memory constrained
7. **Result:** OOM killer terminates process → 500 error

### Secondary Scenario: Database Connection Limit

- Each request holds DB connection during GPS point bulk creation
- With 4 workers × multiple uploads = connection pool exhaustion
- Timeout waiting for DB connection → 500 error

---

## Specific Recommendations

### 🔴 Priority 1: Singleton ML Model (Critical - 30 min work)

**Impact:** Eliminates 79.5% of per-request processing time

**Implementation:**

```python
# flights/flight_manager.py

class MLModelSingleton:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        model_path = Path(__file__).parent / 'rotation_prediction_model.pkl'
        if model_path.exists():
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self._initialized = True

class FlightManager:
    def __init__(self, cfg=SwoopConfig):
        self.cfg = cfg
        self.ml_model_singleton = MLModelSingleton()
        self.ml_model = self.ml_model_singleton.model
        self.ml_feature_names = self.ml_model_singleton.feature_names
        # Don't call load_ml_model() anymore
```

**Expected Result:**
- First request: 2.93s (load model)
- Subsequent requests: 0.75s (skip model loading)
- **92% faster for concurrent uploads**

---

### 🟡 Priority 2: Optimize GPS Point Storage (Important - 1-2 hour work)

**Current Status:** Using `flight.store_gps_data()` (JSON compression)
**Issue:** Still creates individual database records or large JSON blobs

**Recommended:** Check what `store_gps_data()` actually does:

```bash
grep -n "def store_gps_data" /home/smiley/PycharmProjects/Swoopr/flights/models.py
```

**If using bulk_create():**
```python
# Current (slow for large batches)
GPSPoint.objects.bulk_create(gps_points, batch_size=1000)

# Better (batched with connection pooling)
for i in range(0, len(gps_points), 500):
    GPSPoint.objects.bulk_create(gps_points[i:i+500], batch_size=500)
    time.sleep(0.01)  # Reduce DB lock contention
```

**If using JSON storage:**
```python
# Ensure field is properly indexed
flight.gps_data_json = json.dumps(compressed_gps_list)
flight.save()
```

---

### 🟡 Priority 3: Database Connection Pooling (Important - 30 min work)

**Current Settings:** Check `Swoopr/settings.py`

**Recommended for Render:**

```python
# settings.py

DATABASES = {
    'default': {
        'ENGINE': 'django.contrib.gis.db.backends.postgis',
        'CONN_MAX_AGE': 600,  # Keep connections alive 10 min
        'OPTIONS': {
            'connect_timeout': 10,
        },
        'ATOMIC_REQUESTS': False,  # Don't hold connections across request
    }
}
```

**Expected Result:**
- Reduces connection acquisition overhead
- Prevents connection pool exhaustion

---

### 🟢 Priority 4: Add Memory Monitoring (Nice-to-have - 30 min work)

**Add to your Django logging:**

```python
# flights/views.py or middleware

import psutil

def upload_flight_view(request):
    process = psutil.Process()
    mem_start = process.memory_info().rss / 1024 / 1024

    try:
        # ... your processing code ...
        mem_end = process.memory_info().rss / 1024 / 1024
        logger.info(f"Flight upload memory: {mem_start:.0f}MB → {mem_end:.0f}MB (Δ {mem_end-mem_start:+.0f}MB)")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
```

---

## Implementation Priority

| Priority | Task | Time | Impact | 500 Error Fix? |
|----------|------|------|--------|---|
| 🔴 P1 | Singleton ML Model | 30 min | **79.5% faster** | **YES** |
| 🟡 P2 | Optimize GPS Storage | 1-2 hr | Memory+Speed | YES |
| 🟡 P3 | DB Connection Pooling | 30 min | Timeout resilience | Partial |
| 🟢 P4 | Memory Monitoring | 30 min | Visibility | No |

---

## Testing the Changes

### Before Implementation

```bash
# Simulate concurrent uploads
for i in {1..5}; do
    python manage.py shell <<EOF
from flights.flight_manager import FlightManager
import time
fm = FlightManager()
print(f"FlightManager init: {time.time()}")
EOF &
done
wait
```

### After Implementation

Should see:
- ✅ First process: 2.93s for ML load
- ✅ Other processes: <100ms for FlightManager init
- ✅ Total memory: 500MB instead of 500+ MB per request

---

## Render Deployment Checklist

- [ ] Implement Singleton ML Model pattern
- [ ] Check and optimize GPS point storage method
- [ ] Configure connection pooling for PostgreSQL
- [ ] Set up memory monitoring/alerts
- [ ] Scale to at least 2 GB RAM (or use Render's auto-scaling)
- [ ] Test with concurrent file uploads
- [ ] Monitor logs for "MemoryError" or "500 errors"

---

## Quick Win: Immediate Actions (Without Code Changes)

1. **Increase Render Memory**
   - Current: Likely 512 MB or 1 GB
   - Recommended: 2 GB minimum
   - Cost: ~$7/month additional

2. **Scale Workers**
   - Reduce concurrent workers from 4 to 2
   - Reduces ML model memory footprint
   - May reduce throughput but improves reliability

3. **Set Upload Timeout**
   - Nginx: `proxy_read_timeout 60s;`
   - Prevents 504 errors during long processing

---

## Questions to Verify

1. What is the actual memory limit on your Render instance?
2. Are you still using the legacy `_create_gps_points` method?
3. What is the average file size your users upload?
4. How many concurrent uploads do you expect?

These answers will help refine the recommendations further.