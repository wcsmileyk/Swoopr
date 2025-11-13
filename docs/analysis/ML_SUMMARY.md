# ML System Summary - Quick Reference

## What The System Does (In Plain English)

Your ML system is a **rotation angle predictor**. When someone uploads a skydive video:

1. Extract 11 numbers from the flight (altitude, speed, duration, heading changes, etc.)
2. Feed these into a trained Random Forest model (100 decision trees)
3. Model predicts rotation angle in degrees
4. If confident (>0.4), use ML result; otherwise fall back to traditional algorithm

## The 3 Parts

### Part 1: Training (Offline, already done)
```
Your old FlySight files → Run through gswoop (ground truth) → Extract features → Train ML model
```
- Ground truth source: gswoop (industry standard)
- Training data: ~100 example flights
- Output: `rotation_prediction_model.pkl` (786 KB)

### Part 2: Model (Pre-trained, loaded at startup)
```
Type: RandomForestRegressor (100 trees)
Features: 11 numerical values from flight data
Output: Predicted rotation angle + confidence score (0.0-1.0)
Speed: 30ms per prediction
```

### Part 3: Prediction (At runtime when users upload)
```
New flight uploaded → Extract 11 features → Get prediction → Compare confidence → Decide
```

## Key Files

| File | Purpose |
|------|---------|
| `ml_training_pipeline.py` | Generates training data from gswoop + Swoopr |
| `multi_metric_ml_pipeline.py` | Trains the actual Random Forest model |
| `flights/flight_manager.py` (lines 43-78) | Singleton pattern (loads model once) |
| `flights/flight_manager.py` (lines 1164-1231) | Feature extraction + prediction |
| `flights/rotation_prediction_model.pkl` | The trained model file (786 KB) |
| `ml_training_data.csv` | Training examples (100 rows, 30+ columns) |
| `ml_validation_suite.py` | Script to validate model (YOU CAN RUN THIS!) |

## The 11 Features

What numbers feed into the model:

1. **flight_duration** - Total flight time (seconds)
2. **turn_duration** - Time from flare to max ground speed (seconds)
3. **entry_altitude** - Altitude at flare (feet)
4. **max_gspeed_altitude** - Altitude at peak ground speed (feet)
5. **altitude_loss** - Total altitude lost during turn (feet)
6. **entry_speed** - Ground speed at flare (mph)
7. **max_vspeed** - Maximum downward speed (mph)
8. **max_gspeed** - Maximum forward speed (mph)
9. **heading_start** - Compass heading at start (degrees 0-360)
10. **heading_end** - Compass heading at end (degrees 0-360)
11. **net_heading_change** - Total heading change (degrees, ±360)

## How To Validate (YOU SHOULD DO THIS!)

Your files are valuable for validation. Run this:

```bash
cd /home/smiley/PycharmProjects/Swoopr

# Quick test (first 10 files)
python ml_validation_suite.py 10

# Medium test (50 files)
python ml_validation_suite.py 50

# Full validation (all files)
python ml_validation_suite.py
```

This will:
1. Run gswoop on each file (ground truth)
2. Run ML model prediction
3. Run traditional algorithm
4. Compare errors
5. Generate report: `validation_results.csv`

## What The Report Will Tell You

**If ML is much better:**
```
✅ ML better: 80/100 (80%)
   Average improvement: +35.2%
→ KEEP IT! You have a working ML system!
```

**If ML is about the same:**
```
⚠️  ML better: 51/100 (51%)
   Average improvement: +2.3%
→ CONSIDER RETRAINING with more data or better features
```

**If ML is worse:**
```
❌ ML better: 30/100 (30%)
   Average improvement: -15.2%
→ Either retrain or disable (keep traditional only)
```

## The 83.1% Improvement Claim

**What it means:** Compared to just guessing the average rotation, ML is 83.1% better

**Is it good?** Yes, but...
- Real test: How does ML compare to the TRADITIONAL algorithm?
- That's what the validation suite checks

## How To Understand The Code

### Loading the model (happens once at startup)
```python
# flights/flight_manager.py, line 132-135
ml_singleton = MLModelSingleton()
self.ml_model = ml_singleton.model  # Pre-loaded Random Forest
self.ml_feature_names = ml_singleton.feature_names  # ['flight_duration', ...]
self.ml_model_loaded = ml_singleton.model is not None
```

### Making a prediction (happens for every flight)
```python
# flights/flight_manager.py, line 1209-1225
features = self.extract_ml_features(df, flare_idx, max_gspeed_idx)
feature_vector = np.array([[features[name] for name in self.ml_feature_names]])
ml_rotation = self.ml_model.predict(feature_vector)[0]
confidence = min(1.0, max(0.3, 1.0 - abs(ml_rotation) / 1200))

if confidence > 0.4:
    use ML prediction
else:
    use traditional algorithm
```

## Training The Model (If You Want To Retrain)

You would run:
```bash
python ml_training_pipeline.py       # Generate training data from files
python multi_metric_ml_pipeline.py   # Train the model
```

This requires:
1. Files in `~/FlySight/Training/`
2. gswoop command-line tool installed
3. sklearn, pandas, numpy, joblib

## Potential Issues & Solutions

### Issue 1: Model not loading
```
❌ Error loading ML model
```
**Fix:** Rerun `multi_metric_ml_pipeline.py` to regenerate model file

### Issue 2: Predictions way off
```
⚠️  ML error: 150°, Traditional error: 20°
```
**Fix:** Either retrain with more data, or disable ML (keep traditional only)

### Issue 3: Confidence always low
```
⚠️  Average confidence: 0.2 (usually uses fallback)
```
**Fix:** Adjust confidence threshold (currently 0.4) or retrain

## Questions You Can Now Answer

**Q: How is the ML model trained?**
A: Random Forest (100 trees) trained on 11 flight features vs gswoop ground truth

**Q: What does it predict?**
A: Rotation angle in degrees with confidence 0.0-1.0

**Q: Is it lightweight?**
A: Yes - 786 KB file, 30ms per prediction

**Q: What if it's wrong?**
A: Falls back to traditional algorithm (designed as safety fallback)

**Q: How do I know if it's working?**
A: Run `python ml_validation_suite.py` and check if ML errors < traditional errors

**Q: Can I improve it?**
A: Yes - more training data, better features, different algorithm, hyperparameter tuning

**Q: Do I need it?**
A: Unknown until you validate! The traditional algorithm is working fine (~0.9% error vs gswoop). ML might help but needs proof.

## Next Steps

1. **Run validation**: `python ml_validation_suite.py 50`
2. **Check results**: Look at `validation_results.csv`
3. **Decide**: Keep, retrain, or disable
4. **Document**: Share findings so you remember

## Want To Dive Deeper?

Read: `ML_SYSTEM_WALKTHROUGH.md` (long detailed explanation)

## Files You Can Reference

```
Documentation:
  - ML_SYSTEM_WALKTHROUGH.md (this detailed walkthrough)
  - ML_SUMMARY.md (this quick reference)

Training Code:
  - ml_training_pipeline.py
  - multi_metric_ml_pipeline.py

Integration Code:
  - flights/flight_manager.py (lines 43-78, 1164-1231)

Validation:
  - ml_validation_suite.py (run this!)

Model:
  - flights/rotation_prediction_model.pkl
  - ml_training_data.csv
```

---

**Bottom Line:** You have a working ML system that needs validation. Run the validation suite with your files. It will tell you if the ML model is actually helping or just taking up disk space.
