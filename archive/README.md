# Archive

This directory contains files that were generated during development but are not currently part of the active project. These files are archived for reference but are not required for the application to run.

## Contents

### scripts/
One-time test and fix scripts that are no longer needed:

- **test_flight_naming.py** - Test script for flight naming system
- **test_complete_naming_system.py** - Comprehensive test for flight naming
- **test_gate_parser.py** - Parser test for gate files
- **test_ux_improvements.py** - UX improvement testing
- **fix_flight_235.py** - One-time fix for a specific flight (flight 235)
- **debug_chart_data.py** - Debug script for chart data issues
- **diagnose_gps_storage.py** - GPS data storage diagnostics

**Why archived:** These are one-time diagnostic and testing scripts. The functionality they tested is either now integrated into the codebase or no longer needed.

### templates/
Work-in-progress templates that haven't been integrated:

- **select_swoop_window.html** - Chart-based swoop window selector
  - Interactive UI for manually selecting swoop segments on a chart
  - Status: Not yet integrated into the application
  - Related feature: Manual swoop window selection

**Why archived:** Feature not yet implemented. Template is ready but the backend integration and view routing haven't been completed.

## When to Reference

You might want to look at these files if:

1. **Implementing manual swoop selection** - See `select_swoop_window.html`
2. **Debugging specific historical issues** - See corresponding `fix_*.py` or `debug_*.py` scripts
3. **Understanding testing approach** - See `test_*.py` scripts
4. **Gate file parsing** - See `test_gate_parser.py`

## Active Diagnostic Tools

For current diagnostics, use the tools in the root directory instead:
- `diagnose_ml_in_production.py` - Modern ML diagnostics
- `ml_validation_suite.py` - ML validation and comparison
- `performance_profiler.py` - Performance profiling

## To Restore

If you need to restore a file from archive:
```bash
mv archive/scripts/filename.py .
mv archive/templates/filename.html templates/location/
```

## Cleanup Policy

Files are archived when:
- They are one-time fixes that have been applied
- They are test scripts for features that have been implemented
- They are work-in-progress templates that aren't currently being developed
- They are diagnostic scripts that have been superseded by newer tools

Files remain in root directory when they are:
- Regularly used tools (`ml_validation_suite.py`, `performance_profiler.py`)
- Production diagnostic scripts (`diagnose_ml_in_production.py`)
- Examples or templates that are referenced by documentation
