
================================================================================
SWOOPR PERFORMANCE ANALYSIS REPORT
================================================================================

OPERATION TIMING & RESOURCE USAGE
--------------------------------------------------------------------------------
Operation                           Time (s)     Memory (MB)     IO Read (MB)   
--------------------------------------------------------------------------------
Initialize FlightManager            2.73         +100.9          N/A            
Read FlySight File                  0.27         +9.3            N/A            
Analyze Swoop (No DB)               0.25         +1.2            N/A            
  └─ Rotation metrics (traditional) 0.21         +0.7            N/A            
  └─ ML prediction                  0.03         +0.2            N/A            
  └─ ML feature extraction          0.00         +0.0            N/A            
  └─ Flare detection                0.00         +0.0            N/A            
  └─ Max speeds                     0.00         +0.0            N/A            
  └─ Landing detection              0.00         +0.0            N/A            
--------------------------------------------------------------------------------
TOTAL                               3.49         +112.3

BOTTLENECK ANALYSIS
--------------------------------------------------------------------------------
1. Initialize FlightManager: 2.73s (78.2% of total)
2. Read FlySight File: 0.27s (7.7% of total)
3. Analyze Swoop (No DB): 0.25s (7.1% of total)
4.   └─ Rotation metrics (traditional): 0.21s (5.9% of total)
5.   └─ ML prediction: 0.03s (1.0% of total)

MEMORY-INTENSIVE OPERATIONS
--------------------------------------------------------------------------------
1. Initialize FlightManager: 40.1MB peak
2. Read FlySight File: 4.0MB peak
3.   └─ Landing detection: 1.4MB peak
4.   └─ Flare detection: 1.0MB peak
5.   └─ Rotation metrics (traditional): 0.5MB peak

IO-INTENSIVE OPERATIONS
--------------------------------------------------------------------------------
1. Initialize FlightManager: 0.0MB read, 0.0MB written
2. Read FlySight File: 0.0MB read, 0.0MB written
3.   └─ Landing detection: 0.0MB read, 0.0MB written
4.   └─ Flare detection: 0.0MB read, 0.0MB written
5.   └─ Max speeds: 0.0MB read, 0.0MB written
