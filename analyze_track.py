import numpy as np
import pandas as pd

# --- Load file (skip FlySight units row) ---
df = pd.read_csv('/mnt/e/skydive_tracks/gps_02260.csv', skiprows=[1])
df.columns = [c.strip() for c in df.columns]
df['ts'] = pd.to_datetime(df['time'], utc=True)
df = df.sort_values('ts').reset_index(drop=True)
df['timestamp'] = df['ts'].astype('int64') / 1e9
df['t_s'] = df['timestamp'] - df['timestamp'].iloc[0]

for col in ['hMSL', 'velN', 'velE', 'velD', 'vAcc', 'heading']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# AGL: hMSL minus DZ elevation (min hMSL in track)
dz_elev = df['hMSL'].min()
df['altitude_agl'] = df['hMSL'] - dz_elev
df['h_speed'] = np.sqrt(df['velN']**2 + df['velE']**2)

print(f'Track: {len(df)} points, {df["t_s"].iloc[-1]:.0f}s duration')
print(f'DZ elevation (min hMSL): {dz_elev:.1f}m ({dz_elev/0.3048:.0f}ft)')
print(f'Max AGL: {df["altitude_agl"].max()/0.3048:.0f}ft')

# --- Constants matching canopy_stats.py ---
MAX_CANOPY_VSPEED_MS = 70 * 0.44704
MAX_CANOPY_HSPEED_MS = 100 * 0.44704
MIN_CANOPY_VSPEED_MS = 11 * 0.3048
MIN_ALT_AGL_M = 1000 * 0.3048
HEADING_THRESHOLD_DEG = 10.0
MIN_SEGMENT_DURATION_S = 2.0
MAX_GAP_S = 1.0
MAX_HSPEED_CHANGE_MS = 10 * 0.44704
MPS_TO_MPH = 1.0 / 0.44704

# --- Canopy start detection ---
smooth_win = 15
smooth_agl = df['altitude_agl'].rolling(smooth_win, center=True, min_periods=1).median()
smooth_vspeed = df['velD'].rolling(smooth_win, center=True, min_periods=1).mean()

rough_exit_idx = int(smooth_agl.idxmax())
landing_idx = len(df) - 1
post_exit = np.arange(rough_exit_idx, landing_idx)

ff_mask = smooth_vspeed.iloc[post_exit].values >= 10.0
ff_onset_idx = int(post_exit[np.argmax(ff_mask)]) if ff_mask.any() else rough_exit_idx

# Estimate fs from median dt
dts = np.diff(df['t_s'].values)
dt = float(np.nanmedian(dts[dts > 0])) if len(dts) > 0 else 0.2
fs = 1.0 / dt if dt > 0 else 5.0
min_wait_n = int(round(5.0 * fs))
neg_win = int(round(2.0 * fs))

earliest = min(ff_onset_idx + min_wait_n, landing_idx)

if smooth_vspeed.iloc[earliest] < MAX_CANOPY_VSPEED_MS:
    canopy_start_idx = min(earliest + min_wait_n, landing_idx)
else:
    dvspeed = np.diff(smooth_vspeed.values, prepend=smooth_vspeed.values[0])
    post = np.arange(earliest, landing_idx)
    if len(post) == 0:
        canopy_start_idx = earliest
    else:
        neg = (dvspeed[post] < 0).astype(int)
        neg_roll = pd.Series(neg).rolling(neg_win, min_periods=neg_win).sum().values
        hits = np.where(neg_roll >= neg_win)[0]
        if hits.size == 0:
            peak_idx = int(post[0]) + int(np.argmax(smooth_vspeed.values[post]))
            canopy_start_idx = min(peak_idx + min_wait_n, landing_idx)
        else:
            drop_start_rel = max(0, int(hits[0]) - neg_win + 1)
            canopy_start_idx = min(int(post[drop_start_rel]) + min_wait_n, landing_idx)

canopy_start_ts = df.iloc[canopy_start_idx]['timestamp']
print(f'Canopy start: t_s={df.iloc[canopy_start_idx]["t_s"]:.1f}s, '
      f'AGL={df.iloc[canopy_start_idx]["altitude_agl"]/0.3048:.0f}ft')

# --- Filter to canopy phase ---
canopy = df[
    (df['timestamp'] >= canopy_start_ts) &
    (df['altitude_agl'] >= MIN_ALT_AGL_M) &
    (df['velD'] > MIN_CANOPY_VSPEED_MS) &
    (df['velD'] < MAX_CANOPY_VSPEED_MS) &
    (df['h_speed'] < MAX_CANOPY_HSPEED_MS)
].copy().reset_index(drop=True)

print(f'Canopy-phase points after filter: {len(canopy)}')


def circular_range(angles):
    if len(angles) <= 1:
        return 0.0
    s = sorted(angles)
    gaps = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    gaps.append(360 - s[-1] + s[0])
    return 360 - max(gaps)


def find_straight_segments(times, headings, h_speeds):
    n = len(times)
    segments = []
    i = 0
    while i < n:
        j = i
        hw = [headings[j]]
        sw = [h_speeds[j]]
        while j + 1 < n:
            if times[j + 1] - times[j] > MAX_GAP_S:
                break
            candidate = hw + [headings[j + 1]]
            if circular_range(candidate) <= HEADING_THRESHOLD_DEG:
                hw.append(headings[j + 1])
                sw.append(h_speeds[j + 1])
                j += 1
            else:
                break
        duration = times[j] - times[i]
        spd_range = max(sw) - min(sw)
        if duration >= MIN_SEGMENT_DURATION_S and spd_range <= MAX_HSPEED_CHANGE_MS:
            segments.append((i, j))
            i = j + 1
        else:
            i += 1
    return segments


if len(canopy) < 4:
    print('Not enough canopy-phase points for stats.')
else:
    segs = find_straight_segments(
        canopy['timestamp'].tolist(),
        canopy['heading'].tolist(),
        canopy['h_speed'].tolist(),
    )

    straight_idx = set()
    for s, e in segs:
        straight_idx.update(range(s, e + 1))
    straight = canopy.iloc[sorted(straight_idx)].copy()

    # vAcc gate
    v_acc = pd.to_numeric(straight['vAcc'], errors='coerce').values
    valid = ~np.isnan(v_acc)
    if valid.sum() > 2:
        med = float(np.median(v_acc[valid]))
        mad = float(np.median(np.abs(v_acc[valid] - med))) or 1.0
        vacc_thr = med + 2.5 * mad
        vacc_ok = np.where(valid, v_acc <= vacc_thr, True)
    else:
        vacc_ok = np.ones(len(straight), dtype=bool)

    # Glide ratios per segment
    timestamps = straight['timestamp'].values
    alt_agl = straight['altitude_agl'].values
    h_spd_arr = straight['h_speed'].values

    seg_starts = [0]
    for i in range(1, len(straight)):
        if timestamps[i] - timestamps[i - 1] > MAX_GAP_S:
            seg_starts.append(i)
    seg_starts.append(len(straight))

    ratios = []
    for k in range(len(seg_starts) - 1):
        s = seg_starts[k]
        e = seg_starts[k + 1]
        if e - s < 2:
            continue
        if not vacc_ok[s] or not vacc_ok[e - 1]:
            continue
        alt_loss = alt_agl[s] - alt_agl[e - 1]
        if alt_loss <= 0:
            continue
        h_dist = float(np.sum(h_spd_arr[s:e - 1] * np.diff(timestamps[s:e])))
        if h_dist > 0:
            ratios.append(h_dist / alt_loss)

    h_vals = straight['h_speed'].values
    v_vals = straight['velD'].values

    print()
    print('=== CANOPY STATS (gps_02260.csv) ===')
    print(f'Straight-line segments:   {len(segs)}')
    print(f'Avg glide ratio:          {np.mean(ratios):.2f}' if ratios else 'Avg glide ratio:          N/A')
    print(f'Avg horizontal speed:     {np.mean(h_vals) * MPS_TO_MPH:.1f} mph')
    print(f'Avg vertical speed:       {np.mean(v_vals) * MPS_TO_MPH:.1f} mph')
    print(f'Max horizontal speed:     {np.max(h_vals) * MPS_TO_MPH:.1f} mph')
    print(f'Min horizontal speed:     {np.min(h_vals) * MPS_TO_MPH:.1f} mph')
    print(f'Max vertical speed:       {np.max(v_vals) * MPS_TO_MPH:.1f} mph')
    print(f'Min vertical speed:       {np.min(v_vals) * MPS_TO_MPH:.1f} mph')

    print()
    print('Segment detail:')
    for k, (s, e) in enumerate(segs):
        seg = canopy.iloc[s:e + 1]
        dur = seg['timestamp'].iloc[-1] - seg['timestamp'].iloc[0]
        alt = seg['altitude_agl'].mean() / 0.3048
        h = seg['h_speed'].mean() * MPS_TO_MPH
        v = seg['velD'].mean() * MPS_TO_MPH
        hdg = seg['heading'].mean()
        spd_range = (seg['h_speed'].max() - seg['h_speed'].min()) * MPS_TO_MPH
        print(f'  [{k + 1:2d}] dur={dur:.1f}s  alt={alt:.0f}ft  '
              f'h={h:.1f}mph  v={v:.1f}mph  hdg={hdg:.0f}deg  spd_range={spd_range:.1f}mph')
