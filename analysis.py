# ══════════════════════════════════════════════════════════════════════════════
# P7 — F1 Driver Cardiovascular Demand Estimator
# ══════════════════════════════════════════════════════════════════════════════
#
# Derives a cardiovascular demand profile for an F1 driver from raw telemetry
# using a two-component HR proxy model grounded in peer-reviewed physiology.
# No wearable data required — pure physics-to-physiology translation.
#
# Sources:
#   Tornaghi et al. 2023  — Heart rate profiling in F1 race (Science & Sports)
#   Tripoli et al. 2024   — Acute cardiovascular response to gravity changes
#                           (IAC 2024, Politecnico di Torino)
#   Blomqvist & Stone 1983 — Cardiovascular adjustments to gravitational stress
#                           (Handbook of Physiology, Chapter 28)
#
# Usage:
#   Run in Google Colab or locally.
#   All outputs saved to outputs/ directory.
# ══════════════════════════════════════════════════════════════════════════════

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
import warnings
import os

warnings.filterwarnings('ignore')
os.makedirs('outputs', exist_ok=True)


# ── Section 1: Configuration & Physiological Constants ────────────────────────

# Physiological parameters — all sourced from peer-reviewed literature
HR_MAX     = 200    # bpm — Tornaghi et al. 2023 (F1-specific, lab-measured)
GLOC_THRESHOLD = 4.5  # Gz — Blomqvist & Stone 1983 (relaxed subjects,
                       #      conservative lower bound; trained drivers likely
                       #      tolerate higher — no motorsport-specific threshold
                       #      exists in peer-reviewed literature)
                      # *Note: It is a mechanical stress reference, not a true G-Loc threshold


# Tripoli et al. 2024: HR increases linearly from ~77 bpm at 1g → ~113 bpm
# at 2.5g for a seated subject. Slope = +36 bpm over +1.5g = 24 bpm/g.
TRIPOLI_SLOPE = 24.0   # bpm per g above 1g

# Tornaghi et al. 2023: sustained in-race HR lower bound = 74% HRmax.
# This represents the cardiovascular floor of race-pace driving — exertion,
# thermal load, cognitive demand — even at low G (Tornaghi 2023).
# Note: Tornaghi 2023 data from Melbourne (16–18°C ambient) — a conservative baseline.
# Hotter circuits (Bahrain, Singapore) would elevate this floor further.


HR_EXERTION_BASE_PCT = 0.74
HR_EXERTION_BASE     = HR_EXERTION_BASE_PCT * HR_MAX   # 148 bpm

# HR zone thresholds as % of HRmax (Tornaghi et al. 2023 framing)
ZONES = {
    'Recovery':  (0.00, 0.60),
    'Aerobic':   (0.60, 0.74),
    'Sustained': (0.74, 0.82),   # Tornaghi race mean
    'Peak Race': (0.82, 0.92),   # Tornaghi race peaks
    'Maximal':   (0.92, 1.00),   # Tornaghi qualifying peaks
}
ZONE_COLORS = {
    'Recovery':  '#4ade80',
    'Aerobic':   '#facc15',
    'Sustained': '#fb923c',
    'Peak Race': '#f87171',
    'Maximal':   '#dc2626',
}

# Visual palette
BG     = '#ffffff'
ACCENT = '#E8002D'
DARK   = '#1a1a2e'
LIGHT  = '#e2e2e2'

# Circuit configurations — VER 2023 Race sessions
CIRCUITS = {
    'Suzuka':      {'session_key': 9173, 'length_m': 5807,
                    'color': '#E8002D'},
    'Silverstone': {'session_key': 9149, 'length_m': 5891,
                    'color': '#1E41FF'},
    'Monaco':      {'session_key': 9094, 'length_m': 3337,
                    'color': '#00A39A'},
}

print("✅ Configuration loaded")
print(f"   HR_REST={HR_REST} bpm | HR_MAX={HR_MAX} bpm | "
      f"Hemo. stress ref {GLOC_THRESHOLD}g | Tripoli slope={TRIPOLI_SLOPE} bpm/g")
print(f"   Exertion base={HR_EXERTION_BASE:.0f} bpm "
      f"({HR_EXERTION_BASE_PCT*100:.0f}% HRmax, Tornaghi 2023)")


# ── Section 2: OpenF1 Data Fetching ───────────────────────────────────────────

def fetch_openf1(endpoint, params):
    """Standard OpenF1 API call — params dict auto-encoded."""
    url = f"https://api.openf1.org/v1/{endpoint}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_openf1_raw(endpoint, params_exact):
    """
    OpenF1 filter operators (>, <) must NOT be URL-encoded.
    requests encodes > as %3E which returns 404.
    This function builds the query string manually.
    """
    base = f"https://api.openf1.org/v1/{endpoint}"
    qs   = "&".join(f"{k}={v}" for k, v in params_exact.items())
    url  = f"{base}?{qs}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def get_driver_number(session_key, driver_abbr='VER'):
    """Get driver number from session."""
    df = fetch_openf1('drivers', {
        'session_key' : session_key,
        'name_acronym': driver_abbr
    })
    if df.empty:
        raise ValueError(f"Driver {driver_abbr} not found in "
                         f"session {session_key}")
    return df.iloc[0]['driver_number']


def get_lap_location(session_key, driver_number, lap_number):
    """
    Fetch x,y,z car location for a specific lap via OpenF1 /location.
    Returns ~3.7 Hz position data as DataFrame.
    """
    laps = fetch_openf1('laps', {
        'session_key'  : session_key,
        'driver_number': driver_number,
        'lap_number'   : lap_number
    })
    if laps.empty:
        return pd.DataFrame()

    lap_start = pd.to_datetime(laps.iloc[0]['date_start'])
    lap_end   = lap_start + pd.Timedelta(seconds=130)

    loc = fetch_openf1_raw('location', {
        'session_key'  : session_key,
        'driver_number': driver_number,
        'date>'        : lap_start.strftime('%Y-%m-%dT%H:%M:%S'),
        'date<'        : lap_end.strftime('%Y-%m-%dT%H:%M:%S'),
    })

    if loc.empty:
        return pd.DataFrame()

    loc['date'] = pd.to_datetime(loc['date'])
    return loc.sort_values('date').reset_index(drop=True)


print("✅ OpenF1 fetcher ready")


# ── Section 3: G-Force Physics Engine ─────────────────────────────────────────

def compute_gforces(pos_df, smooth=True, circuit_length_m=5807):
    """
    Derive G-forces from OpenF1 location data (~3.7 Hz).

    Scale detection: empirical from known circuit length.
      scale = circuit_length_m / raw_path_length_in_units

    G-force method:
      G_long : longitudinal — from smoothed speed derivative
      G_lat  : lateral      — heading-rate method (v × ω / g)
                              More stable than curvature at low sample rates
                              because heading is a first derivative vs
                              curvature which requires second derivatives,
                              amplifying noise quadratically.
      G_total: vector magnitude

    Physical F1 limits used for clipping:
      G_long braking  ≤ 5.0g  (carbon brakes, short burst)
      G_long accel    ≤ 2.0g  (power-limited)
      G_lat           ≤ 4.5g  (qualifying, new tyre)
    """
    df = pos_df.copy()
    df['t']  = (df['date'] - df['date'].iloc[0]).dt.total_seconds()
    df['dx'] = df['x'].diff()
    df['dy'] = df['y'].diff()
    df['dt'] = df['t'].diff()
    df = df[df['dt'] > 0].copy()

    # ── Empirical unit scale ──────────────────────────────────────────────────
    raw_path = np.sqrt(df['dx']**2 + df['dy']**2).sum()
    scale    = circuit_length_m / raw_path
    print(f"   Scale: {scale:.4f} m/unit | "
          f"path {raw_path*scale:.0f} m (expected {circuit_length_m} m)")

    # ── Raw speed → smooth speed ──────────────────────────────────────────────
    df['ds']       = np.sqrt(df['dx']**2 + df['dy']**2) * scale
    df['speed_ms'] = (df['ds'] / df['dt']).clip(0, 95)

    n  = len(df)
    w1 = min(25, (n // 8) * 2 + 1)
    w1 = w1 if w1 % 2 == 1 else w1 + 1
    w1 = max(w1, 7)
    df['speed_s'] = savgol_filter(
        df['speed_ms'].fillna(0), w1, 3).clip(0, 95)
    print(f"   Speed: {df['speed_s'].min():.1f}–"
          f"{df['speed_s'].max():.1f} m/s = "
          f"{df['speed_s'].max()*3.6:.0f} km/h  (w1={w1})")

    # ── Longitudinal G ────────────────────────────────────────────────────────
    df['G_long'] = (df['speed_s'].diff() / df['dt']) / 9.81
    df['G_long'] = df['G_long'].clip(-5.0, 2.0)

    # ── Lateral G via heading rate ────────────────────────────────────────────
    df['heading'] = np.unwrap(np.arctan2(
        df['dy'].fillna(0), df['dx'].fillna(0)))
    df['dheading'] = df['heading'].diff()
    df['omega']    = df['dheading'] / df['dt']

    w2 = min(21, (n // 8) * 2 + 1)
    w2 = w2 if w2 % 2 == 1 else w2 + 1
    w2 = max(w2, 7)
    df['omega_s'] = savgol_filter(df['omega'].fillna(0), w2, 2)
    df['G_lat']   = (df['speed_s'] * df['omega_s'].abs() / 9.81).clip(0, 4.5)

    # ── Total G + final smoothing pass ────────────────────────────────────────
    df['G_total'] = np.sqrt(df['G_long']**2 + df['G_lat']**2).clip(0, 5.5)

    if smooth:
        w3 = min(9, (n // 12) * 2 + 1)
        w3 = w3 if w3 % 2 == 1 else w3 + 1
        w3 = max(w3, 5)
        df['G_long']  = savgol_filter(
            df['G_long'].fillna(0), w3, 2).clip(-5.0, 2.0)
        df['G_lat']   = savgol_filter(
            df['G_lat'].fillna(0),  w3, 2).clip(0, 4.5)
        df['G_total'] = np.sqrt(
            df['G_long']**2 + df['G_lat']**2).clip(0, 5.5)

    df['speed_ms'] = df['speed_s']
    return df.dropna(subset=['G_total']).reset_index(drop=True)


print("✅ G-force engine ready")


# ── Section 4: Cardiovascular Proxy Model ─────────────────────────────────────

def compute_hr_proxy(g_df):
    """
    Two-component HR model:

      HR_total(t) = 74% × HRmax  +  24 bpm/g × max(0, G(t) − 1.0)
                    ─────────────   ──────────────────────────────────
                    Exertion floor  G-force increment
                    Tornaghi 2023   Tripoli 2024

    Component 1 (74% HRmax = 148 bpm):
      Sustained cardiovascular cost of race-pace driving — exertion,
      thermal load, cognitive demand — even at low G (Tornaghi 2023).

    Component 2 (Tripoli slope):
      G-induced HR increment above 1g baseline.
      Validated: at G=1.5g → HR=160 bpm (80% HRmax) ✓ race sustained range
                 at G=2.5g → HR=184 bpm (92% HRmax) ✓ race peak range
                 (both within Tornaghi's observed bounds)

    Also derives:
      Stroke volume depression: −33.4% from 1g → 2.5g (Tripoli 2024)
      Cardiac output index: HR × SV, normalised to 1.0 at exertion baseline

    LIMITATION:
      The two HR components cannot be separated experimentally without
      controlled testing isolating thermal, cognitive, and G-force
      contributions. This model estimates TOTAL cardiovascular demand.
      Direct wearable measurement remains the gold standard.
    """
    df = g_df.copy()

    # HR proxy
    df['HR_proxy'] = (HR_EXERTION_BASE
                      + TRIPOLI_SLOPE * np.maximum(0, df['G_total'] - 1.0))
    df['HR_proxy'] = df['HR_proxy'].clip(HR_EXERTION_BASE, HR_MAX)

    # %HRmax
    df['pct_HRmax'] = df['HR_proxy'] / HR_MAX

    # Zone assignment
    def assign_zone(pct):
        for zone, (lo, hi) in ZONES.items():
            if lo <= pct < hi:
                return zone
        return 'Maximal'
    df['zone'] = df['pct_HRmax'].apply(assign_zone)

    # Hemodynamic Stress Index (0 = safe, 1 = at Blomqvist threshold)
    df['hemodynamic_stress_idx'] = (df['G_total'] / GLOC_THRESHOLD).clip(0, 1)

    # Stroke volume depression — Tripoli 2024: SV falls −33.4% over 1g→2.5g
    df['SV_pct_baseline'] = (1.0 - 0.334
                             * np.maximum(0, df['G_total'] - 1.0) / 1.5)
    df['SV_pct_baseline'] = df['SV_pct_baseline'].clip(0.4, 1.0)

    # Cardiac output index (normalised to 1.0 at exertion baseline)
    df['CO_index'] = (df['HR_proxy'] / HR_EXERTION_BASE) * df['SV_pct_baseline']

    return df


print("✅ Cardiovascular model ready")


# ── Section 5: Data Fetching Pipeline ─────────────────────────────────────────

def fetch_circuit_data(circuits=CIRCUITS, driver_abbr='VER', target_lap=10):
    """
    Fetch and process location + CV data for all circuits.
    Falls back through laps 10, 8, 12, 15 if insufficient points returned.
    """
    circuit_data = {}

    for name, cfg in circuits.items():
        print(f"\nFetching {name} (session {cfg['session_key']})...")
        try:
            drv = get_driver_number(cfg['session_key'], driver_abbr)
            loc = pd.DataFrame()

            for lap in [target_lap, 8, 12, 15]:
                loc = get_lap_location(cfg['session_key'], drv, lap)
                if len(loc) >= 50:
                    print(f"   Lap {lap}: {len(loc)} points ✓")
                    break
                print(f"   Lap {lap}: {len(loc)} points — trying next...")

            if len(loc) < 50:
                print(f"   ⚠ Insufficient data for {name}, skipping")
                continue

            g  = compute_gforces(loc, smooth=True,
                                 circuit_length_m=cfg['length_m'])
            cv = compute_hr_proxy(g)

            circuit_data[name] = {
                'g': g, 'cv': cv,
                'color': cfg['color'],
                'length_m': cfg['length_m']
            }

            print(f"   G_total  max/mean : "
                  f"{cv['G_total'].max():.2f}g / {cv['G_total'].mean():.2f}g")
            print(f"   HR_proxy max/mean : "
                  f"{cv['HR_proxy'].max():.0f} / "
                  f"{cv['HR_proxy'].mean():.0f} bpm")

        except Exception as e:
            print(f"   ⚠ Error: {e}")

    print(f"\n✅ Loaded {len(circuit_data)} circuits: "
          f"{list(circuit_data.keys())}")
    return circuit_data


# ── Section 6: Module 1 — Lap Cardiovascular Arc ──────────────────────────────

def plot_module1(cv_df, lap_num=10, circuit='Suzuka',
                 driver='VER', year=2023):
    """
    Single-lap cardiovascular demand overview:
      - Circuit map colored by %HRmax
      - HR proxy curve with Tornaghi zone bands
      - G-force components (G_long, G_lat, G_total)
      - Time in HR zone bar chart
      - Stroke volume & cardiac output index
    """
    fig = plt.figure(figsize=(18, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    gs = GridSpec(3, 2, figure=fig,
                  left=0.06, right=0.97,
                  top=0.88,  bottom=0.07,
                  hspace=0.45, wspace=0.35)

    ax_map  = fig.add_subplot(gs[0:2, 0])
    ax_hr   = fig.add_subplot(gs[0, 1])
    ax_g    = fig.add_subplot(gs[1, 1])
    ax_zone = fig.add_subplot(gs[2, 0])
    ax_sv   = fig.add_subplot(gs[2, 1])

    for ax in [ax_map, ax_hr, ax_g, ax_zone, ax_sv]:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_edgecolor(LIGHT)
            spine.set_linewidth(0.5)

    t   = cv_df['t'].values
    hr  = cv_df['HR_proxy'].values
    g   = cv_df['G_total'].values
    x   = cv_df['x'].values
    y   = cv_df['y'].values
    sv  = cv_df['SV_pct_baseline'].values * 100
    co  = cv_df['CO_index'].values

    # ── Circuit map ───────────────────────────────────────────────────────────
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    pct    = cv_df['pct_HRmax'].values
    norm   = plt.Normalize(0.74, 1.00)
    cmap   = plt.cm.get_cmap('RdYlGn_r')
    lc     = LineCollection(segs, cmap=cmap, norm=norm,
                            linewidth=3.5, alpha=0.92)
    lc.set_array(pct[:-1])
    ax_map.add_collection(lc)

    mid = len(x) // 2
    ax_map.annotate('', xy=(x[mid+3], y[mid+3]),
                    xytext=(x[mid], y[mid]),
                    arrowprops=dict(arrowstyle='->', color=DARK, lw=1.5))

    ax_map.set_xlim(x.min()-400, x.max()+400)
    ax_map.set_ylim(y.min()-400, y.max()+400)
    ax_map.set_aspect('equal')
    ax_map.axis('off')
    ax_map.set_title(f'{circuit} — Lap {lap_num}\n'
                     f'Cardiovascular Load Map',
                     color=DARK, fontsize=11, fontweight='bold', pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax_map, orientation='horizontal',
                      fraction=0.04, pad=0.02)
    cb.set_label('%HRmax', color=DARK, fontsize=8)
    cb.set_ticks([0.74, 0.82, 0.92, 1.00])
    cb.set_ticklabels(['74%', '82%', '92%', '100%'])
    cb.ax.tick_params(colors=DARK, labelsize=7)

    # ── HR proxy curve ────────────────────────────────────────────────────────
    ax_hr.axhspan(0.74*HR_MAX, 0.82*HR_MAX,
                  color='#fb923c', alpha=0.12,
                  label='Race sustained (Tornaghi)')
    ax_hr.axhspan(0.82*HR_MAX, 0.92*HR_MAX,
                  color='#f87171', alpha=0.12,
                  label='Race peaks (Tornaghi)')
    ax_hr.axhspan(0.92*HR_MAX, HR_MAX,
                  color='#dc2626', alpha=0.10,
                  label='Qualifying peaks (Tornaghi)')

    ax_hr.plot(t, hr, color=ACCENT, linewidth=1.8, zorder=3)
    ax_hr.axhline(HR_MAX, color='#dc2626', linewidth=0.8,
                  linestyle='--', alpha=0.6,
                  label=f'HRmax {HR_MAX} bpm')

    ax_hr.set_ylabel('HR proxy (bpm)', color=DARK, fontsize=8)
    ax_hr.set_ylim(130, 210)
    ax_hr.set_xlim(t[0], t[-1])
    ax_hr.tick_params(colors=DARK, labelsize=7)
    ax_hr.set_title('Heart Rate Proxy', color=DARK,
                    fontsize=9, fontweight='bold')
    ax_hr.legend(fontsize=6, loc='upper right',
                 facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

    # ── G-force components ────────────────────────────────────────────────────
    ax_g.plot(t, cv_df['G_long'].abs(), color='#3b82f6',
              linewidth=1.2, alpha=0.8, label='|G_long|')
    ax_g.plot(t, cv_df['G_lat'], color='#f59e0b',
              linewidth=1.2, alpha=0.8, label='G_lat')
    ax_g.plot(t, g, color=DARK, linewidth=1.6, label='G_total')
    ax_g.axhline(GLOC_THRESHOLD, color='#dc2626', linewidth=0.9,
                 linestyle='--', alpha=0.7,
                 label=f'Hemo. stress ref {GLOC_THRESHOLD}g')
    ax_g.fill_between(t, g, alpha=0.06, color=DARK)

    ax_g.set_ylabel('G-force (g)', color=DARK, fontsize=8)
    ax_g.set_xlabel('Time (s)', color=DARK, fontsize=8)
    ax_g.set_xlim(t[0], t[-1])
    ax_g.set_ylim(0, 5.5)
    ax_g.tick_params(colors=DARK, labelsize=7)
    ax_g.set_title('G-Force Components', color=DARK,
                   fontsize=9, fontweight='bold')
    ax_g.legend(fontsize=6, loc='upper right',
                facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

    # ── Zone bar chart ────────────────────────────────────────────────────────
    zone_order = ['Recovery', 'Aerobic', 'Sustained', 'Peak Race', 'Maximal']
    zone_pcts  = [(cv_df['zone'] == z).sum() / len(cv_df) * 100
                  for z in zone_order]

    bars = ax_zone.barh(zone_order, zone_pcts,
                        color=[ZONE_COLORS[z] for z in zone_order],
                        edgecolor='white', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, zone_pcts):
        if val > 2:
            ax_zone.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                         f'{val:.1f}%', va='center',
                         color=DARK, fontsize=7)

    ax_zone.set_xlabel('% of lap time', color=DARK, fontsize=8)
    ax_zone.set_xlim(0, 105)
    ax_zone.tick_params(colors=DARK, labelsize=7)
    ax_zone.set_title('Time in HR Zone', color=DARK,
                      fontsize=9, fontweight='bold')

    # ── SV + CO index ─────────────────────────────────────────────────────────
    ax_sv2 = ax_sv.twinx()

    ax_sv.fill_between(t, sv, 100, alpha=0.25, color='#3b82f6',
                       label='SV depression')
    ax_sv.plot(t, sv, color='#3b82f6', linewidth=1.4)
    ax_sv.axhline(100, color=LIGHT, linewidth=0.6, linestyle='--')
    ax_sv.set_ylabel('Stroke Volume\n(% baseline)',
                     color='#3b82f6', fontsize=7)
    ax_sv.set_ylim(50, 105)
    ax_sv.tick_params(axis='y', colors='#3b82f6', labelsize=7)

    ax_sv2.plot(t, co, color='#a855f7', linewidth=1.4,
                linestyle='--', label='CO index')
    ax_sv2.axhline(1.0, color='#a855f7', linewidth=0.5,
                   linestyle=':', alpha=0.5)
    ax_sv2.set_ylabel('Cardiac Output\nindex (norm.)',
                      color='#a855f7', fontsize=7)
    ax_sv2.set_ylim(0.5, 1.3)
    ax_sv2.tick_params(axis='y', colors='#a855f7', labelsize=7)

    ax_sv.set_xlabel('Time (s)', color=DARK, fontsize=8)
    ax_sv.set_xlim(t[0], t[-1])
    ax_sv.tick_params(axis='x', colors=DARK, labelsize=7)
    ax_sv.set_title('Stroke Volume & Cardiac Output',
                    color=DARK, fontsize=9, fontweight='bold')

    lines1, labels1 = ax_sv.get_legend_handles_labels()
    lines2, labels2 = ax_sv2.get_legend_handles_labels()
    ax_sv.legend(lines1+lines2, labels1+labels2, fontsize=6,
                 loc='lower left', facecolor=BG,
                 edgecolor=LIGHT, labelcolor=DARK)

    # ── Header & stats ────────────────────────────────────────────────────────
    fig.text(0.5, 0.945,
             'P7 — F1 Driver Cardiovascular Demand Estimator',
             ha='center', color=DARK, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.918,
             f'{driver}  ·  {year} {circuit} GP  ·  Lap {lap_num}  ·  '
             f'HR model: Tripoli et al. 2024 + Tornaghi et al. 2023  ·  '
             f'Hemodynamic stress ref: Blomqvist & Stone 1983',
             ha='center', color='#555555', fontsize=8)

    stats = [
        ('Peak HR',         f"{cv_df['HR_proxy'].max():.0f} bpm"),
        ('Mean HR',         f"{cv_df['HR_proxy'].mean():.0f} bpm"),
        ('Peak G',          f"{cv_df['G_total'].max():.2f} g"),
        ('Max Hemo. Stress',  f"{cv_df['hemodynamic_stress_idx'].max()*100:.0f}%"),
        ('Max SV drop',     f"{(1-cv_df['SV_pct_baseline'].min())*100:.1f}%"),
        ('Time ≥82%HRmax',  f"{(cv_df['pct_HRmax']>=0.82).mean()*100:.0f}%"),
    ]
    for i, (label, val) in enumerate(stats):
        xp = 0.08 + i * 0.155
        fig.text(xp, 0.895, val,   ha='center', color=DARK,
                 fontsize=10, fontweight='bold')
        fig.text(xp, 0.882, label, ha='center', color='#777777',
                 fontsize=7)

    plt.savefig('outputs/p7_module1_lap_cv_arc.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("✅ Saved → outputs/p7_module1_lap_cv_arc.png")


# ── Section 7: Module 2 — Circuit Comparison ──────────────────────────────────

def plot_module2(circuit_data):
    """
    Three-circuit cardiovascular demand comparison:
      - Row 0: Circuit maps colored by %HRmax
      - Row 1: HR proxy curves (normalised lap %) overlaid
      - Row 2: Grouped bar chart of key cardiovascular metrics
    """
    fig = plt.figure(figsize=(20, 16), facecolor=BG)
    fig.patch.set_facecolor(BG)

    names      = list(circuit_data.keys())
    n_circuits = len(names)

    gs = GridSpec(3, n_circuits, figure=fig,
                  left=0.06, right=0.97,
                  top=0.88,  bottom=0.07,
                  hspace=0.50, wspace=0.30)

    # ── Row 0: circuit maps ───────────────────────────────────────────────────
    for col, name in enumerate(names):
        ax  = fig.add_subplot(gs[0, col])
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

        cv  = circuit_data[name]['cv']
        x, y = cv['x'].values, cv['y'].values
        pct  = cv['pct_HRmax'].values

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        norm   = plt.Normalize(0.74, 1.00)
        cmap   = plt.cm.get_cmap('RdYlGn_r')
        lc     = LineCollection(segs, cmap=cmap, norm=norm,
                                linewidth=3.0, alpha=0.92)
        lc.set_array(pct[:-1])
        ax.add_collection(lc)
        ax.set_xlim(x.min()-500, x.max()+500)
        ax.set_ylim(y.min()-500, y.max()+500)
        ax.set_aspect('equal')
        ax.axis('off')

        mean_hr  = cv['HR_proxy'].mean()
        mean_pct = mean_hr / HR_MAX * 100
        ax.set_title(f'{name}\nMean HR {mean_hr:.0f} bpm '
                     f'({mean_pct:.0f}% HRmax)',
                     color=DARK, fontsize=10, fontweight='bold', pad=6)

    # ── Row 1: HR curves (normalised lap %) ───────────────────────────────────
    ax_hr = fig.add_subplot(gs[1, :])
    ax_hr.set_facecolor(BG)
    for sp in ax_hr.spines.values():
        sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

    ax_hr.axhspan(0.74*HR_MAX, 0.82*HR_MAX,
                  color='#fb923c', alpha=0.08)
    ax_hr.axhspan(0.82*HR_MAX, 0.92*HR_MAX,
                  color='#f87171', alpha=0.08)
    ax_hr.axhspan(0.92*HR_MAX, HR_MAX,
                  color='#dc2626', alpha=0.07)

    for name, d in circuit_data.items():
        cv     = d['cv']
        t_norm = (cv['t'] - cv['t'].iloc[0]) / \
                 (cv['t'].iloc[-1] - cv['t'].iloc[0]) * 100
        ax_hr.plot(t_norm, cv['HR_proxy'],
                   color=d['color'], linewidth=1.6, alpha=0.85,
                   label=f"{name} (mean {cv['HR_proxy'].mean():.0f} bpm)")

    ax_hr.axhline(HR_MAX, color='#dc2626', linewidth=0.7,
                  linestyle='--', alpha=0.5,
                  label=f'HRmax {HR_MAX} bpm')
    ax_hr.set_ylabel('HR proxy (bpm)', color=DARK, fontsize=9)
    ax_hr.set_xlabel('Lap progress (%)', color=DARK, fontsize=9)
    ax_hr.set_xlim(0, 100)
    ax_hr.set_ylim(130, 210)
    ax_hr.tick_params(colors=DARK, labelsize=8)
    ax_hr.set_title('Heart Rate Proxy — Circuit Comparison '
                    '(normalised lap %)',
                    color=DARK, fontsize=10, fontweight='bold')
    ax_hr.legend(fontsize=8, loc='upper right',
                 facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

    # ── Row 2: grouped bar metrics ────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[2, :])
    ax_bar.set_facecolor(BG)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

    metrics = {
        'Mean HR\n(bpm)':         lambda cv: cv['HR_proxy'].mean(),
        'Peak HR\n(bpm)':         lambda cv: cv['HR_proxy'].max(),
        'Mean %HRmax':            lambda cv: cv['pct_HRmax'].mean()*100,
        'Time ≥82%HRmax\n(%)':    lambda cv: (cv['pct_HRmax']>=0.82
                                              ).mean()*100,
        'Mean G-total\n(g)':      lambda cv: cv['G_total'].mean(),
        'Max SV\ndepression (%)': lambda cv: (1-cv['SV_pct_baseline'].min()
                                              )*100,
        'Mean CO\nindex':         lambda cv: cv['CO_index'].mean(),
    }

    n_metrics = len(metrics)
    bar_width = 0.22
    x_pos     = np.arange(n_metrics)
    colors    = [circuit_data[n]['color'] for n in names]

    for i, name in enumerate(names):
        cv     = circuit_data[name]['cv']
        vals   = [fn(cv) for fn in metrics.values()]
        offset = (i - (n_circuits-1)/2) * bar_width
        bars   = ax_bar.bar(x_pos + offset, vals, bar_width,
                            label=name, color=colors[i],
                            alpha=0.85, edgecolor='white', linewidth=0.4)
        for bar, val in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.5, f'{val:.1f}',
                        ha='center', va='bottom',
                        color=DARK, fontsize=6.5)

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(metrics.keys(), color=DARK, fontsize=8)
    ax_bar.tick_params(axis='y', colors=DARK, labelsize=8)
    ax_bar.set_ylabel('Value', color=DARK, fontsize=9)
    ax_bar.legend(fontsize=8, facecolor=BG,
                  edgecolor=LIGHT, labelcolor=DARK)
    ax_bar.set_title('Cardiovascular Demand Metrics by Circuit',
                     color=DARK, fontsize=10, fontweight='bold')

    # ── Header ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.945,
             'P7 — F1 Driver Cardiovascular Demand Estimator',
             ha='center', color=DARK, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.918,
             'VER · 2023 Season · Lap 10 · '
             'Suzuka vs Silverstone vs Monaco  ·  '
             'HR model: Tripoli et al. 2024 + Tornaghi et al. 2023  ·  '
             'Hemodynamic stress ref: Blomqvist & Stone 1983',
             ha='center', color='#555555', fontsize=8)

    plt.savefig('outputs/p7_module2_circuit_comparison.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("✅ Saved → outputs/p7_module2_circuit_comparison.png")


# ── Section 8: Module 3 — Hemodynamic Stress Index & Recovery Analysis ─────────────────

def plot_module3(circuit_data):
    """
    Per-circuit physiological risk analysis:
      - Row 0: Hemodynamic Stress Index heatmap (colored by danger zone)
      - Row 1: HR trace with cardiac recovery windows highlighted
      - Row 2: Stroke volume depression timeline
    """
    fig = plt.figure(figsize=(20, 14), facecolor=BG)
    fig.patch.set_facecolor(BG)

    names = list(circuit_data.keys())
    n     = len(names)

    gs = GridSpec(3, n, figure=fig,
                  left=0.06, right=0.97,
                  top=0.88,  bottom=0.07,
                  hspace=0.55, wspace=0.32)

    for col, name in enumerate(names):
        d     = circuit_data[name]
        cv    = d['cv']
        color = d['color']
        t     = cv['t'].values
        t_norm = (t - t[0]) / (t[-1] - t[0]) * 100
        gloc  = cv['hemodynamic_stress_idx'].values
        hr    = cv['HR_proxy'].values
        sv    = cv['SV_pct_baseline'].values * 100

        # ── Row 0: Hemodynamic Stress Index ────────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor(BG)
        for sp in ax0.spines.values():
            sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

        ax0.fill_between(t_norm, gloc,
                         where=(gloc < 0.5),
                         color='#4ade80', alpha=0.5, label='Low (<50%)')
        ax0.fill_between(t_norm, gloc,
                         where=((gloc >= 0.5) & (gloc < 0.75)),
                         color='#fb923c', alpha=0.6,
                         label='Moderate (50–75%)')
        ax0.fill_between(t_norm, gloc,
                         where=(gloc >= 0.75),
                         color='#dc2626', alpha=0.7, label='High (>75%)')
        ax0.plot(t_norm, gloc, color=color, linewidth=1.2, alpha=0.8)
        ax0.axhline(1.0, color='#dc2626', linewidth=0.8,
                    linestyle='--', alpha=0.6)
        ax0.axhline(0.75, color='#fb923c', linewidth=0.6,
                    linestyle=':', alpha=0.5)

        ax0.set_xlim(0, 100)
        ax0.set_ylim(0, 1.15)
        ax0.set_ylabel('Hemodynamic Stress Index\n(0=safe, 1=threshold)',
                       color=DARK, fontsize=7)
        ax0.set_title(f'{name}', color=DARK,
                      fontsize=10, fontweight='bold')
        ax0.tick_params(colors=DARK, labelsize=7)

        max_prox = gloc.max()
        pct_high = (gloc >= 0.75).mean() * 100
        ax0.text(0.97, 0.92,
                 f'Peak: {max_prox:.2f}\n>75%: {pct_high:.1f}% of lap',
                 transform=ax0.transAxes, ha='right', va='top',
                 color=DARK, fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=BG, edgecolor=LIGHT, alpha=0.8))
        if col == 0:
            ax0.legend(fontsize=6, loc='upper left',
                       facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

        # ── Row 1: HR + recovery windows ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor(BG)
        for sp in ax1.spines.values():
            sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

        hr_thresh_base = 0.76 * HR_MAX
        in_recovery = hr <= hr_thresh_base

        ax1.fill_between(t_norm, hr,
                         where=~in_recovery,
                         color=color, alpha=0.25, label='High demand')
        ax1.fill_between(t_norm, hr,
                         where=in_recovery,
                         color='#4ade80', alpha=0.35,
                         label='Recovery window')
        ax1.plot(t_norm, hr, color=color, linewidth=1.4)
        ax1.axhline(hr_thresh_base, color='#4ade80', linewidth=0.7,
                    linestyle='--', alpha=0.7, label='Recovery floor')
        ax1.axhline(0.92*HR_MAX, color='#dc2626', linewidth=0.6,
                    linestyle=':', alpha=0.5)

        ax1.set_xlim(0, 100)
        ax1.set_ylim(130, 210)
        ax1.set_ylabel('HR proxy (bpm)', color=DARK, fontsize=7)
        ax1.tick_params(colors=DARK, labelsize=7)

        recovery_pct = in_recovery.mean() * 100
        ax1.text(0.97, 0.95,
                 f'Recovery: {recovery_pct:.1f}% of lap',
                 transform=ax1.transAxes, ha='right', va='top',
                 color=DARK, fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=BG, edgecolor=LIGHT, alpha=0.8))
        ax1.set_title('HR & Recovery Windows', color=DARK,
                      fontsize=8, fontweight='bold')
        if col == 0:
            ax1.legend(fontsize=6, loc='upper left',
                       facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

        # ── Row 2: Stroke volume depression ──────────────────────────────────
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_facecolor(BG)
        for sp in ax2.spines.values():
            sp.set_edgecolor(LIGHT); sp.set_linewidth(0.4)

        ax2.fill_between(t_norm, sv, 100,
                         color='#3b82f6', alpha=0.25,
                         label='SV depression')
        ax2.plot(t_norm, sv, color='#3b82f6', linewidth=1.3)
        ax2.axhline(100, color=LIGHT, linewidth=0.5,
                    linestyle='--', alpha=0.6)
        ax2.axhline(80, color='#f59e0b', linewidth=0.6,
                    linestyle=':', alpha=0.6, label='−20% SV')
        ax2.axhline(70, color='#dc2626', linewidth=0.6,
                    linestyle=':', alpha=0.6, label='−30% SV')

        ax2.set_xlim(0, 100)
        ax2.set_ylim(55, 105)
        ax2.set_ylabel('Stroke Volume\n(% baseline)',
                       color=DARK, fontsize=7)
        ax2.set_xlabel('Lap progress (%)', color=DARK, fontsize=8)
        ax2.tick_params(colors=DARK, labelsize=7)

        min_sv      = sv.min()
        time_below80 = (sv < 80).mean() * 100
        ax2.text(0.97, 0.08,
                 f'Min SV: {min_sv:.1f}%\n<80%: {time_below80:.1f}% of lap',
                 transform=ax2.transAxes, ha='right', va='bottom',
                 color=DARK, fontsize=7,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=BG, edgecolor=LIGHT, alpha=0.8))
        ax2.set_title('Stroke Volume Depression', color=DARK,
                      fontsize=8, fontweight='bold')
        if col == 0:
            ax2.legend(fontsize=6, loc='upper left',
                       facecolor=BG, edgecolor=LIGHT, labelcolor=DARK)

    # ── Header & key finding ──────────────────────────────────────────────────
    fig.text(0.5, 0.945,
             'P7 — Hemodynamic Stress Index & Cardiac Recovery Analysis',
             ha='center', color=DARK, fontsize=15, fontweight='bold')
    fig.text(0.5, 0.918,
             'VER · 2023 Season · Lap 10 · '
             'Hemodynamic stress ref: Blomqvist & Stone 1983 '
             '(4.5g, conservative lower bound)  ·  '
             'SV model: Tripoli et al. 2024  ·  '
             'HR model: Tornaghi et al. 2023 + Tripoli et al. 2024',
             ha='center', color='#555555', fontsize=8)

    recovery_pcts = {
        name: (d['cv']['HR_proxy'].values <= 0.76*HR_MAX).mean()*100
        for name, d in circuit_data.items()
    }
    least = min(recovery_pcts, key=recovery_pcts.get)
    fig.text(0.5, 0.895,
             f'Key finding: {least} provides least cardiac recovery '
             f'({recovery_pcts[least]:.1f}% of lap at baseline HR)',
             ha='center', color=ACCENT, fontsize=9, style='italic')

    plt.savefig('outputs/p7_module3_gloc_recovery.png',
                dpi=180, bbox_inches='tight', facecolor=BG)
    plt.show()
    print("✅ Saved → outputs/p7_module3_gloc_recovery.png")


# ── Section 9: Main ───────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "═"*60)
    print("  P7 — F1 Driver Cardiovascular Demand Estimator")
    print("  VER · 2023 Season · Lap 10")
    print("═"*60 + "\n")

    # Fetch all circuit data
    circuit_data = fetch_circuit_data(
        circuits=CIRCUITS, driver_abbr='VER', target_lap=10)

    if not circuit_data:
        print("❌ No circuit data loaded. Check API connection.")
        exit(1)

    # Module 1 — single lap arc (Suzuka)
    if 'Suzuka' in circuit_data:
        print("\n── Module 1: Lap Cardiovascular Arc ──")
        plot_module1(circuit_data['Suzuka']['cv'],
                     lap_num=10, circuit='Suzuka',
                     driver='VER', year=2023)

    # Module 2 — circuit comparison
    print("\n── Module 2: Circuit Comparison ──")
    plot_module2(circuit_data)

    # Module 3 — Hemodynamic Stress Index & recovery
    print("\n── Module 3: Hemodynamic Stress Index & Recovery ──")
    plot_module3(circuit_data)

    # Summary findings
    print("\n" + "═"*60)
    print("  KEY FINDINGS")
    print("═"*60)
    for name, d in circuit_data.items():
        cv = d['cv']
        rec = (cv['HR_proxy'].values <= 0.76*HR_MAX).mean()*100
        min_sv = (1 - cv['SV_pct_baseline'].min()) * 100
        print(f"  {name:<12} | Mean HR {cv['HR_proxy'].mean():.0f} bpm "
              f"({cv['pct_HRmax'].mean()*100:.0f}%HRmax) | "
              f"Recovery {rec:.1f}% | Max SV drop {min_sv:.1f}%")
    print("═"*60)
    print("\n✅ All outputs saved to outputs/")
