"""QC parameters and pure gate functions for the external-mask backend.

PROVISIONAL THRESHOLDS. Derived from one development case (caso02): median
centerline attenuation <= -750 HU and at least 60% of sampled core points
< -600 HU. These thresholds have not been validated across acquisition
protocols or airway pathology and are provisional QC flags (ROC analysis on
annotated cases is part of the open validation program, see
docs/VALIDATION_BACKLOG.md). The two criteria are partially redundant by design:
the median guards the central tendency, the fraction guards against bimodal
profiles (e.g. a branch half-filled with secretions).

This module has no side effects so tests exercise the SAME logic and
constants the pipeline uses.
"""
import math

import numpy as np

HU_AIR = -750.0    # median HU of the core centerline must be below this
HU_SOFT = -600.0   # ...and at least AIR_FRAC of core points below this
AIR_FRAC = 0.60

# --- partial-volume correction of the median criterion (backlog #29) ---
# A fixed HU_AIR is blind to caliber, and partial-volume mixing guarantees that
# a genuine small airway reads warmer than a large one at the same resolution.
# Measured on a real thin-slice case: branches rejected as 'no-lume' had median
# mask diameter 1.40 mm and median lumen attenuation -680 HU, against 2.80 mm
# and -990 HU for the accepted ones — i.e. the rejected set was the SMALLEST in
# the tree, failing on physics rather than on anatomy.
#
# Model: a lumen of diameter d surrounded by soft tissue, blurred by a Gaussian
# of sigma = PVE_SIGMA_VOXELS * spacing, reads at its centre
#     HU_WALL + (HU_LUMEN_AIR - HU_WALL) * erf(d / (2*sigma*sqrt(2)))
# The threshold follows that curve plus a noise margin, so it relaxes by exactly
# what partial volume can explain and NOT beyond: a lumen warmer than the model
# predicts still fails, which is what keeps a generous mask from inventing
# airways. Bounded on both sides — never stricter than HU_AIR (large airways
# keep the historical gate) and never above HU_AIR_CEILING (nothing that warm is
# air, whatever its size).
#
# Measured effect on that case: 9 of the 17 rejected branches are recovered. Of
# the 8 that stay out, 1 is still too warm for the model and 7 fail the
# air-FRACTION criterion, which is left uncorrected on purpose (its median over
# the group was 67% against the 60% required, but it binds on 7 of 17 branches
# individually). That criterion therefore now does most of the rejecting, which
# is the intended conservative outcome: the gate still discriminates.
#
# PROVISIONAL, like the rest of this module: the sigma factor depends on kernel
# and reconstruction and is NOT calibrated. Backlog #29 tracks the ROC work; the
# per-branch effective threshold is exported (`soglia_aria_hu`) so every
# accept/reject decision can be audited against the model that produced it.
HU_LUMEN_AIR = -1000.0    # pure air in the lumen
HU_WALL_SOFT = 0.0        # soft tissue surrounding it
PVE_SIGMA_VOXELS = 1.0    # blur sigma, in units of the effective spacing
AIR_MARGIN_HU = 150.0     # tolerance for noise and biological variation
HU_AIR_CEILING = -400.0   # hard ceiling: never more permissive than this
ESCAPE_K = 1.5     # half-max escaped if d_mean > K * d_mask + C
ESCAPE_C = 0.5
VOXELS_FLOOR = 3.0
# central large airways exempt from HARD contour-escape demotion. Single source
# of truth: the normative central set already defined in anatomy.py (the ALR4
# airways). Deriving it here — instead of a second literal list — keeps the two
# from diverging. External DL masks tend to UNDER-render these big airways, so a
# large half-max there is usually the mask under-filling, not a contour leak;
# but a real escape (patologia grave, contatto con esofago/vasi/mediastino,
# sezione obliqua, centerline decentrata, stenosi) is NOT impossible centrally.
# Policy: on these AIDs the endpoint is preserved but the discordance is recorded
# as an audit soft-flag (escape_gate_exempt / qc_note), so a central overshoot is
# never indistinguishable from a mask-concordant measurement — it stays flagged
# for visual verification. Decided in witness.py, not silently inside the gate.
from anatomy import ALR4_AIDS as CENTRAL_AIDS   # noqa: E402
ESCAPE_EXEMPT_NOTE = 'central-mask-underfill'

INVALID_QC = ('no-lume', 'sotto-risoluzione', 'fuga-contorno')

# unified per-branch CSV schema — IDENTICAL with and without --mask
# (witness columns stay empty on built-in segmentation runs).
# Audit model: reportable endpoints keep plain names; demoted values move to
# explicit *_raw_nonreportable columns — never mix the two in analysis.
CSV_COLUMNS = [
    'branch_id', 'nome', 'generazione', 'lunghezza_mm',
    'diametro_medio_mm', 'diametro_min_mm', 'spessore_parete_mm',
    'parete_settori_validi_pct', 'parete_oltre_cap_pct', 'wall_area_pct',
    'n_sezioni_tentate', 'n_sezioni_valide', 'metodo_diametro',
    'qc', 'qc_misura', 'qc_note', 'hu_lume', 'aria_pct', 'd_maschera_mm',
    'floor_calibro_mm',
    # Step 1 (audit): metriche di maschera sulla sezione perpendicolare.
    # d_maschera_edt (EDT, ~semiasse minore) resta come 'd_maschera_mm' sopra;
    # queste sono OMOGENEE col half-max (eq) e la dimensione minima nel piano.
    'd_maschera_eq_mm', 'd_maschera_min_mm', 'd_maschera_maj_mm',
    'aspect_mask', 'ct_mask_ratio', 'overshoot_frac',
    'soglia_aria_hu',
    'n_sez_paired_diam', 'n_sez_paired_radial',
    'diametro_raw_nonreportable', 'diametro_min_raw_nonreportable',
    'parete_raw_nonreportable', 'wa_raw_nonreportable',
]


def air_threshold_hu(d_mm=None, spacing_mm=None):
    """Median-attenuation threshold expected for a lumen of diameter `d_mm`
    sampled at `spacing_mm` (see the partial-volume note at the top).

    Returns HU_AIR — the historical fixed gate — whenever the caliber or the
    spacing is unknown or non-positive, so callers that cannot supply them keep
    the previous behaviour exactly.

    Pure. Monotone in d: smaller airways get a warmer (more permissive)
    threshold, large ones converge back onto HU_AIR."""
    try:
        d = float(d_mm)
        s = float(spacing_mm)
    except (TypeError, ValueError):
        return HU_AIR
    if not (d > 0 and s > 0):
        return HU_AIR
    sigma = PVE_SIGMA_VOXELS * s
    frac_air = math.erf(d / (2.0 * sigma * math.sqrt(2.0)))
    expected = HU_WALL_SOFT + (HU_LUMEN_AIR - HU_WALL_SOFT) * frac_air
    thr = expected + AIR_MARGIN_HU
    thr = min(thr, HU_AIR_CEILING)   # mai piu' permissivo del tetto
    return max(thr, HU_AIR)          # mai piu' severo del gate storico


def air_witness(hu, d_mm=None, spacing_mm=None):
    """True if the centerline HU profile shows a real air lumen.

    With `d_mm` and `spacing_mm` the median criterion is corrected for partial
    volume; without them it falls back to the fixed HU_AIR gate. The
    air-FRACTION criterion is deliberately left uncorrected: partial volume is
    a statement about the CENTRAL tendency of the profile, not about how many
    samples clear a second, warmer cut, and relaxing both would weaken the gate
    beyond what the physics justifies."""
    hu = np.asarray(hu, dtype=float)
    if hu.size == 0:
        return False
    thr = air_threshold_hu(d_mm, spacing_mm)
    return bool((np.median(hu) <= thr)
                and (float((hu < HU_SOFT).mean()) >= AIR_FRAC))


def resolution_floor_mm(iso, native_spacing=None):
    """CONSERVATIVE caliber floor in mm (no orientation information):
    VOXELS_FLOOR x the coarsest of ALL native spacings (slice thickness
    included) and the processing ISO. Anchored to the ACQUIRED spacing so
    upsampling cannot loosen it; a coarser processing grid raises it.

    Use orientation_floor_mm when the branch axis is known — it relaxes this
    bound only in the directions the section plane actually samples finely.
    The floor remains a provisional processing bound (three voxels, not a
    measured scanner PSF) and must not be interpreted as a validated
    physical resolution limit.
    """
    base = float(iso)
    if native_spacing:
        try:
            base = max(base, max(float(s) for s in native_spacing))
        except (TypeError, ValueError):
            pass
    return VOXELS_FLOOR * base


def orientation_floor_mm(iso, native_spacing, tangent_zyx=None):
    """Per-section, orientation-dependent caliber floor in mm.

    The measurement section is perpendicular to the axis t; the governing
    sampling step is the WORST effective spacing over all unit directions u
    lying in that plane. With the anisotropic sampling metric
    M = diag(s_z^2, s_y^2, s_x^2), the effective spacing along u is
    sqrt(u' M u); its maximum over the plane u ⊥ t is the largest
    eigenvalue of the projected metric P M P (P = I - t t').
    NOTE: this is NOT max_i(s_i * sqrt(1 - t_i^2)) — for oblique planes the
    worst direction mixes axes and the per-axis projection underestimates
    (e.g. 45 deg between z and y on 0.72/0.72/1.25 mm: 1.022 mm, not 0.884).
    Never below the processing ISO; conservative worst-axis fallback when
    tangent or native spacing are unavailable.
    """
    if not native_spacing or tangent_zyx is None:
        return resolution_floor_mm(iso, native_spacing)
    try:
        s_zyx = np.array([float(native_spacing[2]), float(native_spacing[1]),
                          float(native_spacing[0])])
        t = np.asarray(tangent_zyx, dtype=float)
        n = np.linalg.norm(t)
        if n < 1e-6:
            return resolution_floor_mm(iso, native_spacing)
        t = t / n
        M = np.diag(s_zyx ** 2)
        P = np.eye(3) - np.outer(t, t)
        eff = float(np.sqrt(max(0.0, np.linalg.eigvalsh(P @ M @ P).max())))
        return VOXELS_FLOOR * max(float(iso), eff)
    except (TypeError, ValueError, IndexError):
        return resolution_floor_mm(iso, native_spacing)


def branch_floor_mm(iso, native_spacing, points_zyx, max_samples=24):
    """Branch-level caliber floor: the branch summary endpoint aggregates
    sections with different local orientations, so the floor is the MAX of
    the per-section floors computed on LOCAL tangents (central differences
    over a small window) sampled along the path — a curved branch that
    turns in-plane is floored by its worst-oriented segment, not by the
    global chord."""
    P = np.asarray(points_zyx, dtype=float)
    if len(P) < 2:
        return resolution_floor_mm(iso, native_spacing)
    if len(P) == 2:
        return orientation_floor_mm(iso, native_spacing, P[1] - P[0])
    interior = np.arange(1, len(P) - 1)
    if len(interior) > max_samples:
        interior = np.unique(np.linspace(1, len(P) - 2, max_samples).astype(int))
    floors = []
    for i in interior:
        a, b = max(0, i - 3), min(len(P) - 1, i + 3)
        floors.append(orientation_floor_mm(iso, native_spacing, P[b] - P[a]))
    return max(floors)


def split_reportable(value, reportable):
    """Two-regime invariant for one sample, enforced by construction:
    returns (clinical, raw_nonreportable) — exactly one side carries the
    value, the other is None."""
    return (value, None) if reportable else (None, value)


# provenance fields every run must expose (seg_info.json AND map_data.meta)
PROVENANCE_KEYS = ('backend', 'refined_centerline', 'tight_small_window',
                   'airwaylab_version')


def contour_escaped(d_mean, d_mask):
    """True if the half-max diameter overshoots the mask diameter (pure
    geometric test). The central-airway EXEMPTION is applied by the caller
    (witness.py), which preserves the endpoint but records an audit flag — the
    gate itself stays a single, aid-agnostic geometric criterion."""
    return bool(d_mean is not None and d_mask is not None
                and d_mean > ESCAPE_K * d_mask + ESCAPE_C)


def escape_decision(d_mean, d_mask, aid):
    """Policy applied by witness.py when the (pure) contour-escape gate fires.

    Returns (demote, note):
      - gate NOT fired            -> (False, '')      endpoint kept, no note
      - fired, peripheral branch  -> (True,  '')      HARD demotion 'fuga-contorno'
      - fired, central airway     -> (False, NOTE)    endpoint KEPT but flagged
                                                      (central-mask-underfill),
                                                      for visual verification

    So a central overshoot is preserved yet remains an explicit audit event —
    never indistinguishable from a measurement concordant with the mask."""
    if not contour_escaped(d_mean, d_mask):
        return (False, '')
    if aid in CENTRAL_AIDS:
        return (False, ESCAPE_EXEMPT_NOTE)
    return (True, '')


def csv_row(b):
    """One unified-schema row from a measured (and possibly witnessed) branch."""
    return [b['id'], b.get('name', ''), b['gen'], round(b['length'], 1),
            b.get('d_mean'), b.get('d_min'), b.get('wall'),
            b.get('wall_ok_pct'), b.get('wall_over_cap_pct'), b.get('wa_pct'),
            b.get('n_sez_tentate'), b.get('n_sez_valide'), b.get('metodo', ''),
            b.get('qc', ''), b.get('qc_misura', ''), b.get('qc_note', ''),
            b.get('hu_lume'), b.get('aria_pct'), b.get('d_maschera'),
            b.get('floor_mm'),
            b.get('d_mask_eq'), b.get('d_mask_min'), b.get('d_mask_maj'),
            b.get('aspect_mask'), b.get('ct_mask_ratio'), b.get('overshoot_frac'),
            b.get('soglia_aria_hu'),
            b.get('n_sez_paired_diam'), b.get('n_sez_paired_radial'),
            b.get('d_mean_raw'), b.get('d_min_raw'),
            b.get('wall_raw'), b.get('wa_raw')]


def write_branches_csv(tree, path='out/branches.csv'):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for b in tree['branches']:
            w.writerow(csv_row(b))
