"""QC parameters and pure gate functions for the external-mask backend.

PROVISIONAL THRESHOLDS. Derived from one development case (caso02): median
centerline attenuation <= -750 HU and at least 60% of sampled core points
< -600 HU. These thresholds have not been validated across acquisition
protocols or airway pathology and are provisional QC flags (ROC analysis on
annotated cases is part of the open validation program, see
docs/REVIEW_BACKLOG.md). The two criteria are partially redundant by design:
the median guards the central tendency, the fraction guards against bimodal
profiles (e.g. a branch half-filled with secretions).

This module has no side effects so tests exercise the SAME logic and
constants the pipeline uses.
"""
import numpy as np

HU_AIR = -750.0    # median HU of the core centerline must be below this
HU_SOFT = -600.0   # ...and at least AIR_FRAC of core points below this
AIR_FRAC = 0.60
ESCAPE_K = 1.5     # half-max escaped if d_mean > K * d_mask + C
ESCAPE_C = 0.5
VOXELS_FLOOR = 3.0
# central large airways (trachea, main bronchi, intermediate): the contour-
# escape gate is NOT applied here. External DL masks are known to UNDER-render
# these big airways, so a correct half-max lumen (~18 mm on the trachea)
# overshoots K*d_mask+C and would be wrongly demoted; on the central airways the
# half-max lumen is the trusted measurement, and a peripheral-style contour leak
# is not a plausible failure mode. Aid-based (never display strings).
CENTRAL_AIDS = frozenset({'TRACHEA', 'RMB', 'LMB', 'BI'})

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
    'qc', 'qc_misura', 'hu_lume', 'aria_pct', 'd_maschera_mm',
    'floor_calibro_mm',
    'diametro_raw_nonreportable', 'diametro_min_raw_nonreportable',
    'parete_raw_nonreportable', 'wa_raw_nonreportable',
]


def air_witness(hu):
    """True if the centerline HU profile shows a real air lumen."""
    hu = np.asarray(hu, dtype=float)
    if hu.size == 0:
        return False
    return bool((np.median(hu) <= HU_AIR)
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


def contour_escaped(d_mean, d_mask, aid=None):
    """True if the half-max diameter overshoots the mask diameter.

    Central large airways (aid in CENTRAL_AIDS) are exempt: the DL mask
    under-renders them, so a large half-max there is the trusted lumen, not a
    peripheral contour leak — demoting the trachea/main bronchi for 'escape'
    is a false positive."""
    if aid in CENTRAL_AIDS:
        return False
    return bool(d_mean is not None and d_mask is not None
                and d_mean > ESCAPE_K * d_mask + ESCAPE_C)


def csv_row(b):
    """One unified-schema row from a measured (and possibly witnessed) branch."""
    return [b['id'], b.get('name', ''), b['gen'], round(b['length'], 1),
            b.get('d_mean'), b.get('d_min'), b.get('wall'),
            b.get('wall_ok_pct'), b.get('wall_over_cap_pct'), b.get('wa_pct'),
            b.get('n_sez_tentate'), b.get('n_sez_valide'), b.get('metodo', ''),
            b.get('qc', ''), b.get('qc_misura', ''),
            b.get('hu_lume'), b.get('aria_pct'), b.get('d_maschera'),
            b.get('floor_mm'),
            b.get('d_mean_raw'), b.get('d_min_raw'),
            b.get('wall_raw'), b.get('wa_raw')]


def write_branches_csv(tree, path='out/branches.csv'):
    import csv
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for b in tree['branches']:
            w.writerow(csv_row(b))
