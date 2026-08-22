"""Per-branch measurements: lumen diameter (EDT) + wall thickness (FWHM rays).

Input:  out/ct_iso.nii.gz, out/airway_mask.nii.gz, out/tree.json
Output: out/branches.csv, out/tree_measured.json
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import json, csv
from lumen import analyze_section, pca_tangent, perp_basis
from section_metrics import mask_section, mask_ray_radii, overshoot_fraction

tree = json.load(open('out/tree.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
mask = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(bool)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    ct = ct[z0:z1, y0:y1, x0:x1].copy()
    mask = mask[z0:z1, y0:y1, x0:x1].copy()
edt = ndimage.distance_transform_edt(mask, sampling=(ISO, ISO, ISO))

import os as _os
_USE_F = _os.environ.get('AIRWAYLAB_REFINED') == '1'
_SH = np.array(edt.shape)

def branch_path(b):
    """Percorso + raggi EDT; usa il centerline raffinato (fpath) se attivo."""
    if _USE_F and b.get('fpath'):
        path = np.array(b['fpath'], dtype=float)
        ridx = np.clip(np.round(path).astype(int), 0, _SH - 1)
        radii = np.array([edt[tuple(q)] for q in ridx])
    else:
        path = pts[b['path']].astype(float)
        radii = np.array([edt[tuple(pts[p])] for p in b['path']])
    return path, radii

maskf = mask.astype(np.float32)

def cross_section_diam(center, t, r_est_mm):
    """Area-equivalent lumen diameter (mm) on the plane perpendicular to t —
    ora delega al modulo puro section_metrics (stessa logica: trilineare, soglia
    0.5, componente connessa che contiene il centro)."""
    ms = mask_section(maskf, center, t, ISO, r_est_mm)
    return ms['d_eq'] if ms else None

for b in tree['branches']:
    path, radii = branch_path(b)
    n = len(path)
    # exclude points near junctions (within local radius of the ends)
    margin = max(1, int(np.median(radii) / ISO))
    core = slice(margin, n - margin) if n > 2 * margin + 2 else slice(0, n)
    r_core = radii[core]
    if len(r_core) == 0:
        r_core = radii
    core_idx_all = (list(range(n))[core]) or list(range(n))
    # v0.23: NESSUNA preselezione dei punti (la scelta a raggio mediano
    # scartava stenosi e dilatazioni). Si misura su TUTTE le sezioni del
    # tratto centrale a passo ~1 mm e si riportano tentate/valide.
    step_pts = max(1, int(round(1.0 / ISO)))
    sample_idx = core_idx_all[::step_pts] or core_idx_all
    MAX_AX_RATIO = 1.45
    accepted, acc_walls, acc_wfrac, acc_overcap, oblique_seen = [], [], [], [], False
    mask_eqs, mask_mins, overs = [], [], []   # Step 1: audit maschera/half-max
    for i in sample_idx:
        t = pca_tangent(path, i)
        sec = analyze_section(ct, path[i], t, radii[i], ISO)
        if sec is None:
            continue
        if sec['quality']['ax_ratio'] > MAX_AX_RATIO:
            oblique_seen = True
            continue
        accepted.append(sec['d_eq'])
        if sec['wall_med'] is not None:
            acc_walls.append(sec['wall_med'])
            acc_overcap.append(sec['wall_over_cap_frac'])
        acc_wfrac.append(sec['wall_valid_frac'])
        # --- metriche di MASCHERA sulla STESSA sezione (ricentrata) ---
        cadj = (path[i] + (sec['shift'][0] / ISO) * sec['u']
                + (sec['shift'][1] / ISO) * sec['v'])
        ms = mask_section(maskf, cadj, t, ISO, radii[i])
        if ms:
            mask_eqs.append(ms['d_eq'])
            if ms['d_min'] is not None:
                mask_mins.append(ms['d_min'])
            rmax = min(30.0, max(8.0, radii[i] * 3 + 4))
            mr = mask_ray_radii(maskf, cadj, sec['u'], sec['v'], ISO, rmax)
            overs.append(overshoot_fraction(sec['inner'], mr))
    b['n_sez_tentate'] = len(sample_idx)
    b['n_sez_valide'] = len(accepted)
    success = len(accepted) / max(1, len(sample_idx))
    if b['length'] < 3.0:
        b['qc'] = 'moncone'
    elif len(accepted) >= 3 and success >= 0.4:
        b['qc'] = 'ok'
    elif len(accepted) >= 1:
        b['qc'] = 'poche-sezioni'
    elif oblique_seen:
        b['qc'] = 'obliquo'
    else:
        b['qc'] = 'no-parete'
    if accepted:
        d_mean = float(np.median(accepted))
        b['d_min_hm'] = round(float(np.min(accepted)), 2)
        b['d_max_hm'] = round(float(np.max(accepted)), 2)
        b['metodo'] = 'half-max'
    else:
        fb = [cross_section_diam(path[i], pca_tangent(path, i), radii[i])
              for i in core_idx_all[:3]]
        fb = [x for x in fb if x]
        d_mean = float(np.median(fb)) if fb else 2 * float(np.mean(r_core))
        b['metodo'] = 'maschera'
    d_min = b.get('d_min_hm') or 2 * float(np.min(r_core))
    d_max = b.get('d_max_hm') or 2 * float(np.max(r_core))

    # parete: dalle stesse sezioni accettate del lume (ricentrate, perpendicolari)
    wall = float(np.median(acc_walls)) if acc_walls else None
    b['wall_ok_pct'] = round(100 * float(np.mean(acc_wfrac)), 0) if acc_wfrac else None
    b['wall_over_cap_pct'] = round(100 * float(np.mean(acc_overcap)), 0) if acc_overcap else None

    b['d_mean'] = round(d_mean, 2)
    b['d_min'] = round(d_min, 2)
    b['wall'] = round(wall, 2) if wall else None
    # Step 1 (solo audit, gate invariato): calibro AREA-equivalente della
    # maschera (omogeneo col half-max) + larghezza minima + confronto radiale.
    b['d_mask_eq'] = round(float(np.median(mask_eqs)), 2) if mask_eqs else None
    b['d_mask_min'] = round(float(np.median(mask_mins)), 2) if mask_mins else None
    b['overshoot_frac'] = round(float(np.median(overs)), 3) if overs else None
    b['ct_mask_ratio'] = (round(d_mean / float(np.median(mask_eqs)), 2)
                          if mask_eqs and np.median(mask_eqs) > 0 else None)
    # derived indices when wall available
    if wall:
        r_out = d_mean / 2 + wall
        wa_pct = 100 * (r_out**2 - (d_mean / 2)**2) / r_out**2
        b['wa_pct'] = round(wa_pct, 1)
    else:
        b['wa_pct'] = None
json.dump(tree, open('out/tree_measured.json', 'w'))

# unified per-branch CSV: SAME schema with and without --mask (witness
# columns stay empty on built-in runs; witness.py rewrites the file)
from qc_params import write_branches_csv
write_branches_csv(tree)

# summary per generation
import collections
by_gen = collections.defaultdict(list)
for b in tree['branches']:
    by_gen[b['gen']].append(b)
print(f"{'gen':>3} {'n':>3} {'diam medio':>10} {'parete':>7}")
for g in sorted(by_gen):
    ds = [b['d_mean'] for b in by_gen[g] if b.get('qc') == 'ok'] or [b['d_mean'] for b in by_gen[g]]
    ws = [b['wall'] for b in by_gen[g] if b['wall']]
    print(f"{g:>3} {len(by_gen[g]):>3} {np.mean(ds):>9.1f}mm "
          f"{(str(round(float(np.mean(ws)),2))+'mm') if ws else '   -':>7}")
