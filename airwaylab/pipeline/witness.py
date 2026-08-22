"""Air witness + resolution gate + contour-escape gate (two-regime rule).

Rationale (from the DL-backend experiment on caso02): a generous external
mask finds real airways far beyond what caliber measurement can support —
the mask paints small airways at a near-constant minimum width, and below
the resolution floor the half-max boundary carries no size information.
So every branch is checked against the CT itself and demoted, not deleted:

  no-lume            no visible air along the centerline — the only class
                     that questions the branch's existence.
  sotto-risoluzione  real branch (air confirmed) but mask diameter below the
                     resolution floor: counted, mapped, used for territories,
                     NO caliber reported.
  fuga-contorno      half-max diameter escaped far beyond the mask diameter:
                     the measured value is not reportable.

ENFORCEMENT (v0.25): for demoted branches the reportable endpoints
(d_mean, d_min, wall, wa_pct, d_min_hm, d_max_hm) are set to None; the
original values are preserved in *_raw audit fields and exported in the
explicit `*_raw_nonreportable` CSV columns. Downstream aggregations already
select on qc == 'ok', so nulling is belt and braces.

Thresholds are PROVISIONAL (single development case) — see qc_params.py.
HU and EDT are sampled tri-linearly at the (sub-voxel) centerline
coordinates, so the flags do not depend on the rounding phase of the grid.

Run after measure.py, only with the external-mask backend.
Rewrites out/tree_measured.json and out/branches.csv (unified schema).
"""
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from qc_params import (air_witness, branch_floor_mm, escape_decision,
                       resolution_floor_mm, write_branches_csv)

tree = json.load(open('out/tree_measured.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

info = json.load(open('out/seg_info.json')) if os.path.exists('out/seg_info.json') else {}
NATIVE = info.get('native_spacing')
D_MIN_CONS = resolution_floor_mm(ISO, NATIVE)   # conservative (no orientation)

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
mask = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(bool)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    ct = ct[z0:z1, y0:y1, x0:x1]
    mask = mask[z0:z1, y0:y1, x0:x1]
edt = ndimage.distance_transform_edt(mask, sampling=(ISO,) * 3)

NULLED = (('d_mean', 'd_mean_raw'), ('d_min', 'd_min_raw'),
          ('wall', 'wall_raw'), ('wa_pct', 'wa_raw'),
          ('d_min_hm', 'd_min_hm_raw'), ('d_max_hm', 'd_max_hm_raw'))

surv = {}   # gen -> [measurable, sub-res, no-lume, escape]
for b in tree['branches']:
    P = (np.array(b['fpath'], dtype=float) if b.get('fpath')
         else pts[b['path']].astype(float))
    n = len(P)
    m = max(1, n // 6)                       # trim junction ends
    core = P[m:n - m] if n > 2 * m + 2 else P
    # tri-linear sampling at float coordinates: no grid-phase dependence
    hu = ndimage.map_coordinates(ct, core.T, order=1, mode='nearest')
    radii = ndimage.map_coordinates(edt, core.T, order=1, mode='nearest')
    d_mask = float(2.0 * np.median(radii))
    b['hu_lume'] = round(float(np.median(hu)), 1)
    b['aria_pct'] = round(100 * float((hu < -600).mean()))
    b['d_maschera'] = round(d_mask, 2)
    b['qc_misura'] = b.get('qc', '')
    # orientation-dependent floor aggregated over LOCAL tangents along the
    # branch (worst projected-metric direction in each section plane) — a
    # curved branch is floored by its worst-oriented segment, not the chord
    D_MIN = branch_floor_mm(ISO, NATIVE, core)
    b['floor_mm'] = round(D_MIN, 2)

    if not air_witness(hu):
        b['qc'] = 'no-lume'
        tier = 2
    elif d_mask < D_MIN:
        b['qc'] = 'sotto-risoluzione'
        tier = 1
    else:
        # gate di fuga-contorno (puro geometrico) + policy centrale, in una
        # funzione pura testabile: le vie centrali esenti mantengono l'endpoint
        # ma vengono flaggate per audit; la periferia resta hard-demotion.
        demote, note = escape_decision(b.get('d_mean'), d_mask, b.get('aid'))
        if demote:
            b['qc'] = 'fuga-contorno'
            tier = 3
        else:
            if note:
                b['escape_gate_exempt'] = True
                b['qc_note'] = note
            tier = 0                          # keep original measurement qc

    if tier != 0:
        # two-regime enforcement: demoted branches report NO caliber/wall
        for key, raw in NULLED:
            if b.get(key) is not None:
                if raw:
                    b[raw] = b[key]
                b[key] = None
    surv.setdefault(b['gen'], [0, 0, 0, 0])[tier] += 1

json.dump(tree, open('out/tree_measured.json', 'w'))
write_branches_csv(tree)

floors = [b['floor_mm'] for b in tree['branches'] if b.get('floor_mm')]
tot = [0, 0, 0, 0]
print(f'floor calibro orientamento-dipendente: '
      f'{min(floors):.2f}-{max(floors):.2f} mm (conservativo: {D_MIN_CONS:.2f})')
print('gen  measurable  sub-res  no-lume  escape')
for g in sorted(surv):
    k, s, nl, fu = surv[g]
    tot = [tot[0] + k, tot[1] + s, tot[2] + nl, tot[3] + fu]
    print(f'{g:3d}  {k:10d}  {s:7d}  {nl:7d}  {fu:6d}')
print(f'TOT  {tot[0]:10d}  {tot[1]:7d}  {tot[2]:7d}  {tot[3]:6d}')
