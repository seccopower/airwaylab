"""Profilo longitudinale: calibro (e parete) ogni ~1 mm lungo ogni ramo.

Per ogni ramo campiona sezioni perpendicolari al centerline a passo ~1 mm e
registra diametro half-max e spessore di parete. I punti entro il "cono di
giunzione" alle estremita' (un raggio locale) sono esclusi. Output:
out/profiles.json  {branch_id: {"s": [mm], "d": [mm|null], "w": [mm|null]}}
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import json
from lumen import analyze_section, pca_tangent

tree = json.load(open('out/tree_measured.json'))
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

STEP_ALONG_MM = 1.0
profiles = {}
for b in tree['branches']:
    if _USE_F and b.get('fpath'):
        path = np.array(b['fpath'], dtype=float)
        ridx = np.clip(np.round(path).astype(int), 0, _SH - 1)
        radii = np.array([edt[tuple(q)] for q in ridx])
    else:
        path = pts[b['path']].astype(float)
        radii = np.array([edt[tuple(pts[p])] for p in b['path']])
    n = len(path)
    # arclength cumulativa (mm)
    seg = np.linalg.norm(np.diff(path, axis=0) * ISO, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = arc[-1]
    # margini di giunzione: un raggio locale per estremita'
    m0 = float(radii[0])
    m1 = float(radii[-1])
    s_vals, d_vals, w_vals = [], [], []
    s = 0.0
    while s <= total:
        if m0 <= s <= total - m1:
            i = int(np.searchsorted(arc, s))
            i = min(i, n - 1)
            sec = analyze_section(ct, path[i], pca_tangent(path, i), radii[i], ISO)
            if sec is not None and sec['quality']['ax_ratio'] <= 1.8:
                d = round(sec['d_eq'], 2)
                w = round(sec['wall_med'], 2) if sec['wall_med'] is not None else None
            else:
                d, w = None, None
            s_vals.append(round(s, 1))
            d_vals.append(d)
            w_vals.append(w)
        s += STEP_ALONG_MM
    if s_vals:
        # two-regime rule ENFORCED: per i rami declassati dal witness il
        # canale clinico (d, w) e' nullo; le curve grezze restano disponibili
        # per l'audit solo sotto nomi espliciti *_raw_nonreportable
        from qc_params import INVALID_QC
        if b.get('qc') in INVALID_QC:
            prof = {'s': s_vals,
                    'd': [None] * len(s_vals), 'w': [None] * len(s_vals),
                    'd_raw_nonreportable': d_vals,
                    'w_raw_nonreportable': w_vals,
                    'nonreportable': b['qc']}
        else:
            prof = {'s': s_vals, 'd': d_vals, 'w': w_vals}
        profiles[b['id']] = prof

json.dump(profiles, open('out/profiles.json', 'w'))
npts = sum(len(p['s']) for p in profiles.values())
nval = sum(sum(1 for d in p['d'] if d is not None) for p in profiles.values())
print(f'{len(profiles)} rami, {npts} sezioni campionate, {nval} valide ({100*nval/max(1,npts):.0f}%)')
