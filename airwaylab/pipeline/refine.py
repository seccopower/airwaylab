"""Sub-voxel centerline refinement + B-spline smoothing.

The morphological skeleton lives on the voxel grid: oblique branches become
staircases and in wide lumens the skeleton wanders around the true axis
(caso02: raw trachea path 136.9 mm vs a 115.2 mm chord, +19% of pure zig-zag).
This step re-centers every interior path point on the lumen centroid in its
perpendicular plane (tri-linear mask sampling — sub-voxel, not grid-bound),
replaces the polyline with a smoothing B-spline (junction endpoints pinned so
the tree stays connected), and recomputes branch length on the smooth curve.

Downstream steps (measure/snapshots/profile/viz) use the refined path only
when AIRWAYLAB_REFINED=1; the raw voxel path is kept alongside.

Input:  out/tree.json (after labels.py), out/airway_mask.nii.gz
Output: out/tree.json with per-branch fpath (float zyx voxel coords),
        length (spline arc length, mm) and length_raw (voxel path length, mm)
"""
import json

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import splev, splprep

from lumen import pca_tangent, perp_basis

tree = json.load(open('out/tree.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

mask = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(np.float32)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    mask = mask[z0:z1, y0:y1, x0:x1].copy()
edt = ndimage.distance_transform_edt(mask > 0.5, sampling=(ISO,) * 3)
shape = np.array(mask.shape)


def recenter(P, radii_mm):
    """One pass of in-plane sub-voxel recentering on the lumen centroid."""
    Q = P.copy()
    for i in range(1, len(P) - 1):
        t = pca_tangent(P, i)
        u, v = perp_basis(t)
        r = float(np.clip(radii_mm[i] * 1.5, 1.5, 4.0)) / ISO   # window (voxels)
        ax = np.arange(-r, r + 1e-6, 0.5)
        U, V = np.meshgrid(ax, ax)
        grid = P[i][None, :] + U.reshape(-1, 1) * u[None, :] + V.reshape(-1, 1) * v[None, :]
        w = ndimage.map_coordinates(mask, grid.T, order=1, mode='constant', cval=0.0)
        if w.sum() < 1e-3:
            continue
        cu = float((w * U.reshape(-1)).sum() / w.sum())
        cv = float((w * V.reshape(-1)).sum() / w.sum())
        shift = cu * u + cv * v
        # cap the in-plane shift at the local radius (no jumps to neighbours)
        m = np.linalg.norm(shift) * ISO
        cap = max(0.3, radii_mm[i])
        if m > cap:
            shift *= cap / m
        Q[i] = P[i] + shift
    return Q


n_ref = 0
for b in tree['branches']:
    P = pts[b['path']].astype(float)
    b['length_raw'] = b['length']
    if len(P) < 5:
        b['fpath'] = [[round(float(c), 3) for c in p] for p in P]
        continue
    ridx = np.clip(np.round(P).astype(int), 0, shape - 1)
    radii_mm = np.array([edt[tuple(q)] for q in ridx])

    Q = recenter(P, radii_mm)
    Q = recenter(Q, radii_mm)
    Q[0], Q[-1] = P[0], P[-1]

    try:
        k = 3 if len(Q) > 4 else max(1, len(Q) - 1)
        wgt = np.ones(len(Q))
        wgt[0] = wgt[-1] = 50.0                             # pin the junctions
        tck, _ = splprep(Q.T, k=k, s=len(Q) * 0.25, w=wgt)
        arc = np.linalg.norm(np.diff(Q, axis=0), axis=1).sum()
        n_new = max(5, int(round(arc)))                     # ~1 voxel per point
        S = np.array(splev(np.linspace(0, 1, n_new), tck)).T
        S[0], S[-1] = P[0], P[-1]
    except Exception:
        S = Q

    b['fpath'] = [[round(float(c), 3) for c in p] for p in S]
    b['length'] = float(np.linalg.norm(np.diff(S, axis=0) * ISO, axis=1).sum())
    n_ref += 1

json.dump(tree, open('out/tree.json', 'w'))
lo = sum(b['length_raw'] for b in tree['branches'])
ln = sum(b['length'] for b in tree['branches'])
print(f'{n_ref} branches refined · total tree length {lo:.0f} -> {ln:.0f} mm '
      f'({100 * (ln / max(1e-9, lo) - 1):+.1f}%)')
