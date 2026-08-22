"""Build the interactive 3D airway map (HTML) from segmentation + measurements."""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.measure import marching_cubes
import json

tree = json.load(open('out/tree_measured.json'))
info = json.load(open('out/seg_info.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

mask = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(np.float32)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    mask = mask[z0:z1, y0:y1, x0:x1].copy()

# ---------- surface mesh (downsampled + smoothed) ----------
sm = ndimage.gaussian_filter(mask[::2, ::2, ::2], sigma=0.8)
verts, faces, _, _ = marching_cubes(sm, level=0.35, spacing=(ISO * 2, ISO * 2, ISO * 2))
print('mesh:', len(verts), 'verts,', len(faces), 'faces')

# voxel (z,y,x) mm -> display (x=LR, y=AP, z=CC)
def to_xyz(a_zyx_mm):
    return a_zyx_mm[:, 2], a_zyx_mm[:, 1], a_zyx_mm[:, 0]

mx, my, mz = to_xyz(verts)

# ---------- branch polylines ----------
import os as _os
_USE_F = _os.environ.get('AIRWAYLAB_REFINED') == '1'

def _smooth_polyline(p, iters=3, w=2):
    """Levigatura cosmetica (solo display) quando il centerline raffinato
    non e' disponibile: media mobile iterata, estremi fissi."""
    if len(p) < 5:
        return p
    q = p.copy()
    for _ in range(iters):
        acc = q.copy()
        for k in range(1, w + 1):
            acc[k:-k] = acc[k:-k] + q[2 * k:] + q[:-2 * k]
        counts = np.ones(len(q))
        for k in range(1, w + 1):
            counts[k:-k] += 2
        q = acc / counts[:, None]
        q[0], q[-1] = p[0], p[-1]
    return q

branches = []
for b in tree['branches']:
    if _USE_F and b.get('fpath'):
        p = np.array(b['fpath'], dtype=float) * ISO      # spline: gia' liscia
    else:
        p = _smooth_polyline(pts[b['path']].astype(float) * ISO)
    bx, by, bz = p[:, 2], p[:, 1], p[:, 0]
    branches.append({
        'id': b['id'], 'name': b['name'], 'gen': b['gen'],
        'len': round(b['length'], 1), 'd': b['d_mean'], 'dmin': b['d_min'],
        'wall': b['wall'], 'wa': b['wa_pct'],
        'qc': b.get('qc', 'ok'), 'metodo': b.get('metodo', ''),
        'wok': b.get('wall_ok_pct'),
        'x': [round(v, 1) for v in bx],
        'y': [round(v, 1) for v in by],
        'z': [round(v, 1) for v in bz],
    })

import os
snaps = json.load(open('out/snaps.json')) if os.path.exists('out/snaps.json') else {}
profs = json.load(open('out/profiles.json')) if os.path.exists('out/profiles.json') else {}
terrs = json.load(open('out/territories.json')) if os.path.exists('out/territories.json') else {}
pairs = json.load(open('out/pairing.json')) if os.path.exists('out/pairing.json') else {}
for b in branches:
    if b['id'] in snaps:
        b['snap'] = snaps[b['id']]
    if b['id'] in profs:
        b['prof'] = profs[b['id']]
    if b['id'] in terrs:
        b['terr'] = terrs[b['id']]
    if b['id'] in pairs:
        b['vd'] = pairs[b['id']]['vd']
        b['ba'] = pairs[b['id']]['ba']

plugs = json.load(open('out/plugs.json')) if os.path.exists('out/plugs.json') else []

data = {
    'plugs': plugs,
    'mesh': {
        'x': [round(float(v), 1) for v in mx],
        'y': [round(float(v), 1) for v in my],
        'z': [round(float(v), 1) for v in mz],
        'i': faces[:, 0].tolist(), 'j': faces[:, 1].tolist(), 'k': faces[:, 2].tolist(),
    },
    'branches': branches,
    'meta': {
        'volume_ml': round(info['volume_ml'], 1),
        'threshold': info['threshold'],
        **{k: info.get(k) for k in __import__('qc_params').PROVENANCE_KEYS},
        'n_branches': len(branches),
        'gen_max': tree['gen_max'],
        'trachea_d': next((b['d'] for b in branches if b['name'] == 'trachea'), None),
        'lung': json.load(open('out/lung_metrics.json')) if os.path.exists('out/lung_metrics.json') else None,
        'murray': json.load(open('out/murray.json')) if os.path.exists('out/murray.json') else None,
        'vessels': json.load(open('out/vessel_metrics.json')) if os.path.exists('out/vessel_metrics.json') else None,
        'dual': json.load(open('out/dual_metrics.json')) if os.path.exists('out/dual_metrics.json') else None,
        'cpr': json.load(open('out/cpr.json')) if os.path.exists('out/cpr.json') else None,
        'dual_png': open('out/dual_b64.txt').read() if os.path.exists('out/dual_b64.txt') else None,
    },
}
json.dump(data, open('out/map_data.json', 'w'))
print('meta:', data['meta'])
