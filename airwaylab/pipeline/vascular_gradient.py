"""Gradiente di pruning vascolare a livello di soggetto (compute standalone).

Legge out/vessel_mask.nii.gz e out/lung_mask_ds.nii.gz: densita' di piccoli vasi
(sezione < 5 mm^2, stessa apertura r5 di vessels.py) in funzione della distanza
dalla pleura. Scrive out/vascular_gradient.json. Gira dopo vessels.py + lung.py.
"""
import json
import math
import os

import numpy as np
from scipy import ndimage

from provenance import provenance
from vascular_gradient_core import pruning_summary

VES = 'out/vessel_mask.nii.gz'
LUNG = 'out/lung_mask_ds.nii.gz'
INFO = 'out/seg_info.json'

if not (os.path.exists(VES) and os.path.exists(LUNG)):
    print('vascular_gradient: mancano vessel_mask / lung_mask_ds — salto')
    raise SystemExit(0)

import SimpleITK as sitk

iso = json.load(open(INFO)).get('iso', 0.625) if os.path.exists(INFO) else 0.625
iso_ds = iso * 3
r5 = math.sqrt(5 / math.pi)
vox_ml = iso ** 3 / 1000.0
vox_ml_ds = iso_ds ** 3 / 1000.0

ves = sitk.GetArrayFromImage(sitk.ReadImage(VES)).astype(bool)
lung_ds = sitk.GetArrayFromImage(sitk.ReadImage(LUNG)).astype(bool)
if not ves.any() or not lung_ds.any():
    print('vascular_gradient: maschere vuote — salto')
    raise SystemExit(0)

# distanza dalla pleura (mm) sul reticolo ×3
dist_ds = ndimage.distance_transform_edt(lung_ds, sampling=(iso_ds,) * 3).astype(np.float32)

# piccoli vasi a piena risoluzione (fedele a BV5: apertura r5 sul ritaglio)
nz = np.nonzero(ves)
bb = [(int(a.min()), int(a.max()) + 1) for a in nz]
vc = ves[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
edt_in = ndimage.distance_transform_edt(vc, sampling=(iso,) * 3).astype(np.float32)
core = edt_in >= r5
del edt_in
if core.any():
    edt_out = ndimage.distance_transform_edt(~core, sampling=(iso,) * 3).astype(np.float32)
    opened = (edt_out <= r5) & vc
    del edt_out
else:
    opened = np.zeros_like(vc)
small = vc & ~opened
del core, opened

# coordinate dei voxel-vaso (frame completo) -> cella ×3 -> distanza + in-polmone
zc, yc, xc = np.nonzero(vc)
z = zc + bb[0][0]
y = yc + bb[1][0]
x = xc + bb[2][0]
dz, dy, dx = z // 3, y // 3, x // 3
ok = (dz < dist_ds.shape[0]) & (dy < dist_ds.shape[1]) & (dx < dist_ds.shape[2])
d = np.zeros(zc.shape, np.float32)
d[ok] = dist_ds[dz[ok], dy[ok], dx[ok]]
smallflag = small[zc, yc, xc]
inlung = d > 0
d = d[inlung]
smallflag = smallflag[inlung]

# distanze dei voxel di polmone (denominatore di densita')
dl = dist_ds[lung_ds]

EDGES = [0, 5, 10, 15, 20, 25, 30, 1e9]
shells = []
for lo, hi in zip(EDGES, EDGES[1:]):
    center = (lo + hi) / 2 if hi < 1e8 else lo + 2.5
    lung_ml = float(((dl >= lo) & (dl < hi)).sum()) * vox_ml_ds
    allmask = (d >= lo) & (d < hi)
    all_ml = float(allmask.sum()) * vox_ml
    small_ml = float((allmask & smallflag).sum()) * vox_ml
    shells.append({'d_center': center, 'lung_ml': round(lung_ml, 1),
                   'small_ml': round(small_ml, 2), 'all_ml': round(all_ml, 2)})

res = pruning_summary(shells)
res['schema_version'] = 1
res['status'] = 'exploratory'
res['note'] = ('densita piccoli vasi vs distanza dalla pleura; campo di distanza '
               'su griglia x3. Confronti a parita di protocollo.')

# "piccolo vaso" = residuo di apertura morfologica r=1.26 mm sulla maschera vaso:
# e' un proxy morfologico, NON la sezione ortogonale del vaso (BV5 vero). Lo diciamo
# nella provenienza cosi' non venga letto come BV5.
res['provenance'] = provenance(
    'vascular_opening_residual_gradient',
    params={'small_vessel': 'residuo di apertura morfologica r=1.26 mm '
                            '(sqrt(5/pi)); proxy, non sezione ortogonale',
            'periph_mm': 15.0, 'shell_edges_mm': EDGES,
            'distanza': 'EDT dalla pleura su griglia x3'},
    denominators={'n_shell': len(shells),
                  'n_voxel_vaso': int(zc.size)},
    exclusions={'griglia_x3': 'aliasing di fase nel campo di distanza',
                'maschera_vaso': 'strutture dense candidate a vaso, non solo vaso'})
json.dump(res, open('out/vascular_gradient.json', 'w'), indent=1)

print('Gradiente di pruning vascolare:')
print(f"  densita piccoli vasi  periferia {res['density_periph']}  centro {res['density_central']}  "
      f"-> pruning_ratio {res['pruning_ratio']}")
print(f"  frazione BV5 periferica {res['bv5_frac_periph']}  "
      f"gradiente {res['gradient_per_mm']}/mm (R²={res['gradient_r2']})")
