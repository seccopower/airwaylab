"""Descrittori parenchimali oltre la densita' media (compute standalone).

Legge out/ct_iso.nii.gz e out/lung_mask_ds.nii.gz: forma dell'istogramma HU,
eterogeneita' regionale (mosaic) e distribuzione dei cluster LAA. Scrive
out/parenchyma.json. Gira dopo lung.py (serve lung_mask_ds).

Griglia di analisi: la TC e' sottocampionata alla stessa griglia ×3 della maschera
polmonare (allineamento esatto, memoria frugale). E' coerente e riproducibile; i
valori assoluti dipendono da soglia/kernel/volume -> confronti a parita' di protocollo.
"""
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from parenchyma_core import cluster_size_stats, heterogeneity, histogram_shape

CT = 'out/ct_iso.nii.gz'
LUNG = 'out/lung_mask_ds.nii.gz'
INFO = 'out/seg_info.json'
LAA_HU = -950.0

if not (os.path.exists(CT) and os.path.exists(LUNG)):
    print('parenchyma: mancano ct_iso / lung_mask_ds — salto')
    raise SystemExit(0)

iso = json.load(open(INFO)).get('iso', 0.625) if os.path.exists(INFO) else 0.625
ct = sitk.GetArrayFromImage(sitk.ReadImage(CT)).astype(np.float32)[::3, ::3, ::3]
lung = sitk.GetArrayFromImage(sitk.ReadImage(LUNG)).astype(bool)
s = tuple(min(a, b) for a, b in zip(ct.shape, lung.shape))
ct = ct[:s[0], :s[1], :s[2]]
lung = lung[:s[0], :s[1], :s[2]]

if not lung.any():
    print('parenchyma: maschera polmonare vuota — salto')
    raise SystemExit(0)

hu = ct[lung]
hist = histogram_shape(hu)

# --- eterogeneita': medie di densita' a blocchi (~15 mm) ---
iso_ds = iso * 3
B = max(2, int(round(15.0 / iso_ds)))
block_means = []
for zi in range(0, s[0], B):
    for yi in range(0, s[1], B):
        for xi in range(0, s[2], B):
            ml = lung[zi:zi + B, yi:yi + B, xi:xi + B]
            if ml.sum() >= 0.3 * ml.size:
                block_means.append(float(ct[zi:zi + B, yi:yi + B, xi:xi + B][ml].mean()))
het = heterogeneity(block_means)

# --- cluster LAA (< -950) dentro il polmone ---
laa = (ct < LAA_HU) & lung
lab, _ = ndimage.label(laa)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
sizes = sizes[sizes > 0]
clu = cluster_size_stats(sizes.tolist(), total_laa=float(sizes.sum()) if sizes.size else 0.0)

res = {'schema_version': 1, 'grid': 'ds x3', 'laa_threshold_hu': LAA_HU,
       'histogram': hist, 'heterogeneity': het, 'laa_clusters': clu}
json.dump(res, open('out/parenchyma.json', 'w'), indent=1)

print('Parenchima (oltre la densita\' media):')
print(f"  istogramma: MLD {hist['mld_hu']} · SD {hist['sd_hu']} · "
      f"skew {hist['skewness']} · kurt {hist['kurtosis']}")
print(f"  eterogeneita' (mosaic): SD regionale {het['het_sd_hu']} HU · "
      f"IQR {het['het_iqr_hu']} HU (n_blocchi={het['n_blocks']})")
print(f"  cluster LAA: {clu['n_clusters']} · maggiore {clu['largest_frac']} · "
      f"D {clu['D']} (R²={clu['r2']})")
