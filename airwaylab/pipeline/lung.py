"""Segmentazione polmonare + densitometria + dysanapsis.

Output: out/lung_mask_ds.nii.gz (maschera polmoni, downsample x3 per i territori)
        out/lung_metrics.json   (volume, MLD, LAA-950, Perc15, dysanapsis)
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import json

info = json.load(open('out/seg_info.json'))
ISO = info['iso']

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
aw = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(bool)

# body mask (come in segment.py)
body = ct > -500
lab, n = ndimage.label(body)
sizes = ndimage.sum(body, lab, range(1, n + 1))
body = lab == (1 + int(np.argmax(sizes)))
body_filled = np.zeros_like(body)
for z in range(body.shape[0]):
    body_filled[z] = ndimage.binary_fill_holes(body[z])

# polmoni: aria dentro il corpo, tolte le vie aeree, componenti maggiori
air = (ct < -500) & body_filled & ~ndimage.binary_dilation(aw, iterations=2)
lab, n = ndimage.label(air)
sizes = np.bincount(lab.ravel())
sizes[0] = 0
order = np.argsort(sizes)[::-1]
lung = np.zeros_like(air)
vol_vox = ISO ** 3 / 1000  # ml per voxel
taken = 0
for c in order[:4]:
    if sizes[c] * vol_vox < 150:      # componenti < 150 ml: non polmone
        break
    lung |= lab == c
    taken += 1
    if taken == 2:
        break
# chiusura leggera per riprendere vasi e pareti interne
lung = ndimage.binary_closing(lung, structure=np.ones((3, 3, 3)))
lung &= body_filled

from anatomy import QualityError
if not lung.any():
    raise QualityError('lung', 'lung mask is empty (no air regions >= 150 ml inside the body)')
hu = ct[lung]
vol_l = float(lung.sum() * vol_vox / 1000)
mld = float(hu.mean())
laa950 = float((hu < -950).mean() * 100)
perc15 = float(np.percentile(hu, 15))
print(f'polmoni {vol_l:.2f} l · MLD {mld:.0f} HU · LAA-950 {laa950:.1f}% · Perc15 {perc15:.0f} HU')

# dysanapsis alla Smith (JAMA 2020): media geometrica dei lumi delle vie
# anatomicamente etichettate (approssimazione delle 19 posizioni standard)
# diviso cbrt del volume polmonare, in unita' coerenti (adimensionale).
# Riferimento MESA (adulti anziani): 0.033 +/- 0.004.
tree = json.load(open('out/tree_measured.json'))
ds = [b['d_mean'] for b in tree['branches']
      if b.get('name') and b.get('qc') == 'ok' and b.get('d_mean')]
if len(ds) < 8:   # fallback se l'etichettatura e' povera
    ds = [b['d_mean'] for b in tree['branches']
          if b.get('qc') == 'ok' and b['gen'] <= 3 and b['d_mean']]
if ds:
    geo = float(np.exp(np.mean(np.log(ds))))               # mm
    dysanapsis = float((geo / 10) / np.cbrt(vol_l * 1000))  # cm / cm: adimensionale
else:
    geo, dysanapsis = None, None

# ALR4 alla Shimada (J Appl Physiol 2025): trachea, principali dx/sx,
# intermedio. Riferimento sani: ~0.09 M / 0.08 F.
ALR4_SET = {'trachea', 'bronco principale dx', 'bronco principale sx', 'bronco intermedio'}
d4 = [b['d_mean'] for b in tree['branches']
      if b.get('name') in ALR4_SET and b.get('qc') == 'ok' and b.get('d_mean')]
alr4 = float((np.exp(np.mean(np.log(d4))) / 10) / np.cbrt(vol_l * 1000)) if len(d4) == 4 else None
print('ALR4 (Shimada):', alr4 and round(alr4, 4), f'({len(d4)}/4 vie)')
print(f'media geometrica calibri gen0-3: {geo and round(geo,2)} mm · dysanapsis {dysanapsis and round(dysanapsis,3)}')

# maschera downsampled x3 per la partizione dei territori
DS = 3
lung_ds = lung[::DS, ::DS, ::DS].copy()
img = sitk.GetImageFromArray(lung_ds.astype(np.uint8))
ref = sitk.ReadImage('out/ct_iso.nii.gz')
img.SetSpacing((ISO * DS,) * 3)
img.SetOrigin(ref.GetOrigin())          # slicing [::DS] keeps the first voxel
img.SetDirection(ref.GetDirection())
sitk.WriteImage(img, 'out/lung_mask_ds.nii.gz')

json.dump({'lung_volume_l': round(vol_l, 3), 'mld_hu': round(mld, 1),
           'laa950_pct': round(laa950, 2), 'perc15_hu': round(perc15, 1),
           'diam_geo_mean_mm': geo and round(geo, 2),
           'dysanapsis': dysanapsis and round(dysanapsis, 4),
           'alr4': alr4 and round(alr4, 4),
           'ds_factor': DS},
          open('out/lung_metrics.json', 'w'), indent=1)
print('saved out/lung_mask_ds.nii.gz + lung_metrics.json')
