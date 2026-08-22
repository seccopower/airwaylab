"""Segmentazione dei vasi intrapolmonari (TC senza contrasto) + metriche.

Vasi = strutture dense dentro il polmone, escluse le pareti bronchiali.
Output: out/vessel_mask.nii.gz, out/vessel_metrics.json, out/qc_vessels.png
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image
import json

info = json.load(open('out/seg_info.json'))
ISO = info['iso']
vox_ml = ISO ** 3 / 1000

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
aw = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(bool)

# maschera polmonare a piena risoluzione (stessa logica di lung.py)
body = ct > -500
lab, n = ndimage.label(body)
sizes = ndimage.sum(body, lab, range(1, n + 1))
body = lab == (1 + int(np.argmax(sizes)))
body_filled = np.zeros_like(body)
for z in range(body.shape[0]):
    body_filled[z] = ndimage.binary_fill_holes(body[z])
air = (ct < -500) & body_filled & ~ndimage.binary_dilation(aw, iterations=2)
lab, n = ndimage.label(air)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
order = np.argsort(sizes)[::-1]
lung = np.zeros_like(air)
taken = 0
for c in order[:4]:
    if sizes[c] * vox_ml < 150:
        break
    lung |= lab == c
    taken += 1
    if taken == 2:
        break
lung = ndimage.binary_closing(lung, structure=np.ones((3, 3, 3))) & body_filled
print('polmoni (aria)', round(lung.sum() * vox_ml / 1000, 2), 'l')

# riempi i vasi come "fori" interni: fill-holes 2D lungo i tre assi (unione)
lung_filled = lung.copy()
for z in range(lung.shape[0]):
    lung_filled[z] |= ndimage.binary_fill_holes(lung[z])
for y in range(lung.shape[1]):
    lung_filled[:, y] |= ndimage.binary_fill_holes(lung[:, y])
for x in range(lung.shape[2]):
    lung_filled[:, :, x] |= ndimage.binary_fill_holes(lung[:, :, x])
lung = lung_filled
print('polmoni (riempiti)', round(lung.sum() * vox_ml / 1000, 2), 'l')

from anatomy import QualityError
if not lung.any():
    raise QualityError('vessels', 'lung mask is empty')

# vasi: densita' > -700 HU dentro il polmone, escluse le pareti bronchiali
wall_zone = ndimage.binary_dilation(aw, iterations=int(round(2.5 / ISO)))
vess = (ct > -700) & lung & ~wall_zone

# pulizia: via i granelli di rumore (< 0.05 ml)
lab, n = ndimage.label(vess)
sizes = np.bincount(lab.ravel()); sizes[0] = 0
keep = sizes >= int(0.05 / vox_ml)
vess = keep[lab]
print('vasi', round(vess.sum() * vox_ml, 1), 'ml in', int(keep.sum()) - 1 if not keep[0] else int(keep.sum()), 'componenti')

out = sitk.GetImageFromArray(vess.astype(np.uint8))
out.CopyInformation(sitk.ReadImage('out/ct_iso.nii.gz'))
sitk.WriteImage(out, 'out/vessel_mask.nii.gz')

# QC: coronale con vie aeree (rosso) e vasi (blu)
win = np.clip((ct + 1000) / 1400, 0, 1)
cor = np.stack([win.mean(axis=1)] * 3, -1)
vm = vess.max(axis=1)
am = aw.max(axis=1)
cor[vm] = cor[vm] * 0.35 + np.array([0.1, 0.25, 0.62]) * 0.65
cor[am] = [1, 0.2, 0.2]
Image.fromarray((np.flipud(cor) * 255).astype(np.uint8)).save('out/qc_vessels.png')
# assiale
z = ct.shape[0] // 2
axl = np.stack([np.clip((ct[z] + 1000) / 1400, 0, 1)] * 3, -1)
axl[vess[z]] = axl[vess[z]] * 0.35 + np.array([0.1, 0.25, 0.62]) * 0.65
axl[aw[z]] = [1, 0.2, 0.2]
Image.fromarray((axl * 255).astype(np.uint8)).save('out/qc_vessels_ax.png')

# libera memoria e calcola le metriche sul ritaglio
del ct, body, body_filled, air, lab, win, cor, lung, wall_zone, aw
if not vess.any():
    raise QualityError('vessels', 'vessel mask is empty after cleaning')
nz = np.nonzero(vess)
bb = [(int(a.min()), int(a.max()) + 1) for a in nz]
vc = vess[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
del vess, nz
def opened_volume(m, r_mm):
    """Volume (ml) della parte di m che sopravvive a un'apertura di raggio r."""
    edt_in = ndimage.distance_transform_edt(m, sampling=(ISO, ISO, ISO)).astype(np.float32)
    core = edt_in >= r_mm
    del edt_in
    if not core.any():
        return 0.0
    edt_out = ndimage.distance_transform_edt(~core, sampling=(ISO, ISO, ISO)).astype(np.float32)
    opened = (edt_out <= r_mm) & m
    del edt_out, core
    return float(opened.sum() * vox_ml)

r5 = np.sqrt(5 / np.pi)    # 1.26 mm
r10 = np.sqrt(10 / np.pi)  # 1.78 mm
tbv = float(vc.sum() * vox_ml)
if tbv < 20:
    raise QualityError('vessels', f'implausible vascular volume ({tbv:.0f} ml)')
bv5 = tbv - opened_volume(vc, r5)     # volume nei vasi con sezione < 5 mm2
bv10 = tbv - opened_volume(vc, r10)
print(f'TBV {tbv:.0f} ml · BV5 {bv5:.0f} ml ({100*bv5/tbv:.0f}%) · BV10 {bv10:.0f} ml ({100*bv10/tbv:.0f}%)')
json.dump({'tbv_ml': round(tbv, 1), 'bv5_ml': round(bv5, 1), 'bv10_ml': round(bv10, 1),
           'bv5_frac': round(bv5 / tbv, 3), 'bv10_frac': round(bv10 / tbv, 3)},
          open('out/vessel_metrics.json', 'w'), indent=1)
print('saved vessel_mask + metrics + QC')
