"""Discordanza aereo-vascolare: dove il polmone e' servito dai vasi ma
lontano dalle vie aeree visibili.

Per ogni voxel di polmone: distanza dall'albero aereo (d_aw) e dall'albero
vascolare (d_ves). Delta = d_aw - d_ves. Nei territori congruenti i due
alberi penetrano insieme e delta ~ 0; delta molto positivo = parenchima
vascolarizzato senza via aerea corrispondente (ramo sotto risoluzione, non
segmentato o OCCLUSO). E' la mappa su cui cacciare i mucus plug.

Output: out/dual_metrics.json, out/dual_map.png (+ base64 in out/dual_b64.txt)
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image
import json, io, base64

info = json.load(open('out/seg_info.json'))
ISO = info['iso']
DS = json.load(open('out/lung_metrics.json'))['ds_factor']
SP = ISO * DS

lung = sitk.GetArrayFromImage(sitk.ReadImage('out/lung_mask_ds.nii.gz')).astype(bool)
aw = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(bool)
ves = sitk.GetArrayFromImage(sitk.ReadImage('out/vessel_mask.nii.gz')).astype(bool)

# confronto A SCALA PARI: le vie aeree visibili hanno lume >~2 mm, quindi il
# riferimento vascolare sono i vasi di calibro >= 2 mm (apertura morfologica
# di raggio 1 mm), non l'intera microvascolatura.
nz = np.nonzero(ves)
bb = [(int(a.min()), int(a.max()) + 1) for a in nz]
vc = ves[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
del nz
edt_in = ndimage.distance_transform_edt(vc, sampling=(ISO,) * 3).astype(np.float32)
core = edt_in >= 1.0
del edt_in
edt_out = ndimage.distance_transform_edt(~core, sampling=(ISO,) * 3).astype(np.float32)
opened = (edt_out <= 1.0) & vc
del edt_out, core, vc
ves_big = np.zeros_like(ves)
ves_big[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]] = opened
del opened, ves
print('vasi >=2 mm:', round(ves_big.sum() * ISO**3 / 1000, 1), 'ml')

aw_ds = aw[::DS, ::DS, ::DS]
ves_ds = ves_big[::DS, ::DS, ::DS]
del ves_big
# allinea le forme
shp = tuple(min(a, b, c) for a, b, c in zip(lung.shape, aw_ds.shape, ves_ds.shape))
lung = lung[:shp[0], :shp[1], :shp[2]]
aw_ds = aw_ds[:shp[0], :shp[1], :shp[2]]
ves_ds = ves_ds[:shp[0], :shp[1], :shp[2]]

d_aw = ndimage.distance_transform_edt(~aw_ds, sampling=(SP,) * 3).astype(np.float32)
d_ves = ndimage.distance_transform_edt(~ves_ds, sampling=(SP,) * 3).astype(np.float32)
delta = (d_aw - d_ves)
delta[~lung] = np.nan

va = d_aw[lung]; vv = d_ves[lung]; dd = delta[lung]
frac10 = float((dd > 10).mean() * 100)
frac15 = float((dd > 15).mean() * 100)
frac20 = float((dd > 20).mean() * 100)
m = {'d_aw_med_mm': round(float(np.median(va)), 1),
     'd_ves_med_mm': round(float(np.median(vv)), 1),
     'delta_med_mm': round(float(np.median(dd)), 1),
     'frac_delta_gt10_pct': round(frac10, 1),
     'frac_delta_gt15_pct': round(frac15, 1),
     'frac_delta_gt20_pct': round(frac20, 1)}
print(m)
json.dump(m, open('out/dual_metrics.json', 'w'), indent=1)

# mappa coronale: mediana del delta lungo y, colormap divergente blu-grigio-rosso
with np.errstate(all='ignore'):
    cor = np.nanmedian(delta, axis=1)
has = np.isfinite(cor)
# colormap: blu (vicino alle vie) .. grigio .. rosso (lontano dalle vie a
# parita' di vasi). Scala larga: l'asimmetria di visibilita' vasi/bronchi
# rende il delta quasi ovunque positivo; leggere i GRADIENTI e le asimmetrie.
lo, hi = -10.0, 60.0
t = np.clip((cor - lo) / (hi - lo), 0, 1)
c_blue = np.array([42, 120, 214]); c_gray = np.array([240, 239, 236]); c_red = np.array([227, 73, 72])
img = np.zeros(cor.shape + (3,), dtype=np.float32)
mlo = t < 0.5
img[mlo] = c_blue * (1 - t[mlo, None] * 2) + c_gray * (t[mlo, None] * 2)
img[~mlo] = c_gray * (2 - t[~mlo, None] * 2) + c_red * (t[~mlo, None] * 2 - 1)
img[~has] = [252, 252, 251]
# contorno albero aereo per riferimento
aw_proj = aw_ds.max(axis=1)
img[aw_proj] = [11, 11, 11]
im = Image.fromarray(np.flipud(img).astype(np.uint8))
im = im.resize((im.width * DS, im.height * DS), Image.NEAREST)
im.save('out/dual_map.png')
buf = io.BytesIO(); im.save(buf, format='PNG', optimize=True)
open('out/dual_b64.txt', 'w').write('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())
print('saved dual_map.png')
