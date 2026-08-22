"""Analisi arteria/vena da maschere DL separate (TotalSegmentator lung_vessels).

Usa le maschere `lung_arteries` / `lung_veins` — gia' prodotte dallo stesso run
che da' `lung_airways`. Calcola:
  - volumi della MASCHERA arteriosa/venosa + rapporto A/V (globale e per lobo).
    NON e' volume ematico ne' perfusione (TC senza contrasto);
  - nuvole 3D di arterie e vene nel frame di display dell'albero (per av_viz).
Il pruning / BV5 (piccoli vasi) e' RITIRATO: la stima voxelwise da EDT non e' una
stima valida del calibro (guscio dei vasi grandi) — vedi review GPT, blocker #1.

Le maschere native vengono ricampionate sulla griglia iso (riferimento ct_iso,
stesso spazio fisico) e l'EDT gira sul bounding box (memory-frugale).

Percorsi delle maschere native da env AIRWAYLAB_ARTERIES / AIRWAYLAB_VEINS
(impostati dalla CLI se trova i file accanto al --mask). Se assenti, esce senza
errore: lo step e' opzionale e additivo.

Output: out/vascular_av.json (+ arteries_iso/veins_iso non salvate: solo cloud).
"""
import json
import os

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from av_core import aggregate_by_lobe, av_ratio, bvn_volumes

art_path = os.environ.get('AIRWAYLAB_ARTERIES')
vein_path = os.environ.get('AIRWAYLAB_VEINS')
if not art_path or not os.path.exists(art_path):
    print('vasculature: nessuna maschera arteriosa (AIRWAYLAB_ARTERIES) — salto')
    raise SystemExit(0)

tree = json.load(open('out/tree_measured.json'))
ISO = tree['iso']
vox_ml = ISO ** 3 / 1000
off = [tree['bbox'][i][0] for i in range(3)] if 'bbox' in tree else [0, 0, 0]
DS = json.load(open('out/lung_metrics.json'))['ds_factor']
ct_iso = sitk.ReadImage('out/ct_iso.nii.gz')

lab_ds = sitk.GetArrayFromImage(sitk.ReadImage('out/territory_labels_ds.nii.gz')).astype(np.int32)
tindex = json.load(open('out/territory_index.json'))
lut = {int(k): v['lobe'] for k, v in tindex.items()}


def resample(path):
    rs = sitk.ResampleImageFilter()
    rs.SetReferenceImage(ct_iso)
    rs.SetInterpolator(sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(rs.Execute(sitk.ReadImage(path))).astype(bool)


def edt_on_bbox(mask):
    """EDT (mm) dei soli voxel della maschera, sul bounding box. Ritorna
    (edt_values[voxel], (zf,yf,xf) indici pieni)."""
    nz = np.nonzero(mask)
    if len(nz[0]) == 0:
        return np.array([]), (np.array([]),) * 3
    bb = [(int(a.min()), int(a.max()) + 1) for a in nz]
    sub = mask[bb[0][0]:bb[0][1], bb[1][0]:bb[1][1], bb[2][0]:bb[2][1]]
    edt = ndimage.distance_transform_edt(sub, sampling=(ISO,) * 3).astype(np.float32)
    zz, yy, xx = np.nonzero(sub)
    return edt[zz, yy, xx], (zz + bb[0][0], yy + bb[1][0], xx + bb[2][0])


def lobe_of_voxels(zf, yf, xf):
    dz = np.clip(zf // DS, 0, lab_ds.shape[0] - 1)
    dy = np.clip(yf // DS, 0, lab_ds.shape[1] - 1)
    dx = np.clip(xf // DS, 0, lab_ds.shape[2] - 1)
    return np.array([lut.get(int(l), 'CENTRAL') for l in lab_ds[dz, dy, dx]], dtype=object)


def cloud(zf, yf, xf, cap=15000):
    n = len(zf)
    step = max(1, n // cap)
    s = slice(None, None, step)
    return {'x': [round(float((x - off[2]) * ISO), 1) for x in xf[s]],
            'y': [round(float((y - off[1]) * ISO), 1) for y in yf[s]],
            'z': [round(float((z - off[0]) * ISO), 1) for z in zf[s]]}


art = resample(art_path)
edt_a, (za, ya, xa) = edt_on_bbox(art)
art_ml = round(len(edt_a) * vox_ml, 1)
vein_ml = 0.0
cloud_v = {'x': [], 'y': [], 'z': []}
if vein_path and os.path.exists(vein_path):
    vein = resample(vein_path)
    edt_v, (zv, yv, xv) = edt_on_bbox(vein)
    vein_ml = round(len(edt_v) * vox_ml, 1)
    cloud_v = cloud(zv, yv, xv)
    vein_lobe = lobe_of_voxels(zv, yv, xv)
else:
    vein_lobe = np.array([], dtype=object)

art_lobe = lobe_of_voxels(za, ya, xa)
glob = bvn_volumes(edt_a, vox_ml)
per_art = aggregate_by_lobe(edt_a, art_lobe, vox_ml)
# volume venoso per lobo (per il rapporto A/V regionale)
vein_ml_lobe = {}
for lb in set(vein_lobe.tolist()):
    vein_ml_lobe[lb] = round(float((vein_lobe == lb).sum()) * vox_ml, 1)

# volume di parenchima per lobo dalla mappa territori (contesto per i volumi vascolari)
vox_ml_ds = (ISO * DS) ** 3 / 1000
paren_ml = {}
_lbl, _cnt = np.unique(lab_ds[lab_ds > 0], return_counts=True)
for _l, _c in zip(_lbl, _cnt):
    lb = lut.get(int(_l), 'CENTRAL')
    paren_ml[lb] = paren_ml.get(lb, 0.0) + _c * vox_ml_ds

# NB (review GPT, blocker #1): il BV5 per soglia voxelwise dell'EDT NON e' una stima
# valida del volume di piccoli vasi (classifica come "piccolo" il guscio periferico
# di ogni vaso grande). Ritirato dagli output finche' non c'e' una stima di calibro
# segmentale/scale-space validata. Qui restano solo volumi di maschera vascolare e A/V.
per_lobo = {}
for lb in set(list(per_art) + list(vein_ml_lobe)):
    a_ml = per_art.get(lb, {}).get('tbv_ml')
    v_ml = vein_ml_lobe.get(lb)
    par = paren_ml.get(lb)
    per_lobo[lb] = {
        'art_ml': a_ml, 'vein_ml': v_ml,
        'av_ratio': av_ratio(a_ml, v_ml) if (a_ml and v_ml) else None,
        'parenchima_ml': round(par, 1) if par else None,
    }

out = {
    'status': 'exploratory',
    'nota': 'Volumi della MASCHERA vascolare (DL) su TC SENZA contrasto: NON e\' '
            'volume ematico ne\' perfusione. Verificare la qualita\' della '
            'separazione A/V nella vista (QC). Il pruning / BV5 (piccoli vasi) e\' '
            'stato RITIRATO: la stima voxelwise da EDT non e\' valida (guscio dei '
            'vasi grandi), serve un metodo di calibro segmentale/scale-space.',
    'arterie_ml': art_ml, 'vene_ml': vein_ml,
    'av_ratio': av_ratio(art_ml, vein_ml),
    'arterioso_globale': {'tbv_ml': glob['tbv_ml']},
    'per_lobo': per_lobo,
    'cloud_art': cloud(za, ya, xa),
    'cloud_vein': cloud_v,
}
json.dump(out, open('out/vascular_av.json', 'w'), indent=1)
print(f"arterie {art_ml} ml · vene {vein_ml} ml · A/V {out['av_ratio']} "
      f"(volumi maschera vascolare, non ematici; BV5/pruning ritirato)")
for lb, s in sorted(per_lobo.items(), key=lambda x: (x[1]['art_ml'] or 0)):
    print(f"  {lb:8s} art {s['art_ml']} ml · vene {s['vein_ml']} ml · A/V {s['av_ratio']}")
