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
import json, io, base64, os

from discordance import ba_label, coverage_label, lobe_of, regional_summary

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

# --- discordanza REGIONALE per lobo + decomposizione (esplorativa) -----------
# asse OCCLUSIONE: frazione di voxel del lobo con delta > soglia (parenchima
# vascolarizzato ma lontano da via aerea); asse DILATAZIONE: frazione di bronchi
# del lobo con BA ratio > 1 (via piu' larga dell'arteria). Tenuti SEPARATI.
try:
    lab = sitk.GetArrayFromImage(
        sitk.ReadImage('out/territory_labels_ds.nii.gz')).astype(np.int32)
    lab = lab[:shp[0], :shp[1], :shp[2]]
    tindex = json.load(open('out/territory_index.json'))
    pairing = json.load(open('out/pairing.json')) if os.path.exists('out/pairing.json') else {}
    tm = json.load(open('out/tree_measured.json'))
    by_id = {b['id']: b for b in tm['branches']}
    kids = {}
    for _b in tm['branches']:
        kids.setdefault(_b['u'], []).append(_b)
    parent = {}
    for _p in tm['branches']:
        for _c in kids.get(_p['v'], []):
            parent[_c['id']] = _p['id']

    # etichetta voxel -> lobo (le etichette 0/background restano fuori)
    max_lab = int(lab.max())
    lut = np.array([''] * (max_lab + 1), dtype=object)
    for k, v in tindex.items():
        ik = int(k)
        if ik <= max_lab:
            lut[ik] = v.get('lobe', 'CENTRAL')
    delta_by_lobe = {}
    for lb in set(lut) - {''}:
        labs = [i for i in range(1, max_lab + 1) if lut[i] == lb]
        mask_lb = np.isin(lab, labs) & np.isfinite(delta)
        delta_by_lobe[lb] = delta[mask_lb]

    ba_by_lobe = {}
    for aid_branch, p in pairing.items():
        lb = lobe_of(aid_branch, by_id, parent)
        ba_by_lobe.setdefault(lb, []).append(p.get('ba'))

    reg = regional_summary(delta_by_lobe, ba_by_lobe)
    # due assi separati, etichette indipendenti, NESSUN fenotipo combinato
    regional = {lb: {**reg[lb],
                     'coverage_label': coverage_label(reg[lb]['coverage_gap_frac']),
                     'ba_label': ba_label(reg[lb]['ba_gt1_frac'])}
                for lb in reg}

    # --- nuvola 3D del delta nel frame di DISPLAY dell'albero (per dual_viz) --
    # voxel DS -> voxel pieno (*DS) -> coord ritagliate (-off) -> mm (*ISO).
    off = [tm['bbox'][i][0] for i in range(3)] if 'bbox' in tm else [0, 0, 0]
    zz, yy, xx = np.nonzero(lung & np.isfinite(delta))
    if len(zz):
        stride = max(1, len(zz) // 25000)      # tetto ~25k punti
        sel = slice(None, None, stride)
        zz, yy, xx = zz[sel], yy[sel], xx[sel]
        dv = delta[zz, yy, xx]
        cloud = {
            'x': [round(float((x * DS - off[2]) * ISO), 1) for x in xx],
            'y': [round(float((y * DS - off[1]) * ISO), 1) for y in yy],
            'z': [round(float((z * DS - off[0]) * ISO), 1) for z in zz],
            'delta': [round(float(d), 1) for d in dv],
        }
    else:
        cloud = {'x': [], 'y': [], 'z': [], 'delta': []}
    out_reg = {
        'status': 'exploratory',
        'nota': 'due assi DISTINTI e non combinati, esplorativi, non validati, non '
                'diagnosi. coverage_gap_frac = frazione di gap della maschera aerea nei '
                'dintorni della maschera vascolare: numeratore = voxel di parenchima con '
                '(d_aereo - d_vascolare) > soglia; DENOMINATORE = voxel di parenchima del '
                'lobo (campo n_voxel); DISTANZA di ricerca = soglia Delta = mismatch_mm. '
                'E\' COPERTURA algoritmica (via aerea non rappresentata), NON occlusione '
                'ne\' morfometria; la via non rappresentata resta missing, non entra come '
                'diametro zero; dipende dalla profondita\' relativa delle due segmentazioni. '
                'ba_gt1_frac = frazione di bronchi con BA>1, solo su coppie rappresentate e '
                'reportabili (NON distingue dilatazione da assottigliamento arterioso). '
                'I due assi non vanno fusi.',
        'mismatch_mm': 10.0, 'ba_dilatazione': 1.0,
        'per_lobo': regional,
        'cloud': cloud,
    }
    json.dump(out_reg, open('out/discordance_regional.json', 'w'), indent=1)
    print('discordanza regionale per lobo (assi separati):')
    for lb, s in sorted(regional.items(), key=lambda x: -(x[1]['coverage_gap_frac'] or 0)):
        print(f"  {lb:8s} copertura-gap {s['coverage_gap_frac']} · BA>1 {s['ba_gt1_frac']}"
              f" · [{s['coverage_label']}] [{s['ba_label']}]")
except FileNotFoundError as e:
    print('discordanza regionale saltata (manca', e.filename, ')')

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
# --- resa PIU' DEFINITA: campo colore upscalato BILINEARE (gradiente liscio),
# contorno polmone e albero aereo disegnati NITIDI sopra (upscale nearest) ---
scale = DS * 2
H, W = cor.shape


def _up_nn(mask_bool):
    im_m = Image.fromarray((np.flipud(mask_bool) * 255).astype(np.uint8))
    return np.array(im_m.resize((W * scale, H * scale), Image.NEAREST)) > 127


col = Image.fromarray(np.flipud(img).astype(np.uint8))
col = col.resize((W * scale, H * scale), Image.BILINEAR)
arr = np.array(col).astype(np.float32)
# contorno del polmone (riferimento anatomico)
lung_edge = _up_nn(lung.max(axis=1))
lung_edge = lung_edge ^ ndimage.binary_erosion(lung_edge, iterations=max(1, scale // 2))
arr[lung_edge] = [150, 149, 144]
# albero aereo nitido sopra
arr[_up_nn(aw_ds.max(axis=1))] = [11, 11, 11]
im = Image.fromarray(arr.astype(np.uint8))
im.save('out/dual_map.png')
buf = io.BytesIO(); im.save(buf, format='PNG', optimize=True)
open('out/dual_b64.txt', 'w').write('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())
print('saved dual_map.png')
