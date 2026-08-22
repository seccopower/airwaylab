"""CPR rettificata unificata: immagine orizzontale + profilo calibro/parete
sullo stesso asse millimetrico, con le tacche dei rami attraversati.

Output: out/cpr.json {label: {img, len_mm, s[], d[], w[], marks[{s,label}]}}
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image, ImageDraw
import json, io, base64
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
edt = ndimage.distance_transform_edt(mask, sampling=(ISO, ISO, ISO)).astype(np.float32)

branches = tree['branches']
by_id = {b['id']: b for b in branches}
parent = {}
children = {}
for b in branches:
    children.setdefault(b['u'], []).append(b)
for p in branches:
    for c in children.get(p['v'], []):
        parent[c['id']] = p

def chain_to(bid):
    """Sequenza di rami dalla radice a bid + polilinea concatenata + confini."""
    seq = []
    cur = by_id[bid]
    while cur is not None:
        seq.append(cur)
        cur = parent.get(cur['id'])
    seq = seq[::-1]
    coords = []
    marks = []          # (indice_del_punto_di_inizio, etichetta)
    import os as _os
    use_f = _os.environ.get('AIRWAYLAB_REFINED') == '1'
    for b in seq:
        if use_f and b.get('fpath'):
            p = [list(q) for q in b['fpath']]     # centerline raffinato
        else:
            p = [list(pts[i]) for i in b['path']]
        if coords and np.allclose(coords[-1], p[0], atol=1.5):
            p = p[1:]
        marks.append((len(coords), b.get('name') or b['id']))
        coords.extend(p)
    return np.array(coords, dtype=float), marks, seq

def resample_polyline(P, step_mm):
    seg = np.linalg.norm(np.diff(P, axis=0) * ISO, axis=1)
    arc = np.concatenate([[0], np.cumsum(seg)])
    s_new = np.arange(0, arc[-1], step_mm)
    out = np.stack([np.interp(s_new, arc, P[:, k]) for k in range(3)], axis=1)
    if len(out) > 9:
        ker = np.ones(7) / 7
        for k in range(3):
            out[3:-3, k] = np.convolve(out[:, k], ker, mode='valid')[:len(out) - 6]
    return out, s_new, arc

def build(bid, half_mm=14.0, step_mm=0.4, crop_at='bronco principale',
          tail_mm=10.0):
    """CPR della rotta fino a bid. Con crop_at (default: carena) la rotta
    viene tagliata all'inizio del bronco principale meno tail_mm di trachea
    di contesto, e l'asse s diventa 'distanza dalla carena' (trachea in
    negativo): niente piu' 115 mm di trachea identica che comprimono la
    parte distale. crop_at=None -> rotta intera da apice trachea (s da 0)."""
    P0, marks_idx, seq = chain_to(bid)

    offset = 0.0
    if crop_at:
        segl = np.linalg.norm(np.diff(P0, axis=0) * ISO, axis=1)
        arcP = np.concatenate([[0.0], np.cumsum(segl)])
        s_car = None
        for i0, lab in marks_idx:
            if lab and str(lab).startswith(crop_at):
                s_car = float(arcP[min(i0, len(arcP) - 1)])
                break
        if s_car is not None and s_car > tail_mm:
            first = int(np.argmax(arcP >= s_car - tail_mm))
            P0 = P0[first:]
            marks_idx = [(i - first, lab) for i, lab in marks_idx if i >= first]
            offset = s_car - float(arcP[first])     # ~tail_mm: carena a s=0

    P, s, arc0 = resample_polyline(P0, step_mm)
    n = len(P)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-9
    u = np.cross(T[0], [1, 0, 0])
    if np.linalg.norm(u) < 0.3:
        u = np.cross(T[0], [0, 1, 0])
    u /= np.linalg.norm(u)
    rows = []
    wax = np.arange(-half_mm, half_mm, 0.3)
    for i in range(n):
        if i > 0:
            u = u - np.dot(u, T[i]) * T[i]
            u /= np.linalg.norm(u) + 1e-9
        v = np.cross(T[i], u)
        samples = []
        for dv in (-0.6, 0.0, 0.6):
            coords = P[i][None, :] + (wax[:, None] * u[None, :] + dv * v[None, :]) / ISO
            samples.append(ndimage.map_coordinates(ct, coords.T, order=1, mode='nearest'))
        rows.append(np.mean(samples, axis=0))
    img = np.array(rows).T          # ORIZZONTALE: righe = larghezza, colonne = s
    g = np.clip((img + 1350) / 1500, 0, 1)
    rgb = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    im = im.resize((im.width, im.height * 2), Image.LANCZOS)   # un filo piu' leggibile
    buf = io.BytesIO(); im.save(buf, format='PNG', optimize=True)
    img_b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

    # two-regime rule anche qui: ogni posizione lungo il percorso appartiene
    # a un ramo; se quel ramo e' stato declassato dal witness, il valore va
    # nel canale raw esplicito, non in quello clinico
    from qc_params import INVALID_QC
    bounds = []
    for k, (i0, lab) in enumerate(marks_idx):
        a0 = float(arc0[min(i0, len(arc0) - 1)])
        a1 = (float(arc0[min(marks_idx[k + 1][0], len(arc0) - 1)])
              if k + 1 < len(marks_idx) else float(arc0[-1]))
        bounds.append((a0, a1, seq[k]))

    def owner(sm):
        # intervalli semiaperti [a0, a1): il campione sulla biforcazione
        # appartiene al ramo che INIZIA li'; l'ultimo ramo chiude inclusivo
        for k, (a0, a1, b) in enumerate(bounds):
            if a0 - 1e-6 <= sm < a1 or (k == len(bounds) - 1 and sm <= a1 + 1e-6):
                return b
        return bounds[-1][2]

    # profilo lungo la STESSA polilinea, ogni 1 mm
    s_prof, d_prof, w_prof = [], [], []
    d_raw, w_raw, rep_mask = [], [], []
    for sm in np.arange(0, s[-1], 1.0):
        i = min(int(sm / step_mm), n - 1)
        r_est = float(ndimage.map_coordinates(edt, P[i][:, None], order=1)[0])
        sec = analyze_section(ct, P[i], T[i], max(r_est, 0.7), ISO)
        if sec is not None and sec['quality']['ax_ratio'] <= 1.8:
            dv = round(sec['d_eq'], 2)
            wv = round(sec['wall_med'], 2) if sec['wall_med'] is not None else None
        else:
            dv, wv = None, None
        reportable = owner(sm).get('qc') not in INVALID_QC
        rep_mask.append(1 if reportable else 0)
        from qc_params import split_reportable
        dc, dr = split_reportable(dv, reportable)
        wc, wr = split_reportable(wv, reportable)
        d_prof.append(dc); d_raw.append(dr)
        w_prof.append(wc); w_raw.append(wr)
        s_prof.append(round(sm - offset, 1))

    marks = [{'s': round(float(arc0[min(i, len(arc0) - 1)]) - offset, 1),
              'label': lab} for i, lab in marks_idx]
    return {'img': img_b64, 'len_mm': round(float(s[-1]), 0),
            's_min': round(-offset, 1), 's_max': round(float(s[-1]) - offset, 1),
            's': s_prof, 'd': d_prof, 'w': w_prof,
            'd_raw_nonreportable': d_raw, 'w_raw_nonreportable': w_raw,
            'reportable': rep_mask, 'marks': marks}

# ---- atlante: una CPR per OGNI via segmentaria etichettata ----
# (rotta = trachea -> segmentario -> discesa lungo il figlio piu' lungo)
import re

LOBI = ('lobare sup dx', 'lobare medio', 'lobare inf dx',
        'lobare sup sx', 'lingulare', 'lobare inf sx')
LOBE_ORDER = {n: i for i, n in enumerate(LOBI)}

parent_of = {}
for p in branches:
    for c in children.get(p['v'], []):
        parent_of[c['id']] = p

def lobe_of(b):
    cur = b
    for _ in range(20):
        if cur.get('name') in LOBI:
            return cur['name']
        cur = parent_of.get(cur['id'])
        if cur is None:
            return '?'
    return '?'

def route_summary(o):
    """Sintesi quantitativa della rotta (solo tratto reportabile)."""
    dv = [x for x in o['d'] if x is not None]
    wv = [x for x in o['w'] if x is not None]
    rep = o['reportable']
    return {
        'rep_pct': round(100 * sum(rep) / max(1, len(rep))),
        'd_min_rep': round(min(dv), 2) if dv else None,
        'wall_med_rep': round(float(np.median(wv)), 2) if wv else None,
    }

seg = [b for b in branches
       if b.get('name') and re.match(r'^B\d', b['name'])]
seg.sort(key=lambda b: (LOBE_ORDER.get(lobe_of(b), 9), b['name']))

out = {}
# rotta dedicata: la trachea da sola (asse da apice, senza crop)
tr = next((b for b in branches if b.get('name') == 'trachea'), None)
if tr is not None:
    o = build(tr['id'], crop_at=None)
    o['lobo'] = 'vie centrali'
    o['summary'] = route_summary(o)
    out['trachea'] = o
    print(f"trachea   [vie centrali] {o['len_mm']:.0f} mm")

# rotte segmentarie: dalla carena (ultimo cm di trachea come contesto)
for b in seg:
    cur = b
    while children.get(cur['v']):
        cur = max(children[cur['v']], key=lambda c: c['length'])
    o = build(cur['id'])
    o['lobo'] = lobe_of(b)
    o['summary'] = route_summary(o)
    out[b['name']] = o
    s = o['summary']
    print(f"{b['name']:9s} [{o['lobo']}] {o['len_mm']:.0f} mm "
          f"(carena→fine {o['s_max']:.0f}) · reportabile {s['rep_pct']}% · "
          f"Ø min {s['d_min_rep']} · parete med {s['wall_med_rep']}")

json.dump(out, open('out/cpr.json', 'w'))

# sintesi per rotta anche come CSV (una riga per segmentario)
import csv as _csv
with open('out/routes.csv', 'w', newline='') as f:
    w = _csv.writer(f)
    w.writerow(['rotta', 'lobo', 'lunghezza_mm', 'tratto_reportabile_pct',
                'diametro_min_reportabile_mm', 'parete_mediana_reportabile_mm'])
    for name, o in out.items():
        s = o['summary']
        w.writerow([name, o['lobo'], o['len_mm'], s['rep_pct'],
                    s['d_min_rep'], s['wall_med_rep']])

print(f'{len(out)} CPR unificate, {sum(len(v["img"]) for v in out.values())/1e6:.1f} MB')
