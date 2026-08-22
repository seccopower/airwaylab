"""Rilevatore di candidati mucus plug (prototipo, proponi-e-conferma).

Un plug e' un bronco che si interrompe dove non dovrebbe: questo modulo
esamina i rami TERMINALI di calibro >= 2.5 mm e controlla cosa c'e' oltre il
moncone lungo la direzione del ramo. Se la continuazione ha densita' di
tessuto molle (non aria = ramo che semplicemente esce dalla risoluzione, e
non vaso = confondente principale), il terminale diventa candidato.
Per ogni candidato: immagine longitudinale (l'interruzione vista di profilo),
testimone arterioso, territorio a valle.

Output: out/plugs.json
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image, ImageDraw
import json, io, base64
from lumen import pca_tangent

tree = json.load(open('out/tree_measured.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
ves = sitk.GetArrayFromImage(sitk.ReadImage('out/vessel_mask.nii.gz')).astype(bool)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    ct = ct[z0:z1, y0:y1, x0:x1].copy()
    ves = ves[z0:z1, y0:y1, x0:x1].copy()

terr = json.load(open('out/territories.json')) if __import__('os').path.exists('out/territories.json') else {}

branches = tree['branches']
children = {}
for b in branches:
    children.setdefault(b['u'], []).append(b)
by_id = {b['id']: b for b in branches}

def ancestor_name(b):
    """Nome anatomico piu' vicino risalendo l'albero."""
    parent_of = {}
    for p in branches:
        for c in children.get(p['v'], []):
            parent_of[c['id']] = p
    cur = b
    for _ in range(20):
        if cur.get('name'):
            return cur['name']
        cur = parent_of.get(cur['id'])
        if cur is None:
            return ''
    return ''

def sample(coords):
    return ndimage.map_coordinates(ct, coords.reshape(-1, 3).T, order=1,
                                   mode='nearest')

def sample_ves(coords):
    return ndimage.map_coordinates(ves.astype(np.float32), coords.reshape(-1, 3).T,
                                   order=0, mode='constant') > 0.5

import os as _os
_USE_F = _os.environ.get('AIRWAYLAB_REFINED') == '1'

def _bpath(b):
    if _USE_F and b.get('fpath'):
        return np.array(b['fpath'], dtype=float)
    return pts[b['path']].astype(float)

MIN_D_HM = 2.5              # gate for half-max diameters (tuned on those)
MIN_D_MASK = 3.0 * ISO      # gate for mask diameters: the 3-voxel floor —
#                             mask diameters near the DL floor are biased low,
#                             so the half-max threshold would silently drop
#                             real plugged stumps (clinical review, caso02:
#                             the B9-sx plug leaf had d_maschera 2.26 mm)
cands = []
leaves = [b for b in branches if not children.get(b['v'])]
for b in leaves:
    # size estimate for candidate screening: reportable caliber when present;
    # mask-diameter fallback ONLY for sotto-risoluzione (a real branch below
    # the caliber floor can still plug) — never for no-lume (existence
    # rejected by the air witness) or fuga-contorno
    if b.get('d_mean'):
        d_est, gate = b['d_mean'], MIN_D_HM
    elif b.get('qc') == 'sotto-risoluzione' and b.get('d_maschera'):
        d_est, gate = b['d_maschera'], MIN_D_MASK
    else:
        continue
    if d_est < gate:
        continue
    path = _bpath(b)
    if len(path) < 4:
        continue
    end = path[-1]
    t = pca_tangent(path, len(path) - 1)
    r_mm = max(1.2, d_est / 2 * 0.7)
    # cilindro di continuazione: da 1.5 a 9 mm oltre il moncone
    u = np.cross(t, [1, 0, 0]); u = u / (np.linalg.norm(u) or 1)
    if np.linalg.norm(u) < 0.5:
        u = np.cross(t, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(t, u)
    ss = np.arange(1.5, 9.0, 0.75)
    rr = np.arange(0, r_mm, 0.5)
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts_cyl = []
    for s in ss:
        for r in rr:
            for a in ang:
                d3 = (np.cos(a) * u + np.sin(a) * v) * r + t * s
                pts_cyl.append(end + d3 / ISO)
    pts_cyl = np.array(pts_cyl)
    hu = sample(pts_cyl)
    vfrac = float(sample_ves(pts_cyl).mean())
    hu_med = float(np.median(hu))
    # aria (< -700): il ramo continua sotto risoluzione -> non candidato
    # tessuto molle (-500..150) e non prevalentemente vaso -> candidato
    if hu_med < -500 or hu_med > 200 or vfrac > 0.55:
        continue
    # testimone arterioso: vasi nel manicotto attorno alla continuazione
    pts_ring = []
    for s in np.arange(2, 10, 1.0):
        for a in ang:
            d3 = (np.cos(a) * u + np.sin(a) * v) * (r_mm + 2.5) + t * s
            pts_ring.append(end + d3 / ISO)
    witness = float(sample_ves(np.array(pts_ring)).mean())
    cands.append({'b': b, 'end': end, 't': t, 'u': u, 'v': v,
                  'hu': round(hu_med), 'vfrac': round(vfrac, 2),
                  'witness': round(witness, 2)})

print(f'{len(leaves)} terminali, {len(cands)} candidati plug')

# immagine longitudinale per candidato: slab lungo l'asse del ramo
def long_view(c):
    b, end, t, u = c['b'], c['end'], c['t'], c['u']
    path = _bpath(b)
    STEP = 0.3
    L0, L1, W = -14.0, 16.0, 10.0
    sax = np.arange(L0, L1, STEP)
    wax = np.arange(-W, W, STEP)
    S, Wg = np.meshgrid(sax, wax)
    imgs = []
    for dv in (-0.6, 0, 0.6):
        coords = (end[None, None, :] + (S[..., None] * t[None, None, :] +
                  Wg[..., None] * u[None, None, :] + dv * c['v'][None, None, :]) / ISO)
        pl = sample(coords.reshape(-1, 3)).reshape(S.shape)
        imgs.append(pl)
    pl = np.mean(imgs, axis=0)
    g = np.clip((pl + 1350) / 1500, 0, 1)
    rgb = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    # linea del moncone (s = 0)
    x0 = int((0 - L0) / STEP)
    d.line([(x0, 0), (x0, im.height)], fill=(42, 120, 214), width=2)
    d.line([(4, im.height - 8), (4 + int(10 / STEP), im.height - 8)], fill=(255, 255, 255), width=2)
    im = im.resize((im.width * 2, im.height * 2), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

out = []
for i, c in enumerate(sorted(cands, key=lambda c: -(c['hu'] + 500) * (1 + c['witness']))):
    b = c['b']
    out.append({
        'pid': f'plug{i:02d}', 'branch': b['id'], 'gen': b['gen'],
        'zona': ancestor_name(b) or '–', 'd_moncone': b.get('d_mean') or b.get('d_maschera'),
        'hu_oltre': c['hu'], 'frac_vaso': c['vfrac'], 'testimone': c['witness'],
        'terr_ml': terr.get(b['id']), 'img': long_view(c),
    })
json.dump(out, open('out/plugs.json', 'w'))
tot = sum(len(o['img']) for o in out) / 1e6
print(f'salvati {len(out)} candidati ({tot:.1f} MB immagini)')
for o in out:
    print(' ', o['pid'], o['branch'], 'gen', o['gen'], o['zona'], 'Ø', o['d_moncone'],
          'HU', o['hu_oltre'], 'testimone', o['testimone'], 'terr', o['terr_ml'])
