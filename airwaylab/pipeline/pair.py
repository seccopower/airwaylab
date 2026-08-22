"""Accoppiamento bronco-arteria e rapporto broncoarterioso (BA ratio).

Per ogni ramo aereo QC-ok cerca il vaso satellite: vicino (< 10 mm dal
centerline), parallelo (|cos| > 0.6) e di calibro plausibile. BA ratio =
diametro lume bronchiale / diametro del vaso accoppiato.

Output: out/pairing.json {airway_id: {vid, vd, dist, ba}}
"""
import numpy as np
import json
from scipy.spatial import cKDTree

at = json.load(open('out/tree_measured.json'))
vt = json.load(open('out/vessel_tree.json'))
ISO = at['iso']
apts = np.array(at['points'])
vpts = np.array(vt['points'])
offA = np.array([at['bbox'][i][0] for i in range(3)]) if 'bbox' in at else np.zeros(3)
offV = np.array([vt['bbox'][i][0] for i in range(3)])

def pca_dir(P):
    P = P - P.mean(axis=0)
    if len(P) < 3:
        return np.array([0, 0, 1.0])
    _, _, vtx = np.linalg.svd(P, full_matrices=False)
    return vtx[0]

# vasi candidati: calibro e lunghezza da "arteria satellite"
vb = [b for b in vt['branches'] if b['d'] >= 1.2 and b['length'] >= 5.0]
print(f'{len(vb)} vasi candidati')
samples, samp_branch = [], []
vdirs = {}
for bi, b in enumerate(vb):
    P = (vpts[b['path']] + offV) * ISO
    vdirs[bi] = pca_dir(P)
    step = max(1, len(P) // 8)
    for p in P[::step]:
        samples.append(p)
        samp_branch.append(bi)
samples = np.array(samples)
tree_kd = cKDTree(samples)

pairing = {}
for a in at['branches']:
    if a.get('qc') != 'ok' or not a.get('d_mean'):
        continue
    P = (apts[a['path']] + offA) * ISO
    adir = pca_dir(P)
    n = len(P)
    mids = P[n // 4: 3 * n // 4 or n]
    if len(mids) == 0:
        mids = P
    qs = mids[:: max(1, len(mids) // 5)][:5]
    cand = {}
    for q in qs:
        for j in tree_kd.query_ball_point(q, r=10.0):
            bi = samp_branch[j]
            d = float(np.linalg.norm(samples[j] - q))
            cand.setdefault(bi, []).append(d)
    best = None
    for bi, ds in cand.items():
        if len(ds) < 2:
            continue
        align = abs(float(np.dot(adir, vdirs[bi])))
        if align < 0.6:
            continue
        vd = vb[bi]['d']
        # il vaso satellite ha calibro comparabile al bronco (0.3x-3x)
        if not (0.3 * a['d_mean'] < vd < 3.0 * a['d_mean'] + 3):
            continue
        dist = float(np.mean(ds))
        if dist > 8.0 + a['d_mean'] / 2:
            continue
        # il vaso satellite e' il vaso DOMINANTE vicino e parallelo, non il
        # rametto piu' vicino: premia il calibro oltre a distanza e parallelismo
        score = dist - 2.0 * vd - 3.0 * align
        if best is None or score < best[0]:
            best = (score, bi, dist, align)
    if best is not None:
        _, bi, dist, align = best
        vd = vb[bi]['d']
        pairing[a['id']] = {'vid': vb[bi]['id'], 'vd': vd,
                            'dist': round(dist, 1), 'align': round(align, 2),
                            'ba': round(a['d_mean'] / vd, 2)}

json.dump(pairing, open('out/pairing.json', 'w'))
print(f'{len(pairing)} bronchi accoppiati')
bas = [p['ba'] for p in pairing.values()]
import collections
by_gen = collections.defaultdict(list)
for a in at['branches']:
    if a['id'] in pairing:
        by_gen[a['gen']].append(pairing[a['id']]['ba'])
print('BA ratio mediano per generazione:')
for g in sorted(by_gen):
    print(f'  gen {g}: {np.median(by_gen[g]):.2f} (n={len(by_gen[g])})')
print(f'BA mediano complessivo: {np.median(bas):.2f}')
