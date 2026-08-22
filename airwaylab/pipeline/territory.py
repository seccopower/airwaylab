"""Territori parenchimali per ramo + test della legge di Murray.

Ogni voxel di polmone viene assegnato al ramo TERMINALE piu' vicino; il
territorio di un ramo qualsiasi e' la somma dei territori del suo sottoalbero.
Fit log-log calibro/territorio sui rami QC-ok.

Output: out/territories.json {branch_id: terr_ml}, out/murray.json
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import json

tree = json.load(open('out/tree_measured.json'))
ISO = tree['iso']
pts = np.array(tree['points'])
lm = json.load(open('out/lung_metrics.json'))
DS = lm['ds_factor']

lung = sitk.GetArrayFromImage(sitk.ReadImage('out/lung_mask_ds.nii.gz')).astype(bool)
vox_ml = (ISO * DS) ** 3 / 1000

# offset bbox: le coordinate dell'albero sono nel volume ritagliato
off = np.array([tree['bbox'][i][0] for i in range(3)]) if 'bbox' in tree else np.zeros(3, int)

branches = tree['branches']
children = {}
for b in branches:
    children.setdefault(b['u'], []).append(b)
leaves = [b for b in branches if not children.get(b['v'])]
print(f'{len(leaves)} rami terminali')

# volume di etichette dei semi: meta' distale di ogni terminale
seed = np.zeros(lung.shape, dtype=np.int32)
for li, b in enumerate(leaves, start=1):
    path = pts[b['path']][len(b['path']) // 2:]        # meta' distale
    for p in path:
        z, y, x = ((p + off) / DS).astype(int)
        if 0 <= z < seed.shape[0] and 0 <= y < seed.shape[1] and 0 <= x < seed.shape[2]:
            seed[z, y, x] = li

# assegnazione al seme piu' vicino (euclidea)
ind = ndimage.distance_transform_edt(seed == 0, return_indices=True)[1]
assigned = seed[tuple(ind)]
assigned[~lung] = 0

counts = np.bincount(assigned.ravel(), minlength=len(leaves) + 1)
terr_leaf = {leaves[i - 1]['id']: float(counts[i] * vox_ml) for i in range(1, len(leaves) + 1)}

# propagazione: territorio del ramo = somma del sottoalbero
terr = {}
def subtree(b):
    if b['id'] in terr:
        return terr[b['id']]
    t = terr_leaf.get(b['id'], 0.0)
    for c in children.get(b['v'], []):
        t += subtree(c)
    terr[b['id']] = t
    return t

import sys
sys.setrecursionlimit(10000)
roots = [b for b in branches if b['gen'] == 0] or branches[:1]
for r in roots:
    subtree(r)
for b in branches:
    subtree(b)

tot = sum(terr_leaf.values())
print(f'parenchima assegnato: {tot/1000:.2f} l (polmoni {lm["lung_volume_l"]} l)')

# --- test di Murray: log10(territorio) vs log10(calibro), rami QC-ok ---
xs, ys, ids = [], [], []
for b in branches:
    t = terr.get(b['id'], 0.0)
    if b.get('qc') == 'ok' and b.get('d_mean') and t > 10:
        xs.append(np.log10(b['d_mean']))
        ys.append(np.log10(t))
        ids.append(b['id'])
slope, r2, n = None, None, len(xs)
if n >= 8:
    A = np.vstack([xs, np.ones(n)]).T
    coef, res, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    slope = float(coef[0])
    yhat = A @ coef
    ss_res = float(np.sum((np.array(ys) - yhat) ** 2))
    ss_tot = float(np.sum((np.array(ys) - np.mean(ys)) ** 2))
    r2 = 1 - ss_res / max(1e-9, ss_tot)
    print(f'Murray: territorio ~ calibro^{slope:.2f} (R2 {r2:.2f}, n={n}) — teoria: ^3')

json.dump({k: round(v, 1) for k, v in terr.items()}, open('out/territories.json', 'w'))

# --- mappa etichette per-voxel (DS grid) + indice etichetta->ramo/lobo -------
# serve alla discordanza REGIONALE (dual.py): aggregare delta e BA per lobo.
sitk.WriteImage(sitk.GetImageFromArray(assigned.astype(np.int32)),
                'out/territory_labels_ds.nii.gz')
parent = {}
for _p in branches:
    for _c in children.get(_p['v'], []):
        parent[_c['id']] = _p['id']
by_id = {b['id']: b for b in branches}
LOBE_AID = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')


def _lobe_of(bid):
    cur, seen = bid, set()
    for _ in range(60):
        if cur is None or cur in seen:
            break
        seen.add(cur)
        aid = by_id.get(cur, {}).get('aid')
        if aid in LOBE_AID:
            return aid
        cur = parent.get(cur)
    return 'CENTRAL'


terr_index = {str(li): {'branch_id': leaves[li - 1]['id'],
                        'lobe': _lobe_of(leaves[li - 1]['id'])}
              for li in range(1, len(leaves) + 1)}
json.dump(terr_index, open('out/territory_index.json', 'w'))
json.dump({'slope': slope and round(slope, 3), 'r2': r2 and round(r2, 3), 'n': n,
           'assigned_l': round(tot / 1000, 3)}, open('out/murray.json', 'w'))
print('saved territories.json + murray.json')
