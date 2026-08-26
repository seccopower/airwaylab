"""QC di leak/connettivita' della maschera vie aeree (compute standalone).

Calcola, sul risultato della segmentazione gia' in out/, le metriche del nucleo
leak_qc_core (radius-explosion e connettivita'/isole) e scrive out/leak_qc.json.
Queste metriche sono il METRO con cui confronteremo i backend di segmentazione
(TotalSegmentator vs AeroPath vs ...): vedono i leak INTERNI (cisti/bolle) che la
vecchia guardia "fuori dal polmone" non vedeva. Il leak extrapolmonare/esofageo NON
e' calcolato (v1): vedi la nota in leak_qc_core sul perche' serve un vero inviluppo.

Dipendenze nel work dir: out/tree_measured.json, out/airway_mask.nii.gz,
out/seg_info.json. Si salta con grazia se manca l'essenziale. Gira dopo la pipeline.
"""
import json
import os

import numpy as np

from leak_qc_core import (
    BALLOON_MM,
    CENTRAL_AIDS,
    MED_MM,
    leak_summary,
    radius_explosion,
)
from provenance import provenance

TREE = 'out/tree_measured.json'
AW = 'out/airway_mask.nii.gz'
INFO = 'out/seg_info.json'

if not (os.path.exists(TREE) and os.path.exists(AW)):
    print('leak_qc: mancano tree_measured/airway_mask — salto')
    raise SystemExit(0)

import nibabel as nib
from scipy import ndimage

ISO = json.load(open(INFO)).get('iso', 0.625) if os.path.exists(INFO) else 0.625
vox_ml = ISO ** 3 / 1000.0

# --- 1) radius-explosion dai calibri-maschera dell'albero -------------------
tree = json.load(open(TREE))
byv = {b['v']: b for b in tree['branches']}          # ramo per nodo-figlio
branches = []
for b in tree['branches']:
    parent = byv.get(b['u'])
    branches.append({
        'id': b.get('id'), 'aid': b.get('aid'), 'gen': b.get('gen'),
        'd_mask': b.get('d_mask_eq'),
        'parent_d': parent.get('d_mask_eq') if parent else None,
    })
explosion = radius_explosion(branches)

# --- 2) connettivita' / isole sulla maschera a piena risoluzione ------------
mask = np.asarray(nib.load(AW).dataobj).astype(bool)
lab, n = ndimage.label(mask, structure=np.ones((3, 3, 3)))
if n > 0:
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    comp_sizes = sizes[sizes >= 2]          # ignora specki da 1 voxel
    n_components = int((sizes >= 2).sum())
    total_vox = int(sizes.sum())
    largest_vox = int(sizes.max())
    largest_frac = largest_vox / total_vox if total_vox else 1.0
    leaked_ml = (total_vox - largest_vox) * vox_ml
else:
    n_components, total_vox, largest_frac, leaked_ml = 0, 0, 1.0, 0.0
total_ml = total_vox * vox_ml

# NB: il leak extrapolmonare/esofageo NON e' calcolato qui (v1). La maschera
# polmonare sottrae le vie aeree per costruzione, quindi "fuori dal polmone" e'
# ~100% e non separa l'esofago da trachea/principali: serve un vero inviluppo
# delle vie aeree + ROI negative annotate (pezzo successivo). Vedi leak_qc_core.

res = leak_summary(explosion, n_components, largest_frac, leaked_ml, total_ml=total_ml)
res['schema_version'] = 1
res['status'] = 'exploratory'
res['provenance'] = provenance(
    'leak_qc',
    params={'radius_explosion_ratio_hi': 1.6, 'radius_explosion_floor_mm': 4.0,
            'central_aids_esclusi': sorted(CENTRAL_AIDS),
            'soglie_esplorative_mm': {'balloon': BALLOON_MM, 'medio': MED_MM}},
    denominators={'n_componenti': n_components, 'albero_ml': round(total_ml, 2)},
    exclusions={'leak_extrapolmonare': 'non calcolato in v1 (serve inviluppo '
                                       'vie aeree + ROI negative annotate)'})
json.dump(res, open('out/leak_qc.json', 'w'), indent=1)

m = res['metrics']
print('QC leak/connettivita\' della segmentazione vie aeree:')
print(f"  radius-explosion: {m['n_radius_explosion']} rami piu' larghi del genitore")
print(f"  componenti: {m['n_components']} (principale {100 * m['largest_component_frac']:.1f}%, "
      f"isole {m['leaked_islands_ml']} ml) · albero {m['airway_total_ml']} ml")
if not res['ok']:
    print('  ATTENZIONE — possibili leak:')
    for f in res['flags']:
        print(f"    [{f['severity']}] {f['msg']}")
