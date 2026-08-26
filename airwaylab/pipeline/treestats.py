"""Morfometria dell'albero a livello di soggetto (compute standalone).

Legge out/tree_measured.json: conteggi (rami/terminali/generazioni/lunghezza) e AFD
(box-counting sullo scheletro 3D). Scrive out/treestats.json. Gira dopo measure.py.
"""
import json
import os

import numpy as np

from provenance import provenance
from treestats_core import box_count_dimension, count_summary, geometric_sizes

TREE = 'out/tree_measured.json'

if not os.path.exists(TREE):
    print('treestats: manca tree_measured.json — salto')
    raise SystemExit(0)

tree = json.load(open(TREE))
branches = tree.get('branches', [])
iso = float(tree.get('iso', 0.625))

counts = count_summary(branches)

# scheletro in mm per il box-counting
pts = np.asarray(tree.get('points', []), dtype=float)
afd = {'afd': None, 'r2': None, 'n_points': 0, 'series': []}
if pts.ndim == 2 and pts.shape[0] >= 8:
    pts_mm = pts * iso
    extent = float(np.ptp(pts_mm, axis=0).max())
    sizes = geometric_sizes(extent, n=6, smallest=2.0)
    afd = box_count_dimension(pts_mm, sizes)

res = {'schema_version': 1, 'status': 'exploratory', 'counts': counts, 'fractal': afd,
       'nota': 'conteggi e AFD dipendono da profondita di segmentazione e '
               'protocollo: confronti solo mela-con-mela (stesso backend/versione)'}

# n_branches = TUTTI i rami del grafo (include no-lume/sotto-risoluzione): e' un
# conteggio della SEGMENTAZIONE, non delle vie aeree misurabili. Esponiamo anche
# quanti rami sono qc=='ok' (misurabili) perche' il numero non venga letto come
# "vie aeree visibili".
n_qc_ok = sum(1 for b in branches if b.get('qc') == 'ok')
res['provenance'] = provenance(
    'tree_morphometry',
    params={'afd_method': 'afd_3d_fixedgrid',
            'afd_scales': [s for s, _ in (afd.get('series') or [])],
            'afd_grid_origin': 'singola (min della nuvola di punti)'},
    denominators={'n_branches_grafo': counts.get('n_branches'),
                  'n_terminali_grafo': counts.get('n_terminals'),
                  'n_branches_qc_ok': n_qc_ok,
                  'afd_n_punti_scheletro': afd.get('n_points')},
    exclusions={'note': 'i conteggi NON filtrano per qc: includono rami '
                        'no-lume/sotto-risoluzione (parte della segmentazione)'})
json.dump(res, open('out/treestats.json', 'w'), indent=1)

print('Morfometria dell\'albero:')
print(f"  rami {counts['n_branches']} · terminali {counts['n_terminals']} · "
      f"gen max {counts['max_gen']} · lunghezza totale {counts['total_length_mm']} mm")
if afd['afd'] is not None:
    print(f"  AFD (dimensione frattale) {afd['afd']}  (R²={afd['r2']}, {afd['n_points']} punti scheletro)")
