"""Discordanza aereo-vascolare REGIONALE + decomposizione (nucleo puro).

La mappa voxel di dual.py e' globale e, per l'asimmetria di visibilita'
bronchi/vasi, ha delta quasi ovunque positivo: si leggono i GRADIENTI, non i
valori assoluti. Regionalizzando (per lobo/territorio) si confronta lobo con
lobo, il che cancella gran parte del bias comune e recupera il pattern.

Decomposizione (indici ESPLORATIVI, non diagnosi): la discordanza ha due cause
opposte, che un singolo delta mescola —
  * occlusione / de-ventilazione : parenchima vascolarizzato ma LONTANO da una
    via aerea (delta molto positivo) -> ramo occluso o sotto-risoluzione;
  * dilatazione / rimodellamento  : via aerea piu' larga dell'arteria satellite
    (BA ratio > 1) -> sospetto bronchiectasia / rimodellamento.
Si esportano i DUE indici separati, mai fusi in uno.

Puro: nessun I/O, solo numpy. Testato in tests/test_discordance.py.
"""
import numpy as np

LOBE_AID = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')


def lobe_of(bid, by_id, parent, lobe_aids=LOBE_AID):
    """Risale i genitori fino al primo aid lobare; 'CENTRAL' per trachea/mains."""
    cur = bid
    seen = set()
    for _ in range(60):
        if cur is None or cur in seen:
            break
        seen.add(cur)
        aid = by_id.get(cur, {}).get('aid')
        if aid in lobe_aids:
            return aid
        cur = parent.get(cur)
    return 'CENTRAL'


def regional_summary(delta_by_lobe, ba_by_lobe, mismatch_mm=10.0, ba_dil=1.0):
    """Sintesi per lobo dei due assi della discordanza.

    delta_by_lobe : {lobo -> array dei delta voxel (mm)}
    ba_by_lobe    : {lobo -> lista dei BA ratio dei bronchi accoppiati}
    Ritorna {lobo -> {...}} con mediana e frazione di mismatch (asse occlusione)
    e mediana e frazione BA>1 (asse dilatazione), tenuti SEPARATI."""
    out = {}
    lobi = set(delta_by_lobe) | set(ba_by_lobe)
    for lobo in lobi:
        d = np.asarray(delta_by_lobe.get(lobo, []), dtype=float)
        d = d[np.isfinite(d)]
        ba = [x for x in ba_by_lobe.get(lobo, []) if x is not None]
        out[lobo] = {
            'n_voxel': int(d.size),
            'delta_med_mm': round(float(np.median(d)), 1) if d.size else None,
            'mismatch_frac': round(float((d > mismatch_mm).mean()), 3) if d.size else None,
            'n_ba': len(ba),
            'ba_med': round(float(np.median(ba)), 2) if ba else None,
            'ba_gt1_frac': round(float(np.mean([x > ba_dil for x in ba])), 3) if ba else None,
        }
    return out


def classify_lobe(stat, mismatch_hi=0.5, ba_hi=0.5):
    """Etichetta ESPLORATIVA di quale asse domina in un lobo, dai due indici
    separati. NON e' una diagnosi.

    Nota onesta: l'asse BA>1 NON distingue dilatazione bronchiale (bronchiectasia/
    rimodellamento) da assottigliamento arterioso (vascular pruning, tipico
    dell'enfisema) — il rapporto da solo non basta. Per questo l'etichetta e'
    neutra 'BA>1', non 'dilatazione'.

    Ritorna dict {occlusione_idx, ba_gt1_idx, prevalenza} con prevalenza in
    {'occlusione','BA>1','mista','nella norma','dati insufficienti'}."""
    mm = stat.get('mismatch_frac')
    dil = stat.get('ba_gt1_frac')
    if mm is None and dil is None:
        prev = 'dati insufficienti'
    elif (mm or 0) >= mismatch_hi and (dil or 0) >= ba_hi:
        prev = 'mista'
    elif (mm or 0) >= mismatch_hi:
        prev = 'occlusione'
    elif (dil or 0) >= ba_hi:
        prev = 'BA>1'
    else:
        prev = 'nella norma'
    return {'occlusione_idx': mm, 'ba_gt1_idx': dil, 'prevalenza': prev}
