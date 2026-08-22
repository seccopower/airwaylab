"""Discordanza aereo-vascolare REGIONALE + decomposizione (nucleo puro).

La mappa voxel di dual.py e' globale e, per l'asimmetria di visibilita'
bronchi/vasi, ha delta quasi ovunque positivo: si leggono i GRADIENTI, non i
valori assoluti. Regionalizzando (per lobo/territorio) si confronta lobo con
lobo, il che cancella gran parte del bias comune e recupera il pattern.

Decomposizione (indici ESPLORATIVI e DESCRITTIVI, non diagnosi): un singolo delta
mescola due segnali, che teniamo separati —
  * mismatch di distanza : parenchima vicino a un vaso ma LONTANO dall'albero aereo
    (delta molto positivo). NON e' occlusione: e' in gran parte via aerea non
    rappresentata / sotto-risoluzione, per l'asimmetria di profondita' di
    segmentazione tra albero vascolare (piu' profondo) e aereo;
  * rapporto bronco-arteria : via aerea piu' larga dell'arteria satellite (BA > 1).
    NON distingue dilatazione bronchiale da assottigliamento arterioso.
Si esportano i DUE indici separati, mai fusi in uno, con nomi neutri.

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
    """Etichetta DESCRITTIVA ed esplorativa di quale asse domina in un lobo, dai
    due indici separati. NON e' una diagnosi.

    ATTENZIONE (review GPT, blocker #2): l'asse 'mismatch' e' basato sulla distanza
    voxel dai due alberi (delta = d_aereo - d_vaso). Un delta alto NON dimostra
    un'occlusione: l'albero vascolare si segmenta piu' a fondo di quello aereo, quindi
    il delta misura in gran parte l'ASIMMETRIA di profondita' di segmentazione (via
    aerea non rappresentata / sotto-risoluzione), non l'occlusione. Un'occlusione
    richiede evidenza CT positiva (revisione radiologica). Per questo l'etichetta e'
    'via aerea non rappresentata', non 'occlusione'.
    Analogamente l'asse BA NON distingue dilatazione bronchiale da assottigliamento
    arterioso: etichetta neutra 'rapporto bronco-arteria elevato'.

    Ritorna dict {mismatch_idx, ba_gt1_idx, prevalenza} con prevalenza in
    {'via aerea non rappresentata','rapporto bronco-arteria elevato','mista',
     'nessuna discordanza elevata','dati insufficienti'}."""
    mm = stat.get('mismatch_frac')
    dil = stat.get('ba_gt1_frac')
    if mm is None and dil is None:
        prev = 'dati insufficienti'
    elif (mm or 0) >= mismatch_hi and (dil or 0) >= ba_hi:
        prev = 'mista'
    elif (mm or 0) >= mismatch_hi:
        prev = 'via aerea non rappresentata'
    elif (dil or 0) >= ba_hi:
        prev = 'rapporto bronco-arteria elevato'
    else:
        prev = 'nessuna discordanza elevata'
    return {'mismatch_idx': mm, 'ba_gt1_idx': dil, 'prevalenza': prev}
