"""Discordanza aereo-vascolare REGIONALE + decomposizione (nucleo puro).

La mappa voxel di dual.py e' globale e, per l'asimmetria di visibilita'
bronchi/vasi, ha delta quasi ovunque positivo: si leggono i GRADIENTI, non i
valori assoluti. Regionalizzando (per lobo/territorio) si confronta lobo con
lobo, il che cancella gran parte del bias comune e recupera il pattern.

DUE assi DISTINTI (indici ESPLORATIVI e DESCRITTIVI, non diagnosi), mai fusi in un
unico fenotipo lobare perche' misurano cose diverse (review GPT, blocker #2):
  * COPERTURA (coverage_gap_frac) : parenchima vicino a un vaso ma NON coperto dalla
    maschera delle vie aeree. E' COPERTURA algoritmica (via aerea non rappresentata),
    NON occlusione e NON morfometria: la via non rappresentata resta 'missing', non
    entra come diametro zero. Cause possibili indistinguibili: risoluzione, profondita'
    di segmentazione, errore, o reale interruzione anatomica;
  * MORFOMETRIA bronco-arteria (ba_*) : rapporto d_bronco/d_arteria, calcolato SOLO su
    coppie rappresentate e reportabili. NON distingue dilatazione bronchiale da
    assottigliamento arterioso.
Un'occlusione CT-verificata richiederebbe evidenza positiva (annotazione futura).

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
    """Sintesi per lobo di DUE assi DISTINTI e non combinati (review GPT, blocker #2):

    1) COPERTURA delle maschere (coverage_gap_frac): frazione di parenchima del lobo
       vicino a un vaso ma NON coperto dalla maschera delle vie aeree. E' una misura di
       COPERTURA algoritmica (via aerea non rappresentata), NON una misura morfometrica
       e NON un'occlusione: la via non rappresentata resta 'missing', non entra come
       diametro zero. Cause possibili: risoluzione, profondita' di segmentazione, errore
       di segmentazione o reale interruzione anatomica (indistinguibili senza revisione).
    2) MORFOMETRIA bronco-arteria (ba_*): calcolata SOLO dove entrambi rappresentati e
       reportabili (coppie misurate). Il rapporto NON distingue dilatazione bronchiale
       da assottigliamento arterioso.

    delta_by_lobe : {lobo -> array dei delta voxel (mm) = d_aereo - d_vaso}
    ba_by_lobe    : {lobo -> lista dei BA ratio dei bronchi accoppiati e reportabili}
    I due assi NON vanno fusi in un unico fenotipo lobare (misurano cose diverse:
    copertura algoritmica vs geometria appaiata)."""
    out = {}
    lobi = set(delta_by_lobe) | set(ba_by_lobe)
    for lobo in lobi:
        d = np.asarray(delta_by_lobe.get(lobo, []), dtype=float)
        d = d[np.isfinite(d)]
        ba = [x for x in ba_by_lobe.get(lobo, []) if x is not None]
        out[lobo] = {
            'n_voxel': int(d.size),
            'delta_med_mm': round(float(np.median(d)), 1) if d.size else None,
            'coverage_gap_frac': round(float((d > mismatch_mm).mean()), 3) if d.size else None,
            'n_ba': len(ba),
            'ba_med': round(float(np.median(ba)), 2) if ba else None,
            'ba_gt1_frac': round(float(np.mean([x > ba_dil for x in ba])), 3) if ba else None,
        }
    return out


def coverage_label(coverage_gap_frac, hi=0.5):
    """Etichetta dell'asse COPERTURA (non morfometria, non occlusione), riferita a
    una soglia esplorativa esplicita. NON e' una diagnosi."""
    if coverage_gap_frac is None:
        return 'dati insufficienti'
    if coverage_gap_frac >= hi:
        return f'copertura via aerea incompleta (≥{int(hi * 100)}% soglia espl.)'
    return f'copertura via aerea adeguata (<{int(hi * 100)}% soglia espl.)'


def ba_label(ba_gt1_frac, hi=0.5):
    """Etichetta dell'asse MORFOMETRICO bronco-arteria, riferita a una soglia
    esplorativa esplicita. Un rapporto elevato NON distingue dilatazione bronchiale
    da assottigliamento arterioso. NON e' una diagnosi."""
    if ba_gt1_frac is None:
        return 'dati insufficienti'
    if ba_gt1_frac >= hi:
        return f'rapporto bronco-arteria elevato (≥{int(hi * 100)}% dei rami)'
    return f'rapporto bronco-arteria non elevato (<{int(hi * 100)}% dei rami)'
