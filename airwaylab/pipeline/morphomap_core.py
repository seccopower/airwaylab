"""Nucleo puro della mappa strutturale multi-asse (vedi morphomap.py).

Tiene DUE assi SEPARATI, per lobo/territorio, e li mette in relazione senza
collassarli in un rapporto:

  conduttanza bronchiale  q   = flusso modellato che raggiunge il territorio
                                (dal modello di flusso; surrogato di ventilazione)
  distruzione parenchimale     = enfisema: frazione di voxel < -950 HU (LAA)
                                e frazione di tessuto f = clip(1 + HU/1000, 0, 1)

Indice derivato — quanta conduttanza e' diretta a parenchima distrutto:
  ds_flux_lobo = somma_terr( q_terr * laa_terr )
  quota_conduttanza_verso_distrutto (globale) = somma ds_flux / somma q
Alto = molta ventilazione modellata verso polmone enfisematoso = candidato
SPAZIO MORTO strutturale. NON e' ventilazione misurata ne' spazio morto misurato:
e' conduttanza (modello) x distruzione (TC), tenute distinte.

NOTA di disegno. Per il polmone f_tessuto e' sempre piccolo (~0.1-0.2: il polmone
e' quasi tutto aria anche da sano), quindi (1 - f_tessuto) satura vicino a 1 e non
discrimina i lobi. L'indice usa percio' l'LAA (soglia a -950 HU), che e' lo standard
QCT dell'enfisema ed e' discriminante; f_tessuto resta esposto come asse informativo
(utile in relativo, non come deficit assoluto).

Puro: solo aritmetica. Testato in tests/test_morphomap.py.
"""
LOBE_AID = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')

SCHEMA_VERSION = 1


def tissue_fraction(hu):
    """Frazione di tessuto (non-aria) da un valore HU: clip(1 + HU/1000, 0, 1).
    Decomposizione aria-tessuto a due compartimenti (aria = -1000, tessuto = 0)."""
    x = 1.0 + float(hu) / 1000.0
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def aggregate_lobes(territories, lobes=LOBE_AID):
    """Aggrega per lobo una lista di territori. Ogni territorio e' un dict:
        {'lobe', 'q', 'laa', 'f_tissue', 'n'}
      q        = conduttanza (flusso, ml/s; i valori negativi non contribuiscono)
      laa      = frazione di enfisema del territorio [0..1]
      f_tissue = frazione di tessuto media del territorio [0..1]
      n        = numero di voxel del territorio (peso volumetrico)

    Ritorna (per_lobo, globale). per_lobo[lb] ha:
      n_vox, cond_q, cond_frac (quota del flusso totale), laa, f_tissue,
      ds_flux (q*laa sommato), ds_share (quota del mismatch totale)."""
    def qpos(t):
        return t['q'] if t['q'] and t['q'] > 0 else 0.0

    q_tot = sum(qpos(t) for t in territories) or 1.0
    ds_tot = sum(qpos(t) * t['laa'] for t in territories)
    n_all = sum(t['n'] for t in territories) or 1

    per = {}
    for lb in lobes:
        ts = [t for t in territories if t['lobe'] == lb]
        if not ts:
            continue
        nv = sum(t['n'] for t in ts) or 1
        q = sum(qpos(t) for t in ts)
        laa = sum(t['laa'] * t['n'] for t in ts) / nv
        ftis = sum(t['f_tissue'] * t['n'] for t in ts) / nv
        ds = sum(qpos(t) * t['laa'] for t in ts)
        per[lb] = {
            'n_vox': int(nv),
            'cond_q': round(q, 1),
            'cond_frac': round(q / q_tot, 3),
            'laa': round(laa, 3),
            'f_tissue': round(ftis, 3),
            'ds_flux': round(ds, 1),
            'ds_share': round(ds / ds_tot, 3) if ds_tot else None,
        }
    glob = {
        'cond_to_destroyed': round(ds_tot / q_tot, 3),
        'laa_lung': round(sum(t['laa'] * t['n'] for t in territories) / n_all, 3),
        'f_tissue_lung': round(sum(t['f_tissue'] * t['n'] for t in territories) / n_all, 3),
    }
    return per, glob


def classify_lobe(laa, ds_share, laa_hi=0.40, share_hi=0.20, laa_lo=0.25):
    """Etichetta ESPLORATIVA del pattern del lobo (non diagnosi).

    Incrocia distruzione (laa) e quota del mismatch di conduttanza (ds_share)."""
    if laa is None or ds_share is None:
        return 'dati insufficienti'
    if laa >= laa_hi and ds_share >= share_hi:
        return 'conduttanza alta verso parenchima distrutto (candidato spazio morto)'
    if laa >= laa_hi:
        return 'distrutto, conduttanza contenuta'
    if laa < laa_lo:
        return 'parenchima conservato'
    return 'intermedio'
