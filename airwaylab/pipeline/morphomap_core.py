"""Nucleo puro della mappa strutturale multi-asse (vedi morphomap.py).

Tiene DUE assi SEPARATI, per lobo/territorio, e li mette in relazione senza
collassarli in un rapporto:

  conduttanza bronchiale  q   = flusso modellato che raggiunge il territorio
                                (dal modello di flusso, esplorativo)
  bassa attenuazione inspiratoria = frazione di voxel < -950 HU (LAA) e frazione
                                di tessuto f = clip(1 + HU/1000, 0, 1)

Indice derivato — carico di bassa attenuazione pesato per la conduttanza:
  ds_flux_lobo = somma_terr( q_terr * laa_terr )
  indice globale = somma ds_flux / somma q = somma_terr( (q/somma q) * laa )
                 = MEDIA della LAA territoriale PESATA per le quote di conduttanza.

ATTENZIONE all'interpretazione (review GPT, blocker #3 e major #4):
- questo indice NON e' ventilazione reale, ne' ventilazione persa, ne' frazione
  verso polmone distrutto, ne' spazio morto. E' una media pesata di un descrittore
  di attenuazione;
- su SINGOLA inspiratoria la bassa attenuazione (LAA -950) NON equivale a distruzione
  parenchimale: puo' riflettere iperinflazione o air-trapping non enfisematoso,
  particolarmente nella popolazione asmatica. L'air-trapping si valuta sull'espiratoria;
- l'indice dipende dalla soglia HU (kernel/dose/ricostruzione) e dal volume inspiratorio:
  confrontabile tra pazienti solo a parita' di protocollo;
- le quote di conduttanza q derivano dal modello di flusso e sono in gran parte guidate
  da diametri periferici IMPUTATI, non misurati.

NOTA di disegno. Per il polmone f_tessuto e' sempre piccolo (~0.1-0.2: il polmone
e' quasi tutto aria anche da sano), quindi (1 - f_tessuto) satura vicino a 1 e non
discrimina i lobi. L'indice usa percio' l'LAA (soglia a -950 HU); f_tessuto resta
esposto come asse informativo (utile in relativo).

Puro: solo aritmetica. Testato in tests/test_morphomap.py.
"""
import numpy as np

LOBE_AID = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')

SCHEMA_VERSION = 1


def voxelwise_destruction(ct, ds_shape, laa_hu=-950.0):
    """Riduce la TC iso alla griglia ds (territori) accumulando i sotto-voxel.

    La griglia ds e' un downsampling a fattore intero della iso, con la stessa
    origine (ogni cella ds = blocco fac x fac x fac di voxel iso). Il fattore e'
    dedotto per asse (NON assunto uguale sui tre assi, NE' cubico: es. una griglia
    171x171x153 e' valida). Per ogni cella ds ritorna:
      laa      = frazione di sotto-voxel < laa_hu (enfisema, a piena risoluzione)
      f_tissue = media di clip(1 + HU/1000, 0, 1) sui sotto-voxel

    Frugale in memoria: nessun volume iso intero materializzato oltre alla TC.
    """
    ct = np.asarray(ct, dtype=np.float32)
    fac = [max(1, int(round(ct.shape[i] / ds_shape[i]))) for i in range(3)]
    ctp = np.pad(ct, [(0, ds_shape[i] * fac[i] - ct.shape[i]) for i in range(3)],
                 constant_values=-1000.0)
    laa = np.zeros(ds_shape, np.float32)
    ftis = np.zeros(ds_shape, np.float32)
    for a in range(fac[0]):
        for b in range(fac[1]):
            for c in range(fac[2]):
                sub = ctp[a::fac[0], b::fac[1], c::fac[2]][:ds_shape[0], :ds_shape[1], :ds_shape[2]]
                laa += (sub < laa_hu)
                ftis += np.clip(1.0 + sub / 1000.0, 0.0, 1.0)
    n = float(fac[0] * fac[1] * fac[2])
    return laa / n, ftis / n


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
    """Etichetta DESCRITTIVA riferita a SOGLIE esplorative esplicite (non diagnosi,
    non categorie cliniche). Incrocia bassa attenuazione inspiratoria (laa, frazione
    < -950 HU) e quota di flusso simulato del lobo (ds_share). Le parole 'alta'/'bassa'
    sono sempre riferite alle soglie riportate. NB: su singola inspiratoria la bassa
    attenuazione NON equivale a distruzione; l'air-trapping non e' identificabile da
    questa acquisizione (serve l'espiratoria). Nessun riferimento a distruzione/spazio
    morto/ventilazione."""
    if laa is None or ds_share is None:
        return 'dati insufficienti'
    L, S = int(laa_hi * 100), int(share_hi * 100)
    if laa >= laa_hi and ds_share >= share_hi:
        return f'LAA ≥{L}% e quota flusso simulato ≥{S}% (soglie espl.)'
    if laa >= laa_hi:
        return f'LAA ≥{L}%, quota flusso simulato <{S}% (soglie espl.)'
    if laa < laa_lo:
        return f'LAA <{int(laa_lo * 100)}% (soglia espl.)'
    return 'intermedio (soglie espl.)'
