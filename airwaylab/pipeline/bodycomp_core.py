"""Biomarcatori opportunistici dallo stesso torace (nucleo puro).

Una TC toracica contiene molto piu' dei polmoni: osso, muscolo e grasso danno
descrittori "gratis", potenzialmente rilevanti negli asmatici trattati con steroidi.
Calcolo puro a partire da grandezze gia' estratte dalle maschere (medie HU, conteggi
voxel), cosi' resta testabile e spiegabile — niente scatole nere. Sono descrittori
ESPLORATIVI, non misure cliniche validate.

  - OSSO: attenuazione trabecolare media dei corpi vertebrali (HU). Piu' bassa = osso
    meno denso. NESSUN flag automatico: la soglia (~110 HU) non e' validata per questa
    ROI/livello -> valore grezzo, non screening osteoporosi.
  - MUSCOLO: volume del muscolo scheletrico e sua attenuazione media (HU piu' bassa =
    piu' infiltrazione adiposa). NON e' una misura validata di sarcopenia (che
    richiede indici e soglie EWGSOP2).
  - GRASSO: volume sottocutaneo (SAT) e grasso INTERNO del tronco (torso_fat di TS,
    NON VAT segmentato) e loro rapporto.

CAVEAT: valori assoluti dipendono da kernel/dose/kV e dal campo di vista; confronti
solo a parita' di protocollo. Esplorativi, non diagnosi.

Puro: aritmetica. Testato in tests/test_bodycomp_core.py.
"""

# soglia di RIFERIMENTO (lombare), esportata come nota — NON usata come allarme:
# non e' validata per questa ROI (corpo vertebrale intero eroso, livelli toracici).
LOW_BONE_HU_REF = 110.0


def bone_summary(vertebra_hu, low_hu_ref=LOW_BONE_HU_REF):
    """vertebra_hu: {nome_vertebra -> HU media trabecolare}. Ritorna media, minimo e
    per-vertebra. `low_flag` e' RITIRATO (sempre None): la soglia non e' validata per
    questa ROI. `low_hu_ref` resta come nota informativa. Non diagnostico."""
    vals = {k: v for k, v in vertebra_hu.items() if v is not None}
    if not vals:
        return {'mean_hu': None, 'min_hu': None, 'n': 0,
                'per_vertebra': {}, 'low_flag': None, 'low_hu_ref': low_hu_ref}
    xs = list(vals.values())
    return {'per_vertebra': {k: round(v, 1) for k, v in vals.items()},
            'mean_hu': round(sum(xs) / len(xs), 1),
            'min_hu': round(min(xs), 1), 'n': len(xs),
            'low_flag': None, 'low_hu_ref': low_hu_ref}


def muscle_summary(n_vox, mean_hu, vox_ml):
    """Muscolo scheletrico: volume (ml) e attenuazione media (HU)."""
    if not n_vox:
        return {'muscle_ml': None, 'muscle_hu': None}
    return {'muscle_ml': round(n_vox * vox_ml, 1),
            'muscle_hu': round(mean_hu, 1) if mean_hu is not None else None}


def fat_summary(sat_vox, internal_vox, vox_ml, internal_source=None):
    """Grasso sottocutaneo (SAT) e grasso INTERNO del tronco in ml, con rapporto.
    `internal_vox` viene di norma dal `torso_fat` di TotalSegmentator: NON e' VAT
    segmentato, per questo il campo si chiama `internal_fat_ml`, non `vat_ml`.
    `internal_source` (opz.) registra da quale maschera arriva."""
    sat = sat_vox * vox_ml if sat_vox else None
    internal = internal_vox * vox_ml if internal_vox else None
    ratio = (internal / sat) if (sat and internal) else None
    return {'sat_ml': round(sat, 1) if sat is not None else None,
            'internal_fat_ml': round(internal, 1) if internal is not None else None,
            'internal_sat_ratio': round(ratio, 3) if ratio is not None else None,
            'internal_source': internal_source}


def bodycomp_summary(vertebra_hu, muscle, fat):
    """Combina i tre in un dict. `muscle` = (n_vox, mean_hu, vox_ml); `fat` =
    (sat_vox, internal_vox, vox_ml[, internal_source]). Elementi assenti -> None,
    senza far fallire."""
    out = {'bone': bone_summary(vertebra_hu or {})}
    out['muscle'] = muscle_summary(*muscle) if muscle else {'muscle_ml': None, 'muscle_hu': None}
    out['fat'] = fat_summary(*fat) if fat else {
        'sat_ml': None, 'internal_fat_ml': None, 'internal_sat_ratio': None,
        'internal_source': None}
    return out
