"""Biomarcatori opportunistici dallo stesso torace (nucleo puro).

Una TC toracica contiene molto piu' dei polmoni: osso, muscolo e grasso danno
biomarcatori "gratis", clinicamente rilevanti proprio negli asmatici trattati con
steroidi (osteoporosi, sarcopenia). Calcolo puro a partire da grandezze gia'
estratte dalle maschere (medie HU, conteggi voxel), cosi' resta testabile e
spiegabile — niente scatole nere.

  - OSSO (screening osteoporosi opportunistico): attenuazione trabecolare media dei
    corpi vertebrali (HU). Piu' bassa = osso meno denso. NB: soglia indicativa,
    dipende dal livello vertebrale, kernel, kV -> ESPLORATIVO, non diagnostico.
  - MUSCOLO (sarcopenia / miosteatosi): volume del muscolo scheletrico e sua
    attenuazione media (HU piu' bassa = piu' infiltrazione adiposa).
  - GRASSO: volume sottocutaneo e viscerale/tronco e loro rapporto.

CAVEAT: valori assoluti dipendono da kernel/dose/kV e dal campo di vista; confronti
solo a parita' di protocollo. Screening, non diagnosi.

Puro: aritmetica. Testato in tests/test_bodycomp_core.py.
"""

# soglia indicativa (lombare, opportunistica) — SOLO per un flag esplorativo
LOW_BONE_HU = 110.0


def bone_summary(vertebra_hu, low_hu=LOW_BONE_HU):
    """vertebra_hu: {nome_vertebra -> HU media trabecolare}. Ritorna media, minimo,
    per-vertebra e un flag esplorativo (minimo sotto soglia). Non diagnostico."""
    vals = {k: v for k, v in vertebra_hu.items() if v is not None}
    if not vals:
        return {'mean_hu': None, 'min_hu': None, 'n': 0,
                'per_vertebra': {}, 'low_flag': None}
    xs = list(vals.values())
    return {'per_vertebra': {k: round(v, 1) for k, v in vals.items()},
            'mean_hu': round(sum(xs) / len(xs), 1),
            'min_hu': round(min(xs), 1), 'n': len(xs),
            'low_flag': bool(min(xs) < low_hu)}


def muscle_summary(n_vox, mean_hu, vox_ml):
    """Muscolo scheletrico: volume (ml) e attenuazione media (HU)."""
    if not n_vox:
        return {'muscle_ml': None, 'muscle_hu': None}
    return {'muscle_ml': round(n_vox * vox_ml, 1),
            'muscle_hu': round(mean_hu, 1) if mean_hu is not None else None}


def fat_summary(sat_vox, vat_vox, vox_ml):
    """Grasso sottocutaneo (SAT) e viscerale/tronco (VAT) in ml, con rapporto."""
    sat = sat_vox * vox_ml if sat_vox else None
    vat = vat_vox * vox_ml if vat_vox else None
    ratio = (vat / sat) if (sat and vat) else None
    return {'sat_ml': round(sat, 1) if sat is not None else None,
            'vat_ml': round(vat, 1) if vat is not None else None,
            'vat_sat_ratio': round(ratio, 3) if ratio is not None else None}


def bodycomp_summary(vertebra_hu, muscle, fat):
    """Combina i tre in un dict. `muscle` = (n_vox, mean_hu, vox_ml); `fat` =
    (sat_vox, vat_vox, vox_ml). Elementi assenti -> None, senza far fallire."""
    out = {'bone': bone_summary(vertebra_hu or {})}
    out['muscle'] = muscle_summary(*muscle) if muscle else {'muscle_ml': None, 'muscle_hu': None}
    out['fat'] = fat_summary(*fat) if fat else {'sat_ml': None, 'vat_ml': None, 'vat_sat_ratio': None}
    return out
