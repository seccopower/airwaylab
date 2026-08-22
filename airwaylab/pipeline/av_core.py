"""Nucleo puro delle metriche arteria/vena (vedi vasculature.py).

BVn = volume di sangue nei vasi di sezione < n mm^2 — marker QCT di pruning
(perdita dei piccoli vasi periferici). Qui il raggio locale e' stimato con la
EDT dentro la maschera vascolare (CSA = pi*r^2), un'APPROSSIMAZIONE esplorativa
dello standard scale-space (Estepar/COPDGene): il valore assoluto non e'
confrontabile con quello, ma il confronto TRA lobi / TRA tempi sullo stesso
metodo e' coerente. Dipende da kernel/dose: confronti mela-con-mela.

Puro: solo numpy. Testato in tests/test_av.py.
"""
import numpy as np


def radius_for_csa(csa_mm2):
    """Raggio (mm) di un vaso circolare di sezione csa_mm2."""
    return float(np.sqrt(csa_mm2 / np.pi))


def bvn_volumes(edt_mm, vox_ml, csa_list=(5.0, 10.0)):
    """Da un array di raggi locali (EDT, mm) dei SOLI voxel vascolari, ritorna
    il volume totale (TBV) e i BVn per ogni soglia di sezione in csa_list.

    ATTENZIONE — i campi bvN sono DEPRECATI e RITIRATI dagli output (review GPT,
    blocker #1): sogliare l'EDT voxelwise con r<r0 NON classifica i vasi per
    calibro, ma marca il guscio periferico di spessore r0 di OGNI vaso, anche
    grande (frazione erronea di un cilindro di raggio R = 1-(1-r0/R)^2 ~ 44% per
    R=5mm). Non e' una stima valida del BV5. Serve un metodo di calibro segmentale
    /scale-space (Estepar). Qui si usa solo 'tbv_ml' (volume totale della maschera).
    """
    e = np.asarray(edt_mm, dtype=float)
    tbv = float(e.size) * vox_ml
    out = {'tbv_ml': round(tbv, 1)}
    for csa in csa_list:
        r = radius_for_csa(csa)
        vol = float((e < r).sum()) * vox_ml
        n = int(csa)
        out[f'bv{n}_ml'] = round(vol, 1)
        out[f'bv{n}_frac'] = round(vol / tbv, 3) if tbv else None
    return out


def aggregate_by_lobe(edt_mm, lobe_labels, vox_ml, csa_list=(5.0, 10.0), min_ml=2.0):
    """Aggrega i BVn per lobo. edt_mm e lobe_labels sono allineati (un valore
    per voxel vascolare). Salta i lobi con volume < min_ml."""
    e = np.asarray(edt_mm, dtype=float)
    lab = np.asarray(lobe_labels, dtype=object)
    out = {}
    for lb in set(lab.tolist()):
        m = lab == lb
        if m.sum() * vox_ml < min_ml:
            continue
        out[lb] = bvn_volumes(e[m], vox_ml, csa_list)
    return out


def av_ratio(art_ml, vein_ml):
    """Rapporto volumetrico arterie/vene (QC + interesse fisiologico)."""
    return round(art_ml / vein_ml, 2) if vein_ml else None
