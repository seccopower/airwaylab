#!/usr/bin/env python3
"""Biomarcatori opportunistici da una TC toracica + maschere di composizione corporea.

Calcola osso (attenuazione vertebrale), muscolo (volume + HU) e grasso (SAT/VAT) da
maschere gia' prodotte, sulla STESSA TC che le ha generate (stessa geometria).

NON fa parte del comando unico: e' opt-in, come il ponte AeroPath. Le maschere si
producono con i task di composizione corporea di TotalSegmentator nel MEDESIMO spazio
della TC anonimizzata, es. (verifica i nomi dei task/uscite con la tua versione di TS):
    TotalSegmentator -i caso.nii.gz -o caso_body   --task total          # vertebre
    TotalSegmentator -i caso.nii.gz -o caso_body   --task tissue_types   # muscolo/grasso
poi:
    python tools/bodycomp.py caso.nii.gz caso_body caso_bodycomp.json

Lo strumento cerca nelle maschere: *vertebra* (una per corpo vertebrale),
*skeletal_muscle*/*muscle*, *subcutaneous_fat* (SAT), *torso_fat*/*visceral* (grasso
INTERNO del tronco). NB: il `torso_fat` di TotalSegmentator NON e' VAT segmentato —
lo riportiamo come `internal_fat_ml`, non come VAT. Valori esplorativi, non
diagnostici.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "airwaylab", "pipeline"))
from bodycomp_core import bodycomp_summary   # noqa: E402


def _find(segdir, *patterns):
    hits = []
    for p in patterns:
        hits += glob.glob(os.path.join(segdir, p))
    return sorted(set(hits))


def _mask(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path)) > 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Biomarcatori opportunistici da TC + maschere")
    ap.add_argument("ct", help="TC (.nii.gz) nella stessa geometria delle maschere")
    ap.add_argument("segdir", help="cartella con le maschere di composizione corporea")
    ap.add_argument("out", nargs="?", default="bodycomp.json", help="JSON di output")
    args = ap.parse_args(argv)

    if not os.path.exists(args.ct):
        sys.exit(f"TC non trovata: {args.ct}")
    img = sitk.ReadImage(args.ct)
    ct = sitk.GetArrayFromImage(img).astype(np.float32)
    sp = img.GetSpacing()
    vox_ml = (sp[0] * sp[1] * sp[2]) / 1000.0

    # OSSO: HU media (trabecolare) per corpo vertebrale, con lieve erosione per
    # togliere il guscio corticale
    vertebra_hu = {}
    for f in _find(args.segdir, "*vertebra*.nii.gz", "*vertebrae*.nii.gz"):
        m = _mask(f)
        if m.sum() < 50:
            continue
        trab = ndimage.binary_erosion(m, iterations=2)
        use = trab if trab.sum() >= 30 else m
        vertebra_hu[os.path.basename(f).replace(".nii.gz", "")] = float(ct[use].mean())

    # MUSCOLO
    muscle = None
    mf = _find(args.segdir, "*skeletal_muscle*.nii.gz", "*muscle*.nii.gz")
    if mf:
        m = _mask(mf[0])
        if m.any():
            muscle = (int(m.sum()), float(ct[m].mean()), vox_ml)

    # GRASSO: SAT + grasso interno del tronco (torso_fat, NON VAT segmentato)
    sat_vox = internal_vox = None
    internal_source = None
    sf = _find(args.segdir, "*subcutaneous_fat*.nii.gz")
    if sf:
        sat_vox = int(_mask(sf[0]).sum())
    vf = _find(args.segdir, "*torso_fat*.nii.gz", "*visceral*.nii.gz")
    if vf:
        internal_vox = int(_mask(vf[0]).sum())
        internal_source = os.path.basename(vf[0]).replace(".nii.gz", "")
    fat = ((sat_vox, internal_vox, vox_ml, internal_source)
           if (sat_vox or internal_vox) else None)

    res = bodycomp_summary(vertebra_hu, muscle, fat)
    res["schema_version"] = 2
    res["note"] = ("descrittori esplorativi, non diagnosi; valori dipendono da "
                   "kernel/dose/kV e campo di vista -> confronti a parita' di protocollo")
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)

    b, mu, ft = res["bone"], res["muscle"], res["fat"]
    print("Biomarcatori opportunistici (esplorativi, non diagnostici):")
    if b["n"]:
        print(f"  osso: HU vertebrale media {b['mean_hu']} · minimo {b['min_hu']} "
              f"({b['n']} vertebre) — nessun flag automatico (soglia non validata)")
    else:
        print("  osso: nessuna maschera vertebrale trovata")
    if mu["muscle_ml"]:
        print(f"  muscolo: {mu['muscle_ml']} ml · HU {mu['muscle_hu']} "
              f"(HU bassa = infiltrazione adiposa)")
    if ft["sat_ml"] or ft["internal_fat_ml"]:
        src = f" [{ft['internal_source']}]" if ft.get("internal_source") else ""
        print(f"  grasso: SAT {ft['sat_ml']} ml · interno tronco {ft['internal_fat_ml']} ml{src} "
              f"· interno/SAT {ft['internal_sat_ratio']} (torso_fat, non VAT segmentato)")
    print(f"salvato {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
