"""Caratterizza il floor di risoluzione del calibro (backlog #27).

Fa scendere il diametro di un tubo digitale verso il voxel, con PSF e rumore, e
misura ogni sezione con lo STESSO stimatore della pipeline (`analyze_section` di
lumen.py). Ne esce la curva di bias e dispersione dell'half-max, e da quella il
diametro sotto il quale la misura non regge piu': il floor, misurato invece che
assunto.

Uso:
    python tools/floor_sweep.py [--out report.json] [--reps 9] [--quick]

Non richiede dati di paziente ne' un fantoccio fisico. Deterministico a parita'
di seed. Cosa NON e': accuratezza metrologica contro uno standard tracciabile —
e' il comportamento dello stimatore su un bordo sintetico, con sigma della PSF e
rumore che sono parametri, non misure di uno scanner reale.
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(HERE, '..', 'airwaylab', 'pipeline')
sys.path.insert(0, PIPE)

from lumen import analyze_section          # noqa: E402
from phantom_core import (                 # noqa: E402
    floor_from_sweep, recovery_stats, synth_tube,
)
from qc_params import VOXELS_FLOOR         # noqa: E402

# Spacing rappresentativi: sottile, raccomandato, al limite, spesso.
SPACINGS = (0.60, 0.70, 1.00, 1.25)
# Sigma della PSF in unita' di voxel: un kernel morbido sfoca piu' di uno sharp.
PSF_SIGMAS = {'sharp': 0.6, 'morbido': 1.0}
NOISE_HU = {'sharp': 60.0, 'morbido': 25.0}
WALL_MM = 1.0


def diameters_for(spacing, quick=False):
    """Diametri veri da esplorare, espressi in voxel per rendere il risultato
    confrontabile fra spacing diversi."""
    voxels = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0) if quick else \
             (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0)
    return [round(v * spacing, 3) for v in voxels]


def measure_once(d_true, spacing, psf_sigma_mm, noise_hu, seed):
    vol, center = synth_tube(d_true, WALL_MM, spacing,
                             psf_sigma_mm=psf_sigma_mm, noise_hu=noise_hu,
                             seed=seed)
    t = np.array([1.0, 0.0, 0.0])          # asse del tubo = z (ordine zyx)
    sec = analyze_section(vol, center, t, r_est_mm=d_true / 2.0, iso=spacing)
    return None if sec is None else float(sec['d_eq'])


def run(reps=9, quick=False):
    results = []
    for kernel, sigma_vox in PSF_SIGMAS.items():
        for spacing in SPACINGS:
            rows = []
            for d_true in diameters_for(spacing, quick):
                measured, fails = [], 0
                for k in range(reps):
                    try:
                        m = measure_once(d_true, spacing, sigma_vox * spacing,
                                         NOISE_HU[kernel], seed=1000 * k + 7)
                    except Exception:
                        m = None
                    if m is None:
                        fails += 1
                    else:
                        measured.append(m)
                st = recovery_stats(measured, d_true)
                st.update(d_true=d_true, d_voxels=round(d_true / spacing, 2),
                          success_frac=(reps - fails) / reps, n_attempts=reps)
                rows.append(st)
            fl = floor_from_sweep(rows)
            fl['floor_voxels'] = (None if fl['floor_mm'] is None
                                  else round(fl['floor_mm'] / spacing, 2))
            rows = fl.pop('rows')          # con la pendenza gia' calcolata
            results.append({'kernel': kernel, 'psf_sigma_voxels': sigma_vox,
                            'noise_hu': NOISE_HU[kernel], 'spacing_mm': spacing,
                            'floor': fl, 'rows': rows})
    return results


def report(results):
    out = []
    out.append('Floor di risoluzione misurato sul fantoccio digitale')
    out.append('=' * 62)
    out.append('criteri: la misura risponde al vero (pendenza >= 0.5), con')
    out.append('precisione utile (CV <= 10%) e sezione misurabile (>= 90%).')
    out.append("Il bias sistematico NON entra nel criterio: e' un offset quasi")
    out.append("costante in mm dovuto alla PSF, calibrabile, non perdita di")
    out.append("informazione. E' riportato a parte nello sweep JSON.")
    out.append(f'floor attualmente assunto dal codice: {VOXELS_FLOOR:.1f} voxel')
    out.append('')
    out.append(f"{'kernel':9s} {'spacing':>8s} {'floor mm':>9s} {'floor vox':>10s}")
    out.append('-' * 62)
    vox = []
    for r in results:
        f = r['floor']
        fm = '-' if f['floor_mm'] is None else f"{f['floor_mm']:.2f}"
        fv = '-' if f['floor_voxels'] is None else f"{f['floor_voxels']:.2f}"
        if f['floor_voxels'] is not None:
            vox.append(f['floor_voxels'])
        out.append(f"{r['kernel']:9s} {r['spacing_mm']:8.2f} {fm:>9s} {fv:>10s}")
    out.append('')
    if vox:
        out.append(f'floor misurato in voxel: mediana {np.median(vox):.2f}, '
                   f'intervallo {min(vox):.2f}-{max(vox):.2f}')
        verdict = ('il valore assunto e\' CONSERVATIVO: si puo\' abbassare'
                   if max(vox) < VOXELS_FLOOR else
                   'il valore assunto e\' OTTIMISTICO: andrebbe alzato'
                   if min(vox) > VOXELS_FLOOR else
                   'il valore assunto cade dentro l\'intervallo misurato')
        out.append(f'-> {verdict}')
    else:
        out.append('nessuno spacing ha soddisfatto le tolleranze: il floor sta '
                   'oltre il campo esplorato.')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default=None, help='scrive lo sweep completo in JSON')
    ap.add_argument('--reps', type=int, default=9, help='realizzazioni di rumore')
    ap.add_argument('--quick', action='store_true', help='meno diametri')
    a = ap.parse_args()
    res = run(reps=a.reps, quick=a.quick)
    print(report(res))
    if a.out:
        with open(a.out, 'w', encoding='utf-8') as f:
            json.dump({'voxels_floor_assumed': VOXELS_FLOOR, 'sweep': res},
                      f, indent=1)
        print(f'\nsweep completo in {a.out}')


if __name__ == '__main__':
    main()
