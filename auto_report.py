"""AirwayLab — report in un solo comando (uso locale, ricerca).

Da CD/cartella DICOM al report, senza passi intermedi:
    1. sceglie AUTOMATICAMENTE la serie giusta (CT assiale a strato piu' fine)
    2. anonimizza in NIfTI (solo voxel + geometria; ID neutro, es. caso04)
    3. TotalSegmentator (task lung_vessels) -> maschera vie aeree
    4. pipeline AirwayLab v0.25 (--mask --refine) -> report + QC + CSV

Tutto in LOCALE: il DICOM paziente non lascia il PC. Sul disco resta il NIfTI
anonimo; il report finisce in <cartella_progetto>/risultati/.

USO (dal venv del repo):
    python auto_report.py <cartella_dicom> --case caso04
    python auto_report.py C:\\...\\ct\\zaniboni --case caso04

Opzioni:
    --case NOME     ID neutro del caso (default: nome cartella)
    --serie N       forza l'indice di serie (salta l'euristica)
    --out CARTELLA  dove scrivere NIfTI+report (default: cartella genitore del DICOM)
    --fast          passa --fast a TotalSegmentator (piu' veloce, meno preciso)

La scelta della serie viene sempre STAMPATA nel log: se sbaglia, rilancia con
--serie N leggendo l'indice dall'elenco.
"""
import argparse
import os
import subprocess
import sys

import SimpleITK as sitk

# descrizioni NON-diagnostiche / non assiali. I termini lunghi si cercano come
# sottostringa; i token corti ambigui solo come PAROLA INTERA (evita falsi
# positivi tipo 'cor' dentro altre parole).
BAD_SUBSTR = ('scout', 'localizer', 'topogram', 'surview', 'dose', 'report',
              'screen save', 'secondary', 'reformat', 'coronal', 'sagittal')
BAD_TOKEN = {'mpr', 'cor', 'sag', 'vrt', 'mip', 'ssd', 'bone', 'osso'}
# kernel di ricostruzione: per le vie aeree serve un kernel POLMONARE/nitido
# (bordi netti di lume e parete). Il kernel MORBIDO/mediastinico sfuma i bordi
# e degrada segmentazione e misura, pur restando ottimo per la densitometria;
# a parita' di strato va quindi scelto quello polmonare.
LUNG_KERNEL = ('polmon', 'lung', 'parench', 'sharp', 'b50', 'b56', 'b57', 'b60',
               'b70', 'bl5', 'bl6', 'bl7', 'fc5', 'fc8', 'yc')
SOFT_KERNEL = ('mediastin', 'soft', 'standard', 'b20', 'b26', 'b30', 'b31',
               'b35', 'fc0', 'fc1', 'fc2')
SKIP_DIRS = {'express', 'ihe_pdi', 'reports', 'locale', 'bin', 'config', 'help'}


def find_series(root):
    reader = sitk.ImageSeriesReader()
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        if not filenames:
            continue
        try:
            ids = reader.GetGDCMSeriesIDs(dirpath)
        except Exception:
            continue
        for sid in ids:
            files = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if len(files) >= 20:
                found.append((dirpath, sid, files))
    merged = {}
    for dirpath, sid, files in found:
        merged.setdefault(sid, []).extend(files)
    return merged


def series_meta(f):
    """Legge i tag utili da un file della serie."""
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.LoadPrivateTagsOff()
    try:
        r.ReadImageInformation()
    except Exception:
        return {'mod': '?', 'desc': '?', 'thick': None, 'axial': False}

    def tag(t):
        try:
            return r.GetMetaData(t).strip()
        except Exception:
            return ''
    mod = tag('0008|0060') or '?'
    desc = tag('0008|103e') or '?'
    try:
        thick = float(tag('0018|0050'))
    except ValueError:
        thick = None
    # orientazione: assiale se il prodotto vettoriale dei due assi ~ (0,0,±1)
    axial = False
    try:
        o = [float(x) for x in tag('0020|0037').split('\\')]
        if len(o) == 6:
            rx, ry, rz, cx, cy, cz = o
            nz = abs(rx * cy - ry * cx)      # componente z della normale
            axial = nz > 0.9
    except Exception:
        axial = False
    return {'mod': mod, 'desc': desc, 'thick': thick, 'axial': axial}


def score(meta, n_images):
    """Punteggio: piu' alto = candidato migliore per l'analisi delle vie aeree.
    Preferisce CT assiale a strato fine con molte immagini; penalizza le serie
    riconosciute come scout/reformat/dose report."""
    d = meta['desc'].lower()
    if meta['mod'] not in ('CT', '?'):
        return -1e9
    tokens = set(d.replace('_', ' ').replace('-', ' ').split())
    if any(b in d for b in BAD_SUBSTR) or (tokens & BAD_TOKEN):
        return -1e6
    s = 0.0
    s += 300.0 if meta['axial'] else 0.0            # assiale fortemente preferito
    if meta['thick'] is not None:
        # strato fine premiato; <=1.5mm ideale, penalita' crescente oltre
        s += max(0.0, 100.0 - 40.0 * max(0.0, meta['thick'] - 1.5))
        s += 60.0 if meta['thick'] <= 1.5 else 0.0
    s += min(120.0, n_images / 5.0)                 # piu' immagini = piu' fine, con tetto
    # kernel: a parita' d'altro, polmonare >> mediastinico/morbido per le vie aeree
    if any(k in d for k in LUNG_KERNEL):
        s += 200.0
    if any(k in d for k in SOFT_KERNEL):
        s -= 200.0
    return s


def choose_series(series):
    items = sorted(series.items(), key=lambda kv: -len(kv[1]))
    rows = []
    for sid, files in items:
        m = series_meta(files[0])
        rows.append({'sid': sid, 'files': files, 'n': len(files),
                     'meta': m, 'score': score(m, len(files))})
    rows.sort(key=lambda r: -r['score'])
    return rows


def log_series(rows, chosen_idx):
    print(f'\n{len(rows)} serie candidate (ordinate per idoneita\'):')
    for i, r in enumerate(rows):
        m = r['meta']
        th = f"{m['thick']}mm" if m['thick'] is not None else '?mm'
        ax = 'assiale' if m['axial'] else 'NON-assiale'
        mark = '  <== SCELTA' if i == chosen_idx else ''
        print(f"  [{i}] score {r['score']:6.0f}  {r['n']:5d} img  {m['mod']:3s}  "
              f"strato {th:7s}  {ax:11s}  {m['desc'][:40]}{mark}")


def run(cmd, **kw):
    print('\n$ ' + ' '.join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description='AirwayLab report in un comando')
    ap.add_argument('dicom_dir')
    ap.add_argument('--case', default=None)
    ap.add_argument('--serie', type=int, default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()

    src = os.path.abspath(a.dicom_dir)
    if not os.path.isdir(src):
        sys.exit('cartella DICOM non trovata: ' + src)
    case = a.case or os.path.basename(src.rstrip('/\\')) or 'caso'
    outdir = os.path.abspath(a.out or os.path.dirname(src))
    os.makedirs(outdir, exist_ok=True)
    nifti = os.path.join(outdir, case + '.nii.gz')
    seg_dir = os.path.join(outdir, case + '_seg')
    mask = os.path.join(seg_dir, 'lung_airways.nii.gz')
    risultati = os.path.join(outdir, 'risultati')

    print(f'== AirwayLab auto | caso "{case}" ==')
    print(f'DICOM:   {src}')
    print(f'NIfTI:   {nifti}')
    print(f'Report:  {risultati}\\{case}_report.html')

    # 1) scelta serie -----------------------------------------------------------
    print('\n[1/4] scansione serie DICOM...')
    series = find_series(src)
    if not series:
        sys.exit('nessuna serie DICOM (>=20 immagini) trovata in ' + src)
    rows = choose_series(series)
    if a.serie is not None:
        # l'indice --serie e' riferito all'ordine per NUMERO di immagini di
        # `airwaylab anonymize` (il piu' grande = 0): lo passo cosi' com'e'.
        chosen = None
        log_series(rows, -1)
        print(f'-> serie forzata dall\'utente: indice anonymize {a.serie}')
        anon_index = a.serie
    else:
        best = rows[0]
        log_series(rows, 0)
        if best['score'] < 0:
            sys.exit('nessuna serie idonea (tutte scout/reformat?). '
                     'Rilancia con --serie N scegliendo dall\'elenco.')
        # traduci nell'indice atteso da `airwaylab anonymize` (ordina per n img desc)
        by_count = sorted(series.items(), key=lambda kv: -len(kv[1]))
        anon_index = next(i for i, (sid, _) in enumerate(by_count)
                          if sid == best['sid'])
        print(f'-> scelta automatica: indice anonymize {anon_index} '
              f'({best["n"]} img, strato '
              f'{best["meta"]["thick"]}mm, {best["meta"]["desc"][:40]})')

    # 2) anonimizza -------------------------------------------------------------
    print('\n[2/4] anonimizzazione -> NIfTI...')
    run(['airwaylab', 'anonymize', src, nifti, str(anon_index)])

    # 3) TotalSegmentator -------------------------------------------------------
    print('\n[3/4] TotalSegmentator (lung_vessels) — puo\' richiedere minuti su CPU...')
    ts = ['TotalSegmentator', '-i', nifti, '-o', seg_dir, '--task', 'lung_vessels']
    if a.fast:
        ts.append('--fast')
    run(ts)
    if not os.path.exists(mask):
        sys.exit('TotalSegmentator non ha prodotto ' + mask)

    # 4) pipeline AirwayLab -----------------------------------------------------
    print('\n[4/4] pipeline AirwayLab (--mask --refine)...')
    run(['airwaylab', 'run', nifti, '--mask', mask, '--refine',
         '--name', case, '--outdir', risultati])

    print(f'\n== FATTO ==\n  {os.path.join(risultati, case)}_report.html')
    print('  Controlla SEMPRE il QC coronale prima di fidarti dei numeri.')


if __name__ == '__main__':
    main()
