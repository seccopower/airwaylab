"""Anonimizzazione DICOM per uso ricerca — v3 (selezione serie automatica).

Scansiona ricorsivamente una cartella (anche un CD PACS con DICOMDIR e file senza
estensione), elenca le serie e — di default — SCEGLIE AUTOMATICAMENTE la serie CT
piu' adatta all'analisi quantitativa: assiale, sottile, con molte immagini,
ESCLUDENDO ricostruzioni derivate (MIP/proiezioni), scout/topogrammi e report.
Converte in NIfTI (solo voxel + geometria: nessun dato anagrafico).

Uso:
    python anonymize.py <cartella_input> <output.nii.gz> [indice_serie]

Senza indice usa la scelta automatica e stampa la classifica motivata. Con un
indice esplicito forza quella serie. Se la serie scelta e' spessa (>3 mm) o
derivata avvisa in modo evidente: uno spessore alto rende inaffidabile la
quantificazione (albero/calibro/territori/flusso).

Nota: anonimizzazione di base. Per studi formali usare un profilo completo
(DICOM PS3.15) e verificare le policy del proprio ente.
"""
import os
import sys

import SimpleITK as sitk

SKIP_DIRS = {'express', 'ihe_pdi', 'reports', 'locale', 'bin', 'config', 'help'}

# parole nella descrizione o nell'ImageType che indicano una serie NON adatta
BAD_DESC = ('mip', 'scout', 'topogram', 'topogramma', 'localizer', 'localiser',
            'dose', 'report', 'sommario', 'summary', 'monitoring', 'test bolus',
            'bolus', 'reformat', 'reformatted', 'mpr', 'ssd', 'vrt')
BAD_IMAGETYPE = ('mip', 'projection', 'localizer', 'csa mpr', 'reformatted', 'vrt', 'ssd')

MAX_OK_THICKNESS_MM = 3.0   # oltre: quantificazione inaffidabile


def _parse_thickness(t):
    try:
        return float(str(t).replace(',', '.'))
    except Exception:
        return 999.0


def disqualify_reason(meta):
    """Ritorna una stringa se la serie NON e' adatta, altrimenti None.
    meta: dict con modality, desc, image_type, n_images, thickness."""
    if (meta.get('modality') or '').upper() != 'CT':
        return 'non CT'
    d = (meta.get('desc') or '').lower()
    it = (meta.get('image_type') or '').lower()
    if any(b in it for b in BAD_IMAGETYPE):
        return 'ricostruzione derivata (ImageType)'
    if any(b in d for b in BAD_DESC):
        return 'serie derivata/di servizio (descrizione)'
    if meta.get('n_images', 0) < 40:
        return 'troppe poche immagini'
    return None


def rank_series(metas):
    """Ordina le serie dalla piu' adatta alla meno adatta all'analisi quantitativa.

    metas: lista di dict {modality, desc, image_type, n_images, thickness}.
    Ritorna una lista di dict originali arricchiti con: 'orig_index', 'disq'
    (motivo di squalifica o None), ordinata: prima le qualificate per strato
    ASCENDENTE (sottile meglio) poi per n_images DISCENDENTE; in coda le squalificate.
    Puro: nessun I/O. Testato in tests/test_anonymize.py."""
    out = []
    for i, m in enumerate(metas):
        mm = dict(m)
        mm['orig_index'] = i
        mm['thick_val'] = _parse_thickness(m.get('thickness'))
        mm['disq'] = disqualify_reason(m)
        out.append(mm)
    # chiave: (squalificata?, strato, -immagini)
    out.sort(key=lambda m: (m['disq'] is not None, m['thick_val'], -m.get('n_images', 0)))
    return out


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


def describe(f):
    """Legge pochi tag: modalita', descrizione, spessore, ImageType."""
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.LoadPrivateTagsOff()
    r.ReadImageInformation()

    def tag(t):
        try:
            return r.GetMetaData(t).strip()
        except Exception:
            return '?'
    return (tag('0008|0060'), tag('0008|103e'), tag('0018|0050'), tag('0008|0008'))


def main(src, dst, pick=None):
    series = find_series(src)
    if not series:
        sys.exit('nessuna serie DICOM trovata in ' + src)
    items = list(series.items())            # [(sid, files)]
    metas = []
    for sid, files in items:
        mod, desc, thick, itype = describe(files[0])
        metas.append({'modality': mod, 'desc': desc, 'thickness': thick,
                      'image_type': itype, 'n_images': len(files)})

    ranked = rank_series(metas)
    print(f'{len(items)} serie trovate (ordinate per idoneita\' all\'analisi):')
    for m in ranked:
        i = m['orig_index']
        flag = 'X ' + m['disq'] if m['disq'] else 'OK'
        print(f'  [{i}] {m["n_images"]:5d} imm  {m["modality"]:3s}  strato {m["thickness"]} mm  '
              f'[{flag}]  {m["desc"]}')

    if pick is not None:
        idx = int(pick)
        print(f'-> serie forzata da riga di comando: [{idx}]')
    else:
        best = next((m for m in ranked if m['disq'] is None), None)
        if best is None:
            best = ranked[0]
            print('ATTENZIONE: nessuna serie pienamente idonea; uso la meno peggio.')
        idx = best['orig_index']
        print(f'-> scelta automatica: [{idx}] (strato {best["thickness"]} mm, '
              f'{best["n_images"]} immagini) — {best["desc"]}')

    chosen = metas[idx]
    tv = _parse_thickness(chosen.get('thickness'))
    if tv > MAX_OK_THICKNESS_MM:
        print(f'\n*** ATTENZIONE: strato {chosen.get("thickness")} mm (> {MAX_OK_THICKNESS_MM} mm). '
              'Serie SPESSA: la quantificazione (albero/calibro/territori/flusso) sara\' '
              'inaffidabile. Scegli una ricostruzione sottile (~1 mm) se disponibile. ***\n')
    dq = disqualify_reason(chosen)
    if dq:
        print(f'\n*** ATTENZIONE: la serie scelta risulta "{dq}". Verifica la scelta. ***\n')

    sid, files = items[idx]
    print(f'converto la serie [{idx}] ({len(files)} immagini)...')
    reader = sitk.ImageSeriesReader()
    dirs = {os.path.dirname(f) for f in files}
    if len(dirs) == 1:
        ordered = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(next(iter(dirs)), sid)
    else:
        print('serie su piu\' cartelle: ordino per posizione (qualche minuto)...')
        r = sitk.ImageFileReader()
        r.LoadPrivateTagsOff()
        zpos = []
        for f in files:
            r.SetFileName(f)
            r.ReadImageInformation()
            try:
                z = float(r.GetMetaData('0020|0032').split('\\')[2])
            except Exception:
                z = 0.0
            zpos.append((z, f))
        zpos.sort()
        ordered = [f for _, f in zpos]
    reader.SetFileNames(ordered)
    img = reader.Execute()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    sitk.WriteImage(img, dst)
    print('salvato', dst, img.GetSize(), 'spacing', [round(s, 2) for s in img.GetSpacing()])


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
