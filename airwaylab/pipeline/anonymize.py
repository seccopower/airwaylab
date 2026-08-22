"""Anonimizzazione DICOM per uso ricerca — v2 (supporta CD paziente).

Scansiona ricorsivamente una cartella (anche un CD PACS con DICOMDIR e file
senza estensione), elenca tutte le serie trovate, sceglie la serie CT con piu'
immagini e la converte in NIfTI (solo voxel + geometria: nessun dato
anagrafico).

Uso:
    pip install SimpleITK
    python anonimizza.py <cartella_input> <output.nii.gz> [indice_serie]

Senza indice_serie usa la serie con piu' immagini. Lancia una prima volta senza
indice per vedere l'elenco numerato delle serie, poi eventualmente rilancia
scegliendo quella giusta.

Nota: anonimizzazione di base. Per studi formali usare un profilo completo
(DICOM PS3.15) e verificare le policy del proprio ente.
"""
import os
import sys
import SimpleITK as sitk

SKIP_DIRS = {'express', 'ihe_pdi', 'reports', 'locale', 'bin', 'config', 'help'}

def find_series(root):
    reader = sitk.ImageSeriesReader()
    found = []   # (dir, series_id, files)
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
            if len(files) >= 20:          # scarta scout/report
                found.append((dirpath, sid, files))
    # unisci la stessa serie divisa su piu' cartelle
    merged = {}
    for dirpath, sid, files in found:
        merged.setdefault(sid, []).extend(files)
    return merged

def describe(f):
    """Legge pochi tag da un file per descrivere la serie."""
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.LoadPrivateTagsOff()
    r.ReadImageInformation()
    def tag(t):
        try:
            return r.GetMetaData(t).strip()
        except Exception:
            return '?'
    return tag('0008|0060'), tag('0008|103e'), tag('0018|0050')  # modalita', descrizione, spessore

def main(src, dst, pick=None):
    series = find_series(src)
    if not series:
        sys.exit('nessuna serie DICOM trovata in ' + src)
    items = sorted(series.items(), key=lambda kv: -len(kv[1]))
    print(f'{len(items)} serie trovate:')
    for i, (sid, files) in enumerate(items):
        mod, desc, thick = describe(files[0])
        print(f'  [{i}] {len(files):5d} immagini  {mod:3s}  strato {thick} mm  {desc}')
    idx = int(pick) if pick is not None else 0
    sid, files = items[idx]
    print(f'-> converto la serie [{idx}] ({len(files)} immagini)')
    reader = sitk.ImageSeriesReader()
    dirs = {os.path.dirname(f) for f in files}
    if len(dirs) == 1:
        ordered = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(next(iter(dirs)), sid)
    else:
        # serie divisa su piu' cartelle: ordina per posizione spaziale (0020,0032)
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
