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
import json
import os
import sys

import SimpleITK as sitk

from geometry_qc_core import duplicate_positions, geometry_banner

SKIP_DIRS = {'express', 'ihe_pdi', 'reports', 'locale', 'bin', 'config', 'help'}

# parole nella descrizione o nell'ImageType che indicano una serie NON adatta
BAD_DESC = ('mip', 'scout', 'topogram', 'topogramma', 'localizer', 'localiser',
            'dose', 'report', 'sommario', 'summary', 'monitoring', 'test bolus',
            'bolus', 'reformat', 'reformatted', 'mpr', 'ssd', 'vrt')
BAD_IMAGETYPE = ('mip', 'projection', 'localizer', 'csa mpr', 'reformatted', 'vrt', 'ssd')

MAX_OK_THICKNESS_MM = 3.0   # oltre: quantificazione inaffidabile
UNKNOWN_THICKNESS_MM = 999.0  # sentinella di _parse_thickness: spessore non leggibile


def _parse_thickness(t):
    try:
        return float(str(t).replace(',', '.'))
    except Exception:
        return UNKNOWN_THICKNESS_MM


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
    """Legge pochi tag: modalita', descrizione, spessore, ImageType, kernel.

    Il kernel di ricostruzione si MOSTRA e non decide: non entra nel ranking
    delle serie. Serve al lettore, perche' densitometria e morfometria delle
    vie aeree lo vogliono opposto (vedi la sezione kernel del README) e la
    NIfTI, perdendo i tag, non lo porterebbe piu' da nessuna parte."""
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.LoadPrivateTagsOff()
    r.ReadImageInformation()

    def tag(t):
        try:
            return r.GetMetaData(t).strip()
        except Exception:
            return '?'
    return (tag('0008|0060'), tag('0008|103e'), tag('0018|0050'),
            tag('0008|0008'), tag('0018|1210'))


# Campi di acquisizione persistiti accanto alla NIfTI. La conversione perde i
# tag DICOM, quindi kernel, dose e scanner sparirebbero proprio quando servono:
# il kernel per leggere densitometria e pareti (sezione kernel del README), kV e
# mAs perche' il rumore dipende da li' e il rumore sposta ogni soglia, lo scanner
# perche' i confronti valgono solo a protocollo uguale.
ACQ_TAGS = (
    ('modality', '0008|0060'),
    ('series_description', '0008|103e'),
    ('slice_thickness_mm', '0018|0050'),
    ('kernel', '0018|1210'),
    ('kvp', '0018|0060'),
    ('exposure_mas', '0018|1152'),      # Exposure = mAs
    ('tube_current_ma', '0018|1151'),
    ('ctdivol_mgycm', '0018|9345'),     # spesso assente per-fetta: '?' e' normale
    ('manufacturer', '0008|0070'),
    ('model', '0008|1090'),
)


def acq_sidecar_path(ct_path):
    """Percorso del sidecar di acquisizione accanto a una NIfTI.

    X.nii.gz -> X.acq.json, X.nii -> X.acq.json. Puro: nessun I/O."""
    p = str(ct_path)
    for ext in ('.nii.gz', '.nii'):
        if p.endswith(ext):
            return p[:-len(ext)] + '.acq.json'
    return p + '.acq.json'


def merge_acquisition(seg_info, sidecar):
    """Innesta il sidecar sotto seg_info['acquisition'] senza sovrascrivere.

    Tollera sidecar None o vuoto: in quel caso seg_info torna intatto, cosi' una
    NIfTI fornita a mano (senza sidecar) non produce ne' errori ne' un blocco
    'acquisition' vuoto che sembrerebbe un dato mancante invece che assente.
    Puro: nessun I/O."""
    if seg_info is None:
        seg_info = {}
    if not sidecar:
        return seg_info
    seg_info.setdefault('acquisition', dict(sidecar))
    return seg_info


def read_acquisition(f):
    """Legge i parametri di acquisizione da UN file DICOM.

    Stesso patto di describe(): un campo assente vale '?', mai un'eccezione."""
    r = sitk.ImageFileReader()
    r.SetFileName(f)
    r.LoadPrivateTagsOff()
    try:
        r.ReadImageInformation()
    except Exception:
        return {name: '?' for name, _ in ACQ_TAGS}

    def tag(t):
        try:
            v = r.GetMetaData(t).strip()
        except Exception:
            return '?'
        return v or '?'
    return {name: tag(t) for name, t in ACQ_TAGS}


def _zpositions(files):
    """Coordinata z (ImagePositionPatient) di ogni file, nell'ordine dato.

    I file illeggibili prendono 0.0: non inventa una posizione, e il controllo a
    valle vedra' comunque una geometria incoerente."""
    r = sitk.ImageFileReader()
    r.LoadPrivateTagsOff()
    out = []
    for f in files:
        try:
            r.SetFileName(f)
            r.ReadImageInformation()
            out.append(float(r.GetMetaData('0020|0032').split('\\')[2]))
        except Exception:
            out.append(0.0)
    return out


def main(src, dst, pick=None):
    series = find_series(src)
    if not series:
        sys.exit('nessuna serie DICOM trovata in ' + src)
    items = list(series.items())            # [(sid, files)]
    metas = []
    for sid, files in items:
        mod, desc, thick, itype, kernel = describe(files[0])
        metas.append({'modality': mod, 'desc': desc, 'thickness': thick,
                      'image_type': itype, 'n_images': len(files),
                      'kernel': kernel})

    ranked = rank_series(metas)
    print(f'{len(items)} serie trovate (ordinate per idoneita\' all\'analisi):')
    for m in ranked:
        i = m['orig_index']
        flag = 'X ' + m['disq'] if m['disq'] else 'OK'
        print(f'  [{i}] {m["n_images"]:5d} imm  {m["modality"]:3s}  '
              f'strato {m["thickness"]} mm · kernel {m.get("kernel", "?")}  '
              f'[{flag}]  {m["desc"]}')

    if pick is not None:
        idx = int(pick)
        print(f'-> serie forzata da riga di comando: [{idx}] '
              f'(kernel {metas[idx].get("kernel", "?")})')
    else:
        best = next((m for m in ranked if m['disq'] is None), None)
        if best is None:
            best = ranked[0]
            print('ATTENZIONE: nessuna serie pienamente idonea; uso la meno peggio.')
        idx = best['orig_index']
        print(f'-> scelta automatica: [{idx}] (strato {best["thickness"]} mm, '
              f'kernel {best.get("kernel", "?")}, {best["n_images"]} immagini) '
              f'— {best["desc"]}')

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

    # Guardia di geometria. Un export DICOM difettoso puo' ripetere ogni fetta:
    # il reader ordina, meta' delle differenze consecutive vale 0 e lo spacing
    # calcolato crolla (caso reale: 0.08 mm contro 1 mm dichiarato). Va visto
    # PRIMA di scrivere, altrimenti la pipeline misura mm sbagliati in silenzio.
    zs = _zpositions(ordered)
    dup = duplicate_positions(zs)
    if dup['has_duplicates']:
        # Una sola immagine per posizione, poi RIORDINO PER z: con i duplicati
        # l'ordine del reader non e' monotono, e tenere il primo di ogni gruppo
        # lascerebbe una sequenza sfalsata da cui il reader ricava ancora uno
        # spacing sbagliato (misurato: 0.165 mm invece di 0.700).
        best = {}
        for z, f in zip(zs, ordered):
            best.setdefault(round(z * 1000), (z, f))
        ordered = [f for _, (z, f) in sorted(best.items())]

    reader.SetFileNames(ordered)
    img = reader.Execute()
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    sitk.WriteImage(img, dst)
    print('salvato', dst, img.GetSize(), 'spacing', [round(s, 2) for s in img.GetSpacing()])

    # tv e' la sentinella 999.0 quando lo spessore non e' leggibile: in quel caso
    # non c'e' un valore dichiarato da confrontare, non un valore assurdo.
    declared = None if tv >= UNKNOWN_THICKNESS_MM else tv
    # Provenienza di acquisizione accanto alla NIfTI: il formato non ha dove
    # tenerla, e senza questa il kernel non arriverebbe ne' al report ne' alla
    # riproducibilita'. Nessun campo mancante puo' far fallire la scrittura.
    try:
        side = read_acquisition(files[0])
        with open(acq_sidecar_path(dst), 'w', encoding='utf-8') as fh:
            json.dump(side, fh, indent=1, ensure_ascii=False)
        print('provenienza acquisizione:', acq_sidecar_path(dst),
              '| kernel', side.get('kernel', '?'))
    except Exception as e:
        print(f'nota: provenienza di acquisizione non scritta ({e})')

    banner = geometry_banner(img.GetSize(), img.GetSpacing(), declared, dup=dup)
    if banner:
        print('\n' + banner + '\n')


if __name__ == '__main__':
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
