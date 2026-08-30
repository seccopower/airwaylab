"""Controllo di plausibilita' della geometria del volume (nucleo puro).

Perche' serve. La selezione serie di anonymize.py giudica i METADATI (spessore
dichiarato, ImageType, numero di immagini), ma nessuno verifica la geometria che
esce davvero dal reader. Caso reale (CD paziente, serie "Torace HR 1.0 B60s"): il
supporto conteneva OGNI fetta duplicata — 834 file su 417 posizioni z, pixel
identici byte per byte, solo il SOPInstanceUID diverso. SimpleITK ordina le fette,
meta' delle differenze consecutive vale 0, e lo spacing calcolato crolla a 0.08 mm
contro il millimetro dichiarato: un errore di geometria di un fattore ~12. Il
volume veniva scritto senza un solo errore e l'intera pipeline avrebbe misurato
calibri e lunghezze in millimetri sbagliati, senza che nulla lo segnalasse.

Il warning di ITK ("Non uniform sampling or missing slices detected") passa come
rumore fra decine di righe: serve un controllo esplicito, in unita' cliniche.

Filosofia (coerente col resto del progetto). Le soglie sono LARGHE e ancorate a
fatti robusti di acquisizione, non a intervalli di normalita' stretti: nessuna TC
torace clinica ricostruisce sotto ~0.3 mm ne' copre meno di 10 cm. Un flag NON
significa "il dato e' inutilizzabile": significa "verifica la geometria prima di
credere ai millimetri". La coerenza fra spessore DICHIARATO e spacing MISURATO e'
il segnale piu' forte, perche' non dipende da quanto e' insolito il protocollo.

Puro: nessun I/O. Testato in tests/test_geometry_qc_core.py.
"""

# Soglie di acquisizione, non di normalita'. Volutamente larghe: devono lasciar
# passare qualunque TC torace reale e fermare solo le geometrie impossibili.
MIN_Z_SPACING_MM = 0.3     # sotto: nessuna ricostruzione clinica arriva qui
MAX_Z_SPACING_MM = 10.0    # sopra: non e' un volume, sono fette sparse
MIN_INPLANE_MM = 0.1
MAX_INPLANE_MM = 2.0
MIN_COVERAGE_MM = 100.0    # un torace ne copre ~250-350; sotto i 10 cm non lo e'
THICKNESS_RATIO = 3.0      # scarto tollerato fra spessore dichiarato e misurato


def duplicate_positions(z_positions, tol_mm=1e-3):
    """Individua posizioni di fetta ripetute in una serie.

    z_positions: sequenza di coordinate z (mm), nell'ordine dei file.

    Ritorna {'n_total', 'n_unique', 'n_duplicate', 'has_duplicates',
             'multiplicity'} dove multiplicity e' il numero massimo di file che
    condividono la stessa posizione (2 = ogni fetta scritta due volte).

    Le posizioni sono raggruppate arrotondando a tol_mm, cosi' il rumore di
    virgola mobile non crea falsi unici."""
    if tol_mm <= 0:
        raise ValueError('tol_mm deve essere > 0')
    counts = {}
    for z in z_positions:
        key = round(float(z) / tol_mm)
        counts[key] = counts.get(key, 0) + 1
    n_total = len(list(z_positions))
    n_unique = len(counts)
    return {
        'n_total': n_total,
        'n_unique': n_unique,
        'n_duplicate': n_total - n_unique,
        'has_duplicates': n_total > n_unique,
        'multiplicity': max(counts.values()) if counts else 0,
    }


def geometry_plausibility(size, spacing, declared_thickness_mm=None):
    """Valuta se la geometria di un volume e' plausibile per una TC torace.

    size:    (nx, ny, nz) in voxel
    spacing: (sx, sy, sz) in mm
    declared_thickness_mm: spessore dichiarato nei metadati DICOM, se noto.

    Ritorna {'ok': bool, 'flags': [{'code','severity','msg'}], 'coverage_mm': float}
    severity in {'alto','medio'}. 'ok' e' True se non ci sono flag."""
    if len(size) != 3 or len(spacing) != 3:
        raise ValueError('size e spacing devono avere 3 componenti')
    sx, sy, sz = (float(v) for v in spacing)
    nz = int(size[2])
    coverage = nz * sz
    flags = []

    def flag(code, severity, msg):
        flags.append({'code': code, 'severity': severity, 'msg': msg})

    if sz < MIN_Z_SPACING_MM:
        flag('z_spacing_troppo_piccolo', 'alto',
             f'spacing z {sz:.3f} mm sotto il minimo plausibile '
             f'({MIN_Z_SPACING_MM} mm): tipico di fette duplicate o posizioni '
             f'sovrapposte. Le misure in mm sarebbero sbagliate di un fattore '
             f'{MIN_Z_SPACING_MM / sz:.0f} o piu.' if sz > 0 else
             f'spacing z {sz:.3f} mm non valido.')
    elif sz > MAX_Z_SPACING_MM:
        flag('z_spacing_troppo_grande', 'alto',
             f'spacing z {sz:.2f} mm oltre il massimo plausibile '
             f'({MAX_Z_SPACING_MM} mm): non e\' un volume contiguo.')

    for name, val in (('x', sx), ('y', sy)):
        if not (MIN_INPLANE_MM <= val <= MAX_INPLANE_MM):
            flag(f'inplane_{name}_implausibile', 'alto',
                 f'spacing {name} {val:.3f} mm fuori dall\'intervallo '
                 f'{MIN_INPLANE_MM}-{MAX_INPLANE_MM} mm.')

    if coverage < MIN_COVERAGE_MM:
        flag('copertura_insufficiente', 'alto',
             f'copertura z {coverage:.0f} mm sotto {MIN_COVERAGE_MM:.0f} mm: '
             f'{nz} fette x {sz:.3f} mm non coprono un torace.')

    if declared_thickness_mm:
        t = float(declared_thickness_mm)
        if t > 0 and sz > 0:
            ratio = max(t / sz, sz / t)
            if ratio >= THICKNESS_RATIO:
                flag('spessore_incoerente', 'alto',
                     f'spessore dichiarato {t:g} mm ma spacing misurato '
                     f'{sz:.3f} mm (fattore {ratio:.0f}x): i metadati e la '
                     f'geometria non concordano.')

    return {'ok': not flags, 'flags': flags, 'coverage_mm': coverage}


def grids_match(size_a, spacing_a, size_b, spacing_b, spacing_tol_mm=1e-3):
    """Due volumi condividono la stessa griglia di voxel?

    Serve alle cache di maschere: riusare una maschera calcolata su una GRIGLIA
    diversa dalla CT corrente la rende inutilizzabile (caso reale: maschere di una
    conversione precedente, 834 fette a 0.082 mm, riproposte su una CT corretta di
    417 fette a 0.700 mm — la maschera risultava vuota sulla griglia della CT).
    Confrontare i file per data non basta: solo la geometria dice la verita'.

    Ritorna {'ok': bool, 'reason': str|None}. reason e' leggibile a schermo."""
    sa, sb = tuple(int(v) for v in size_a), tuple(int(v) for v in size_b)
    if sa != sb:
        return {'ok': False, 'reason': f'dimensioni {sa} vs {sb}'}
    pa = tuple(float(v) for v in spacing_a)
    pb = tuple(float(v) for v in spacing_b)
    if any(abs(x - y) > spacing_tol_mm for x, y in zip(pa, pb)):
        fmt = lambda t: '(' + ', '.join(f'{v:.3f}' for v in t) + ')'  # noqa: E731
        return {'ok': False, 'reason': f'spacing {fmt(pa)} vs {fmt(pb)} mm'}
    return {'ok': True, 'reason': None}


def geometry_banner(size, spacing, declared_thickness_mm=None, dup=None):
    """Righe di testo per il terminale. Stringa vuota se non c'e' nulla da dire."""
    out = []
    if dup and dup.get('has_duplicates'):
        out.append(
            f'*** ATTENZIONE: {dup["n_duplicate"]} fette con posizione z ripetuta '
            f'({dup["n_total"]} file su {dup["n_unique"]} posizioni, '
            f'molteplicita\' {dup["multiplicity"]}x). Export DICOM difettoso: '
            f'tengo una sola immagine per posizione. ***')
    res = geometry_plausibility(size, spacing, declared_thickness_mm)
    for f in res['flags']:
        out.append(f'*** ATTENZIONE ({f["severity"]}): {f["msg"]} ***')
    # La riga finale vale solo se la geometria e' ANCORA incoerente: dopo una
    # deduplica riuscita lo spacing torna corretto e allarmare sarebbe falso.
    if res['flags']:
        out.append('    La quantificazione in millimetri NON e\' affidabile '
                   'finche\' la geometria non torna.')
    return '\n'.join(out)
