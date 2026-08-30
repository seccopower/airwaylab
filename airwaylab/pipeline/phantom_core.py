"""Fantoccio digitale di via aerea con PSF e rumore (nucleo puro).

Perche' esiste. Il floor di risoluzione del calibro (`VOXELS_FLOOR = 3.0` in
qc_params.py) e' scelto a priori: il README lo dichiara "a heuristic processing
bound, not a validated physical resolution limit". Sul caso di riferimento costa
caro — 63 rami su 218 (29%) con calibro riportabile, e i declassati stanno appena
sotto la soglia (mediana d_maschera/floor = 0.70): un floor a 2 voxel invece di 3
ne renderebbe riportabili 143 (66%). Quel prezzo si paga oggi su un numero non
verificato, che puo' stare buttando via misure vere oppure essere corretto.

Un fantoccio FISICO non e' disponibile ne' lo sara' (backlog, sezione F). Ma il
floor e' dichiarato un limite di PROCESSO, non fisico, e un limite di processo si
caratterizza simulando il processo. Il tubo digitale gia' presente nei test
(`tests/test_section.py`) e' un bordo a gradino ideale: nessuna sfocatura,
nessun rumore, e per questo recupera bene 4, 8 e 14 mm senza dire nulla su dove
il metodo si rompe.

Questo modulo aggiunge i due ingredienti mancanti, nell'ordine in cui l'immagine
si forma davvero:

    oggetto continuo  ->  PSF  ->  integrazione nel voxel  ->  rumore

Il tubo viene costruito su una griglia FINE, sfocato con una gaussiana espressa
in mm, poi mediato a blocchi fino allo spacing di destinazione (l'integrazione
del voxel) e infine sporcato di rumore. Costruirlo direttamente allo spacing
grosso salterebbe l'integrazione e darebbe bordi piu' netti del vero.

Cosa questo NON e'. Non e' accuratezza metrologica contro uno standard
tracciabile: e' la curva di bias e dispersione dello stimatore half-max su un
bordo sintetico. La sigma della PSF e il livello di rumore sono parametri, non
misure di uno scanner reale. Chi legge il risultato deve leggerlo cosi'.

Puro: nessun I/O, deterministico a parita' di seed. Testato in
tests/test_phantom_core.py.
"""
import numpy as np
from scipy import ndimage

# Attenuazioni di riferimento, coerenti col fantoccio gia' usato nei test.
HU_LUMEN = -1000.0      # aria nel lume
HU_WALL = 0.0           # parete, tessuto molle
HU_PARENCHYMA = -850.0  # parenchima circostante

SUPERSAMPLE = 5         # voxel fini per voxel di destinazione, per lato


def _fine_grid_tube(lumen_d_mm, wall_mm, fine_mm, half_field_mm):
    """Sezione del tubo su griglia fine, asse lungo z. Ritorna un piano 2D:
    l'oggetto e' invariante lungo l'asse, quindi basta costruirlo una volta."""
    n = int(round(2 * half_field_mm / fine_mm))
    if n < 4:
        raise ValueError('campo troppo piccolo per lo spacing fine richiesto')
    c = (n - 1) / 2.0
    idx = np.arange(n)
    yy, xx = np.meshgrid(idx, idx, indexing='ij')
    r = np.hypot((yy - c) * fine_mm, (xx - c) * fine_mm)
    sl = np.full((n, n), HU_PARENCHYMA, dtype=np.float64)
    sl[r <= lumen_d_mm / 2.0 + wall_mm] = HU_WALL
    sl[r <= lumen_d_mm / 2.0] = HU_LUMEN
    return sl


def _block_mean(a, k):
    """Media a blocchi k x k: modella l'integrazione del voxel."""
    ny, nx = a.shape
    ny -= ny % k
    nx -= nx % k
    return a[:ny, :nx].reshape(ny // k, k, nx // k, k).mean(axis=(1, 3))


def synth_tube(lumen_d_mm, wall_mm, spacing_mm, psf_sigma_mm=None,
               noise_hu=0.0, n_slices=16, seed=0, half_field_mm=None):
    """Volume 3D di un tubo assiale, con PSF, integrazione del voxel e rumore.

    lumen_d_mm    diametro vero del lume
    wall_mm       spessore vero della parete
    spacing_mm    spacing isotropo di destinazione
    psf_sigma_mm  sigma della gaussiana di sfocatura; None = nessuna PSF
    noise_hu      deviazione standard del rumore gaussiano aggiunto
    seed          rende il rumore riproducibile (CI stabile)

    Ritorna (volume zyx, centro zyx). L'asse del tubo e' z."""
    if not (lumen_d_mm > 0 and spacing_mm > 0):
        raise ValueError('diametro e spacing devono essere positivi')
    if half_field_mm is None:
        half_field_mm = max(6.0, lumen_d_mm / 2.0 + wall_mm + 4.0)
    fine_mm = spacing_mm / SUPERSAMPLE

    sl = _fine_grid_tube(lumen_d_mm, wall_mm, fine_mm, half_field_mm)
    if psf_sigma_mm:
        # la sfocatura vive in mm: sulla griglia fine sono piu' voxel
        sl = ndimage.gaussian_filter(sl, sigma=float(psf_sigma_mm) / fine_mm,
                                     mode='nearest')
    sl = _block_mean(sl, SUPERSAMPLE)

    vol = np.repeat(sl[None, :, :], n_slices, axis=0).astype(np.float32)
    if noise_hu:
        rng = np.random.default_rng(seed)
        vol = vol + rng.normal(0.0, float(noise_hu), size=vol.shape).astype(np.float32)
    ny, nx = sl.shape
    center = np.array([n_slices // 2, (ny - 1) / 2.0, (nx - 1) / 2.0], dtype=float)
    return vol, center


def recovery_stats(measured, d_true):
    """Bias e dispersione di un insieme di misure dello stesso diametro vero.

    Ritorna {'n', 'median', 'bias_mm', 'bias_frac', 'iqr_mm', 'cv'}; campi None
    se non c'e' abbastanza materiale. `n` conta le misure RIUSCITE: i fallimenti
    (analyze_section che ritorna None) vanno contati a parte dal chiamante,
    perche' "non misurabile" e' un esito diverso da "misurato male"."""
    m = np.asarray([x for x in measured if x is not None], dtype=float)
    out = {'n': int(m.size), 'median': None, 'bias_mm': None,
           'bias_frac': None, 'iqr_mm': None, 'cv': None}
    if m.size == 0 or not d_true > 0:
        return out
    med = float(np.median(m))
    out['median'] = med
    out['bias_mm'] = med - float(d_true)
    out['bias_frac'] = (med - float(d_true)) / float(d_true)
    if m.size >= 4:
        q1, q3 = np.percentile(m, [25, 75])
        out['iqr_mm'] = float(q3 - q1)
        out['cv'] = float(np.std(m) / med) if med else None
    return out


def with_response_slope(rows):
    """Aggiunge a ogni riga la pendenza locale d(misurato)/d(vero).

    E' la grandezza che definisce un floor di RISOLUZIONE. Il README descrive
    cosi' il regime sotto il floor: la misura "carries no size information" —
    cioe' smette di rispondere al vero. Una pendenza vicina a 1 dice che la
    misura segue ancora l'oggetto; una pendenza che collassa verso 0 dice che
    non lo segue piu', quale che sia il suo scostamento assoluto.

    Il BIAS non serve a questo scopo e non va confuso con esso: sul fantoccio
    e' un offset quasi costante in mm (la PSF erode il bordo allo stesso modo a
    ogni calibro), quindi in frazione del vero cresce al ridursi del diametro e
    boccerebbe i rami piccoli per un errore SISTEMATICO, che e' calibrabile e
    non e' una perdita di informazione. Resta riportato come diagnostica."""
    out = [dict(r) for r in sorted(rows, key=lambda r: r['d_true'])]
    prev = None
    for r in out:
        r['slope'] = None
        med = r.get('median')
        if med is not None and prev is not None:
            dd = r['d_true'] - prev[1]
            if dd > 0:
                r['slope'] = (med - prev[0]) / dd
        if med is not None:
            prev = (med, r['d_true'])
    return out


def floor_from_sweep(rows, min_slope=0.5, max_cv=0.10, min_success=0.9):
    """Il floor implicito in uno sweep: il diametro vero piu' PICCOLO a partire
    dal quale la misura e' ancora utile, e resta tale per tutti i maggiori.

    Tre criteri, tutti necessari:
      success_frac >= min_success   la sezione e' misurabile
      slope        >= min_slope     la misura RISPONDE ancora al vero
      cv           <= max_cv        e lo fa con precisione utilizzabile

    rows: righe gia' aggregate per diametro; la pendenza viene calcolata qui.

    floor_mm e' None se nessun diametro soddisfa i criteri — esito informativo,
    non errore: significa che il floor sta oltre il campo esplorato."""
    rows = with_response_slope(rows)
    ok = []
    for r in rows:
        good = (r.get('success_frac', 1.0) >= min_success
                and (r.get('cv') is None or r['cv'] <= max_cv)
                and (r.get('slope') is None or r['slope'] >= min_slope))
        ok.append((r['d_true'], good))
    floor = None
    # dal fondo: il primo diametro che rompe la catena fissa il floor sopra di se
    for d, good in reversed(ok):
        if good:
            floor = d
        else:
            break
    return {'floor_mm': floor, 'min_slope': min_slope, 'max_cv': max_cv,
            'min_success': min_success, 'n_diameters': len(ok),
            'all_good': all(g for _, g in ok) if ok else False,
            'rows': rows}
