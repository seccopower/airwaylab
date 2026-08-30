"""Test del fantoccio digitale con PSF e rumore (phantom_core) — backlog #27.

Il tubo dei test storici e' un bordo a gradino ideale e per questo non dice
nulla su dove il metodo si rompe. Qui si verificano gli ingredienti che
mancavano — sfocatura, integrazione del voxel, rumore — e il criterio di floor
che ne deriva. Nessun dato di paziente, deterministico a parita' di seed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from phantom_core import (   # noqa: E402
    HU_LUMEN, HU_PARENCHYMA, HU_WALL,
    floor_from_sweep, recovery_stats, synth_tube, with_response_slope,
)


def test_tubo_ideale_ha_i_tre_livelli():
    vol, c = synth_tube(6.0, 1.0, 0.7)
    z, y, x = (int(round(v)) for v in c)
    assert abs(vol[z, y, x] - HU_LUMEN) < 30            # centro: aria
    assert vol.min() <= HU_LUMEN + 30
    assert vol.max() >= HU_WALL - 60                    # parete presente
    # lontano dal tubo: parenchima
    assert abs(float(vol[z, 1, 1]) - HU_PARENCHYMA) < 60


def test_psf_ammorbidisce_il_bordo():
    """Con la sfocatura il salto lume-parete deve diventare meno ripido."""
    sharp, c = synth_tube(6.0, 1.0, 0.7, psf_sigma_mm=None)
    blur, _ = synth_tube(6.0, 1.0, 0.7, psf_sigma_mm=1.0)
    z, y, _ = (int(round(v)) for v in c)
    g_sharp = np.abs(np.diff(sharp[z, y, :])).max()
    g_blur = np.abs(np.diff(blur[z, y, :])).max()
    assert g_blur < g_sharp


def test_rumore_deterministico_e_dimensionato():
    a, _ = synth_tube(6.0, 1.0, 0.7, noise_hu=50.0, seed=3)
    b, _ = synth_tube(6.0, 1.0, 0.7, noise_hu=50.0, seed=3)
    c, _ = synth_tube(6.0, 1.0, 0.7, noise_hu=50.0, seed=4)
    assert np.array_equal(a, b)          # stesso seed -> identico
    assert not np.array_equal(a, c)      # seed diverso -> diverso
    clean, _ = synth_tube(6.0, 1.0, 0.7, noise_hu=0.0)
    assert 30.0 < float(np.std(a - clean)) < 70.0


def test_ingressi_non_validi():
    for d, sp in ((0.0, 0.7), (-1.0, 0.7), (4.0, 0.0)):
        try:
            synth_tube(d, 1.0, sp)
        except ValueError:
            continue
        raise AssertionError(f'atteso ValueError per d={d} spacing={sp}')


def test_recovery_stats_bias_e_dispersione():
    st = recovery_stats([3.8, 3.9, 4.0, 4.1, 4.2], 4.0)
    assert st['n'] == 5
    assert abs(st['median'] - 4.0) < 1e-9
    assert abs(st['bias_mm']) < 1e-9 and abs(st['bias_frac']) < 1e-9
    assert st['iqr_mm'] is not None and st['cv'] is not None


def test_recovery_stats_ignora_i_fallimenti():
    st = recovery_stats([None, 2.0, None, 2.0], 2.0)
    assert st['n'] == 2 and st['median'] == 2.0
    vuoto = recovery_stats([None, None], 2.0)
    assert vuoto['n'] == 0 and vuoto['median'] is None


def test_pendenza_locale():
    rows = [{'d_true': 1.0, 'median': 0.5}, {'d_true': 2.0, 'median': 1.5},
            {'d_true': 3.0, 'median': 1.6}]
    out = with_response_slope(rows)
    assert out[0]['slope'] is None                 # primo punto: nessuna
    assert abs(out[1]['slope'] - 1.0) < 1e-9       # risponde
    assert abs(out[2]['slope'] - 0.1) < 1e-9       # ha smesso di rispondere


def test_floor_e_il_primo_diametro_di_una_catena_buona():
    """Un punto cattivo in mezzo alza il floor sopra di se: contano solo i
    diametri buoni CONTIGUI dall'alto."""
    rows = [
        {'d_true': 1.0, 'median': 0.2, 'cv': 0.5, 'success_frac': 1.0},
        {'d_true': 2.0, 'median': 1.2, 'cv': 0.02, 'success_frac': 1.0},
        {'d_true': 3.0, 'median': 1.3, 'cv': 0.02, 'success_frac': 1.0},  # piatto
        {'d_true': 4.0, 'median': 2.3, 'cv': 0.02, 'success_frac': 1.0},
        {'d_true': 5.0, 'median': 3.3, 'cv': 0.02, 'success_frac': 1.0},
    ]
    assert floor_from_sweep(rows)['floor_mm'] == 4.0


def test_floor_none_se_nulla_e_utilizzabile():
    rows = [{'d_true': d, 'median': 1.0, 'cv': 0.9, 'success_frac': 0.1}
            for d in (1.0, 2.0, 3.0)]
    fl = floor_from_sweep(rows)
    assert fl['floor_mm'] is None and fl['all_good'] is False


def test_il_bias_costante_non_alza_il_floor():
    """Il punto metodologico: un offset SISTEMATICO in mm non e' perdita di
    informazione. Una serie con bias grande ma pendenza 1 e CV basso deve
    passare, altrimenti si scarterebbero i rami piccoli per un errore
    calibrabile."""
    rows = [{'d_true': d, 'median': d - 0.6, 'cv': 0.02, 'success_frac': 1.0}
            for d in (1.0, 2.0, 3.0, 4.0)]
    fl = floor_from_sweep(rows)
    assert fl['floor_mm'] == 1.0 and fl['all_good'] is True


def test_sweep_reale_il_metodo_degrada_al_ridursi_del_diametro():
    """Test di integrazione con lo stimatore VERO della pipeline: la precisione
    deve peggiorare al calare del calibro, ed e' cio' che definisce un floor."""
    from lumen import analyze_section
    t = np.array([1.0, 0.0, 0.0])
    sp = 0.7
    cv = {}
    for vox in (2.0, 6.0):
        d = vox * sp
        ms = []
        for k in range(5):
            vol, c = synth_tube(d, 1.0, sp, psf_sigma_mm=0.42,
                                noise_hu=60.0, seed=100 * k + 1)
            sec = analyze_section(vol, c, t, r_est_mm=d / 2.0, iso=sp)
            if sec is not None:
                ms.append(sec['d_eq'])
        cv[vox] = recovery_stats(ms, d)['cv']
    assert cv[2.0] is not None and cv[6.0] is not None
    assert cv[2.0] > cv[6.0], f'atteso degrado: {cv}'


def test_sotto_un_diametro_minimo_la_sezione_non_e_misurabile():
    """analyze_section rifiuta r_est sotto 0.6 mm: sotto ~1.7 voxel a 0.7 mm
    non c'e' proprio misura, il che e' gia' un floor implicito nel codice."""
    from lumen import analyze_section
    t = np.array([1.0, 0.0, 0.0])
    d = 1.0 * 0.7
    vol, c = synth_tube(d, 1.0, 0.7, psf_sigma_mm=0.42, noise_hu=60.0, seed=1)
    assert analyze_section(vol, c, t, r_est_mm=d / 2.0, iso=0.7) is None
