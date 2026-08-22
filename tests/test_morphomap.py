"""Test del nucleo puro della mappa strutturale multi-asse (morphomap_core.py).
Nessun dato paziente: solo aritmetica su territori sintetici."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

import numpy as np   # noqa: E402

from morphomap_core import (   # noqa: E402
    aggregate_lobes,
    classify_lobe,
    tissue_fraction,
    voxelwise_destruction,
)


def test_voxelwise_destruction_noncubic():
    # griglia ds NON cubica (2x1x1), fattore 3 per asse -> iso (6,3,3)
    ds_shape = (2, 1, 1)
    ct = np.full((6, 3, 3), 0.0, np.float32)   # tutto tessuto (0 HU)
    # prima meta' in z-blocco 0: enfisema totale; il resto tessuto
    ct[0:3] = -1000.0                            # cella ds [0]: tutta aria/enfisema
    laa, ftis = voxelwise_destruction(ct, ds_shape, laa_hu=-950.0)
    assert laa.shape == ds_shape
    assert laa[0, 0, 0] == 1.0 and laa[1, 0, 0] == 0.0
    assert ftis[0, 0, 0] == 0.0 and ftis[1, 0, 0] == 1.0


def test_voxelwise_destruction_partial_block():
    # una cella ds, blocco 3x3x3: 9 sotto-voxel enfisematosi su 27 -> laa 1/3
    ct = np.full((3, 3, 3), -800.0, np.float32)   # f_tissue 0.2, non enfisema
    ct[0] = -1000.0                               # 9 voxel < -950
    laa, ftis = voxelwise_destruction(ct, (1, 1, 1), laa_hu=-950.0)
    assert abs(laa[0, 0, 0] - 9 / 27) < 1e-6
    # f_tissue: (9*0 + 18*0.2)/27 = 0.1333
    assert abs(ftis[0, 0, 0] - (18 * 0.2) / 27) < 1e-6


def test_tissue_fraction_clip():
    assert tissue_fraction(-1000) == 0.0      # aria pura
    assert tissue_fraction(0) == 1.0          # tessuto/acqua
    assert tissue_fraction(500) == 1.0        # oltre acqua -> clip a 1
    assert abs(tissue_fraction(-800) - 0.2) < 1e-9   # parenchima ~sano
    assert abs(tissue_fraction(-930) - 0.07) < 1e-9  # enfisema


def test_aggregate_axes_kept_separate():
    # due lobi con la STESSA conduttanza ma distruzione opposta
    terr = [
        {'lobe': 'RLL', 'q': 100.0, 'laa': 0.70, 'f_tissue': 0.06, 'n': 1000},
        {'lobe': 'RUL', 'q': 100.0, 'laa': 0.10, 'f_tissue': 0.14, 'n': 1000},
    ]
    per, glob = aggregate_lobes(terr)
    # conduttanza uguale...
    assert per['RLL']['cond_frac'] == per['RUL']['cond_frac'] == 0.5
    # ...ma il carico (q*laa) e' tutto nel lobo a bassa attenuazione
    assert per['RLL']['ds_flux'] == 70.0 and per['RUL']['ds_flux'] == 10.0
    assert per['RLL']['ds_share'] == 0.875   # 70/80
    # numero globale: (70+10)/(200) = 0.40
    assert glob['cond_to_destroyed'] == 0.40


def test_ds_share_weights_conductance():
    # un lobo con alta bassa-attenuazione ma conduttanza trascurabile contribuisce poco
    terr = [
        {'lobe': 'RLL', 'q': 200.0, 'laa': 0.50, 'f_tissue': 0.10, 'n': 500},
        {'lobe': 'RML', 'q': 1.0, 'laa': 0.90, 'f_tissue': 0.05, 'n': 500},
    ]
    per, _ = aggregate_lobes(terr)
    assert per['RLL']['ds_share'] > per['RML']['ds_share']


def test_volume_weighted_means():
    # due territori nello stesso lobo: media pesata sul volume, non semplice
    terr = [
        {'lobe': 'RLL', 'q': 10.0, 'laa': 0.80, 'f_tissue': 0.05, 'n': 300},
        {'lobe': 'RLL', 'q': 10.0, 'laa': 0.20, 'f_tissue': 0.15, 'n': 100},
    ]
    per, _ = aggregate_lobes(terr)
    # (0.80*300 + 0.20*100)/400 = 0.65
    assert per['RLL']['laa'] == 0.65
    assert per['RLL']['n_vox'] == 400


def test_negative_flow_ignored():
    terr = [
        {'lobe': 'RLL', 'q': -5.0, 'laa': 0.50, 'f_tissue': 0.10, 'n': 100},
        {'lobe': 'RUL', 'q': 100.0, 'laa': 0.10, 'f_tissue': 0.14, 'n': 100},
    ]
    per, glob = aggregate_lobes(terr)
    assert per['RLL']['cond_q'] == 0.0
    assert per['RLL']['ds_flux'] == 0.0
    # solo il flusso positivo entra nel denominatore
    assert glob['cond_to_destroyed'] == round(10.0 / 100.0, 3)


def test_absent_lobe_omitted():
    terr = [{'lobe': 'RLL', 'q': 10.0, 'laa': 0.5, 'f_tissue': 0.1, 'n': 100}]
    per, _ = aggregate_lobes(terr)
    assert set(per) == {'RLL'}


def test_classify_lobe():
    # etichette riferite a soglia esplicita
    assert classify_lobe(0.72, 0.49) == 'LAA ≥40% e quota flusso simulato ≥20% (soglie espl.)'
    assert classify_lobe(0.48, 0.13) == 'LAA ≥40%, quota flusso simulato <20% (soglie espl.)'
    assert classify_lobe(0.12, 0.05) == 'LAA <25% (soglia espl.)'
    assert classify_lobe(0.30, 0.20) == 'intermedio (soglie espl.)'
    # niente claim funzionali né 'conduttanza' (q è quota di flusso simulato)
    for laa, sh in [(0.72, 0.49), (0.48, 0.13), (0.12, 0.05), (0.30, 0.20)]:
        lab = classify_lobe(laa, sh)
        for bad in ('spazio morto', 'distrut', 'conservato', 'conduttanza'):
            assert bad not in lab
    # dati mancanti
    assert classify_lobe(None, 0.5) == 'dati insufficienti'
    assert classify_lobe(0.5, None) == 'dati insufficienti'
