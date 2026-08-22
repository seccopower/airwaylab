"""Test del nucleo puro della mappa strutturale multi-asse (morphomap_core.py).
Nessun dato paziente: solo aritmetica su territori sintetici."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from morphomap_core import (   # noqa: E402
    aggregate_lobes,
    classify_lobe,
    tissue_fraction,
)


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
    # ...ma il mismatch (q*laa) e' tutto nel lobo distrutto
    assert per['RLL']['ds_flux'] == 70.0 and per['RUL']['ds_flux'] == 10.0
    assert per['RLL']['ds_share'] == 0.875   # 70/80
    # numero globale: (70+10)/(200) = 0.40
    assert glob['cond_to_destroyed'] == 0.40


def test_ds_share_weights_conductance():
    # un lobo molto distrutto ma con conduttanza trascurabile contribuisce poco
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
    # distrutto + conduttanza alta -> candidato spazio morto
    assert 'spazio morto' in classify_lobe(0.72, 0.49)
    # distrutto ma conduttanza contenuta
    assert classify_lobe(0.48, 0.13) == 'distrutto, conduttanza contenuta'
    # conservato
    assert classify_lobe(0.12, 0.05) == 'parenchima conservato'
    # intermedio
    assert classify_lobe(0.30, 0.20) == 'intermedio'
    # dati mancanti
    assert classify_lobe(None, 0.5) == 'dati insufficienti'
    assert classify_lobe(0.5, None) == 'dati insufficienti'
