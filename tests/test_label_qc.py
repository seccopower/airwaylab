"""Test della guardia di completezza del labeling (label_qc.py). Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from label_qc import LOBES, labeling_banner_html, labeling_status   # noqa: E402


def test_complete():
    st = labeling_status(LOBES)
    assert st['complete'] and st['missing'] == [] and st['n_present'] == 6
    assert labeling_banner_html(LOBES) == ''


def test_incompleto_casoDAS():
    # esattamente casoDAS: solo RML e RLL etichettati
    st = labeling_status(['RML', 'RLL'])
    assert not st['complete']
    assert st['n_present'] == 2
    assert set(st['missing']) == {'RUL', 'LUL', 'LING', 'LLL'}
    banner = labeling_banner_html(['RML', 'RLL'])
    assert 'INCOMPLETA' in banner and 'RUL' in banner and 'LLL' in banner


def test_ignora_central_e_sconosciuti():
    # CENTRAL e valori non-lobari non contano come lobi
    st = labeling_status(['RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL', 'CENTRAL', 'x'])
    assert st['complete'] and st['n_present'] == 6
