"""Test del QC di leak/connettivita' (leak_qc_core). Puro.

Copre le metriche che la vecchia guardia 'fuori dalla maschera polmonare' NON
vedeva: leak interni (radius-explosion) e isole staccate."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from leak_qc_core import (   # noqa: E402
    connectivity_flag,
    leak_summary,
    radius_explosion,
)


def test_radius_explosion_intercetta_il_leak_in_cisti():
    # ramo distale che raddoppia il calibro del genitore = leak in una bolla
    br = [
        {'id': 'b1', 'aid': 'RLL', 'gen': 5, 'd_mask': 6.0, 'parent_d': 6.5},   # ok, tapering
        {'id': 'b2', 'aid': None, 'gen': 9, 'd_mask': 22.0, 'parent_d': 3.0},   # 3->22mm: leak
    ]
    ex = radius_explosion(br)
    assert len(ex) == 1 and ex[0]['id'] == 'b2'
    assert ex[0]['ratio'] >= 1.6


def test_radius_explosion_ignora_le_vie_centrali():
    # la transizione trachea->principale e' larga per anatomia: non e' un leak
    br = [{'id': 'rmb', 'aid': 'RMB', 'gen': 4, 'd_mask': 20.0, 'parent_d': 10.0}]
    assert radius_explosion(br) == []


def test_radius_explosion_ignora_il_rumore_sottile():
    # sotto floor_mm i rapporti sono rumorosi: niente flag
    br = [{'id': 'x', 'aid': None, 'gen': 14, 'd_mask': 2.5, 'parent_d': 1.0}]
    assert radius_explosion(br) == []


def test_radius_explosion_robusto_alla_generazione():
    # nessuna dipendenza dalla gen: un figlio piu' largo del padre e' sempre sospetto
    br = [{'id': 'y', 'aid': None, 'gen': 2, 'd_mask': 9.0, 'parent_d': 5.0}]
    assert len(radius_explosion(br)) == 1


def test_connettivita_segnala_isole():
    f = connectivity_flag(n_components=7, largest_frac=0.90, leaked_ml=8.0)
    assert f and f['code'] == 'islands' and f['severity'] == 'alto'


def test_connettivita_ok_con_specki_minimi():
    # albero unico con qualche voxel sparso: nessun flag
    assert connectivity_flag(n_components=3, largest_frac=0.999, leaked_ml=0.1) is None


def test_summary_pulito_e_ok():
    s = leak_summary(explosion=[], n_components=2, largest_frac=0.999,
                     leaked_ml=0.05, total_ml=100.0)
    assert s['ok'] and s['flags'] == []
    assert s['metrics']['n_radius_explosion'] == 0
    assert s['metrics']['airway_total_ml'] == 100.0


def test_summary_casoDAS_style_flagga():
    ex = [{'id': 'b2', 'gen': 9, 'd_mask': 22.0, 'parent_d': 3.0, 'ratio': 7.33}]
    s = leak_summary(explosion=ex, n_components=9, largest_frac=0.90,
                     leaked_ml=6.0, total_ml=120.0)
    assert not s['ok']
    codes = {f['code'] for f in s['flags']}
    assert 'radius_explosion' in codes and 'islands' in codes
    assert s['metrics']['radius_explosion_worst']['id'] == 'b2'
