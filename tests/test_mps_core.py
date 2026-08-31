"""Test del Mucus Plug Score segmentario (mps_core).

Il criterio per segmento e' binario: quello che va fissato e' che tappi multipli
nello stesso segmento non gonfino il punteggio, e che i tappi non assegnabili
restino visibili invece di sparire. Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from mps_core import (   # noqa: E402
    SEGMENT_CODES, SEGMENT_NAMES, SEGMENTS_18,
    mucus_plug_score, phenotype_for,
)


def test_sono_esattamente_18_segmenti_distinti():
    assert len(SEGMENTS_18) == 18
    assert len(set(SEGMENT_CODES)) == 18
    assert len(SEGMENT_NAMES) == 18
    # destra 10, sinistra 8: a sinistra LB1+2 e LB7+8 sono fusi
    assert sum(1 for c in SEGMENT_CODES if c.startswith('RB')) == 10
    assert sum(1 for c in SEGMENT_CODES if c.startswith('LB')) == 8
    assert 'LB1+2' in SEGMENT_CODES and 'LB7+8' in SEGMENT_CODES


def test_nessun_tappo_score_zero():
    r = mucus_plug_score([])
    assert r['mps'] == 0
    assert r['phenotype'] == 'none'
    assert r['segments_occluded'] == []
    assert r['n_plugs'] == 0
    assert set(r['per_segment'].values()) == {0}


def test_per_segment_ha_sempre_18_chiavi():
    for arg in ([], ['RB1'], ['RB1', None, 'BOH']):
        r = mucus_plug_score(arg)
        assert len(r['per_segment']) == 18
        assert set(r['per_segment']) == set(SEGMENT_CODES)


def test_tre_segmenti_distinti_fenotipo_basso():
    r = mucus_plug_score(['RB1', 'RB4', 'LB9'])
    assert r['mps'] == 3
    assert r['phenotype'] == 'low'
    assert r['n_plugs'] == 3


def test_quattro_segmenti_fenotipo_alto():
    r = mucus_plug_score(['RB1', 'RB4', 'LB9', 'LB10'])
    assert r['mps'] == 4
    assert r['phenotype'] == 'high'


def test_un_solo_segmento_e_gia_basso():
    assert mucus_plug_score(['RB6'])['phenotype'] == 'low'


def test_due_tappi_nello_stesso_segmento_contano_uno():
    r = mucus_plug_score(['RB1', 'RB1'])
    assert r['mps'] == 1                       # il criterio e' binario
    assert r['n_plugs'] == 2                   # ma i tappi restano visibili
    assert r['segments_occluded'] == ['RB1']
    assert r['per_segment']['RB1'] == 1


def test_tappo_non_assegnato_non_entra_nello_score_ma_e_riportato():
    """Un tappo non mappato ABBASSA lo score: va dichiarato, non nascosto."""
    r = mucus_plug_score(['RB1', None, None])
    assert r['mps'] == 1
    assert r['n_unassigned'] == 2
    assert r['n_plugs'] == 3
    assert r['n_unknown_code'] == 0


def test_codice_fuori_lista_conteggiato_a_parte():
    r = mucus_plug_score(['RB1', 'RB11', 'lb3', ''])
    assert r['mps'] == 1                       # solo RB1 e' valido
    assert r['n_unknown_code'] == 3            # RB11, minuscolo, stringa vuota
    assert r['n_unassigned'] == 0
    assert r['n_plugs'] == 4


def test_tutti_i_segmenti_occlusi():
    r = mucus_plug_score(list(SEGMENT_CODES))
    assert r['mps'] == 18
    assert r['phenotype'] == 'high'
    assert set(r['per_segment'].values()) == {1}


def test_ordine_canonico_non_quello_di_arrivo():
    """Due casi con gli stessi segmenti devono essere confrontabili senza
    riordinare a valle."""
    a = mucus_plug_score(['LB10', 'RB4', 'RB1'])
    b = mucus_plug_score(['RB1', 'LB10', 'RB4'])
    assert a['segments_occluded'] == b['segments_occluded'] == ['RB1', 'RB4', 'LB10']


def test_provenienza_e_soglie_dichiarate():
    r = mucus_plug_score(['RB1'])
    assert r['status'] == 'exploratory'
    assert r['method_id'] == 'mps_airwaylab'
    assert r['thresholds'] == {'low': '1-3', 'high': '>=4'}


def test_soglie_non_condivise_fra_chiamate():
    """Il dict delle soglie e' una copia: mutarlo non contamina la chiamata dopo."""
    r = mucus_plug_score([])
    r['thresholds']['low'] = 'manomesso'
    assert mucus_plug_score([])['thresholds']['low'] == '1-3'


def test_accetta_un_iterabile_qualunque():
    from itertools import chain
    r = mucus_plug_score(chain(['RB1'], ['RB2']))
    assert r['mps'] == 2
    assert mucus_plug_score(None)['mps'] == 0     # ingresso assente


def test_phenotype_for_sulle_tre_fasce():
    assert phenotype_for(0) == 'none'
    for n in (1, 2, 3):
        assert phenotype_for(n) == 'low'
    for n in (4, 10, 18):
        assert phenotype_for(n) == 'high'
