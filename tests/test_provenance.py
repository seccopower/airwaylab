"""Test della carta d'identita' condivisa (provenance). Nucleo puro.

Garantisce che ogni descrittore esplorativo porti stato/metodo/backend/versione e
non inventi nulla quando seg_info manca."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from provenance import build_provenance  # noqa: E402


def test_blocco_completo_da_seg_info():
    info = {'airwaylab_version': '1.0.0', 'backend': 'totalsegmentator', 'iso': 0.5}
    p = build_provenance(info, 'pi10_airwaylab',
                         params={'target_pi_mm': 10.0},
                         denominators={'n_airways_ok': 42},
                         exclusions={'qc_non_ok': 7})
    assert p['status'] == 'exploratory'
    assert p['method_id'] == 'pi10_airwaylab'
    assert p['airwaylab_version'] == '1.0.0'
    assert p['backend'] == 'totalsegmentator'
    assert p['iso_mm'] == 0.5
    assert p['params']['target_pi_mm'] == 10.0
    assert p['denominators']['n_airways_ok'] == 42
    assert p['exclusions']['qc_non_ok'] == 7


def test_degrada_senza_seg_info():
    # niente seg_info: i campi di provenienza sono None, non inventati
    p = build_provenance({}, 'tree_morphometry')
    assert p['status'] == 'exploratory'
    assert p['method_id'] == 'tree_morphometry'
    assert p['airwaylab_version'] is None
    assert p['backend'] is None
    assert p['iso_mm'] is None
    assert p['params'] == {} and p['denominators'] == {} and p['exclusions'] == {}


def test_none_e_tollerato_come_seg_info():
    p = build_provenance(None, 'x')
    assert p['backend'] is None and p['method_id'] == 'x'


def test_status_override():
    p = build_provenance({}, 'x', status='validated')
    assert p['status'] == 'validated'
