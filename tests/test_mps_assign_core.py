"""Test dell'assegnazione tappi -> segmenti (mps_assign_core).

L'assegnazione decide il punteggio piu' del calcolo. I casi che contano sono
quelli in cui e' TENTANTE sbagliare: un tappo distale sotto un segmento
etichettato (va risalito), e un tappo che trova solo il lobo (NON va attribuito,
perche' darebbe uno score piu' alto inventando il segmento). Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from mps_assign_core import (   # noqa: E402
    REASON_NO_BRANCH, REASON_NO_SEGMENT, REASON_OK,
    aid_to_segment, assign_plugs_to_segments, assignable_coverage,
    build_parent_map, labeled_segments, resolve_segment,
)
from mps_core import mucus_plug_score   # noqa: E402


# albero minimo:  trachea(1) -> RMB(2) -> RUL(3) -> B1_R(4) -> sotto-seg(5)
#                                      -> RLL(6) -> (nessun segmento etichettato)
TREE = [
    {'id': 1, 'u': 0, 'v': 1, 'aid': 'TRACHEA'},
    {'id': 2, 'u': 1, 'v': 2, 'aid': 'RMB'},
    {'id': 3, 'u': 2, 'v': 3, 'aid': 'RUL'},
    {'id': 4, 'u': 3, 'v': 4, 'aid': 'B1_R'},
    {'id': 5, 'u': 4, 'v': 5},                 # sotto-segmentale, non etichettato
    {'id': 6, 'u': 2, 'v': 6, 'aid': 'RLL'},
    {'id': 7, 'u': 6, 'v': 7},                 # sotto RLL, nessun segmento
]


def test_mappa_aid_destra_e_sinistra():
    assert aid_to_segment('B6_R') == 'RB6'
    assert aid_to_segment('B10_L') == 'LB10'
    # a sinistra i fusi confluiscono tutti nello stesso codice
    for aid in ('B1_L', 'B2_L', 'B1+2_L'):
        assert aid_to_segment(aid) == 'LB1+2'
    for aid in ('B7_L', 'B8_L', 'B7+8_L'):
        assert aid_to_segment(aid) == 'LB7+8'


def test_aid_non_segmentali_non_mappano():
    for aid in ('TRACHEA', 'RMB', 'RUL', 'LING', 'TB_R', None, '', 'BOH'):
        assert aid_to_segment(aid) is None


def test_parent_map_dalla_topologia():
    p = build_parent_map(TREE)
    assert p[4] == 3 and p[5] == 4 and p[3] == 2 and p[6] == 2


def test_tappo_sul_ramo_segmentale_stesso():
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 4}], TREE)
    a = r['assignments'][0]
    assert a['segment'] == 'RB1'
    assert a['levels_up'] == 0 and a['via_branch'] == 4
    assert a['reason'] == REASON_OK
    assert r['n_assigned'] == 1 and r['n_unassigned'] == 0


def test_tappo_distale_risale_al_segmento():
    """Il ramo occluso e' piu' distale del bronco segmentale: va risalito."""
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 5}], TREE)
    a = r['assignments'][0]
    assert a['segment'] == 'RB1'
    assert a['via_branch'] == 4 and a['levels_up'] == 1


def test_solo_lobo_niente_assegnazione():
    """Il caso in cui e' tentante sbagliare: risalendo si trova RLL, non un
    segmento. Attribuirlo al lobo gonfierebbe lo score inventando il segmento."""
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 7}], TREE)
    a = r['assignments'][0]
    assert a['segment'] is None
    assert a['reason'] == REASON_NO_SEGMENT
    assert r['n_unassigned'] == 1
    assert r['unassigned_by_reason'] == {REASON_NO_SEGMENT: 1}


def test_ramo_non_presente_nell_albero():
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 999}], TREE)
    assert r['assignments'][0]['reason'] == REASON_NO_BRANCH
    assert r['segments'] == [None]


def test_denominatore_vero_dei_segmenti_etichettati():
    """Uno score su 14 segmenti etichettati non e' uno score su 18."""
    assert labeled_segments(TREE) == ['RB1']
    r = assign_plugs_to_segments([], TREE)
    assert r['n_segments_labeled'] == 1
    assert r['segments_labeled'] == ['RB1']


def test_segments_labeled_in_ordine_canonico():
    tree = [{'id': 1, 'u': 0, 'v': 1, 'aid': 'B10_L'},
            {'id': 2, 'u': 0, 'v': 2, 'aid': 'B1_R'},
            {'id': 3, 'u': 0, 'v': 3, 'aid': 'B6_R'}]
    assert labeled_segments(tree) == ['RB1', 'RB6', 'LB10']


def test_segments_e_pronta_per_lo_score():
    """L'uscita si innesta direttamente in mucus_plug_score."""
    plugs = [{'pid': 'p1', 'branch': 4}, {'pid': 'p2', 'branch': 5},
             {'pid': 'p3', 'branch': 7}]
    r = assign_plugs_to_segments(plugs, TREE)
    assert r['segments'] == ['RB1', 'RB1', None]
    s = mucus_plug_score(r['segments'])
    assert s['mps'] == 1               # due tappi, stesso segmento
    assert s['n_plugs'] == 3
    assert s['n_unassigned'] == 1      # il tappo del lobo resta visibile


def test_ingressi_vuoti():
    r = assign_plugs_to_segments([], [])
    assert r['n_plugs'] == 0 and r['segments'] == []
    assert r['n_segments_labeled'] == 0
    r2 = assign_plugs_to_segments(None, None)
    assert r2['n_plugs'] == 0


def test_ciclo_nei_genitori_non_blocca():
    """Albero malformato: la risalita deve fermarsi, non girare all'infinito."""
    ciclo = [{'id': 1, 'u': 2, 'v': 1}, {'id': 2, 'u': 1, 'v': 2}]
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 1}], ciclo)
    assert r['assignments'][0]['segment'] is None
    assert r['assignments'][0]['reason'] == REASON_NO_SEGMENT


def test_provenienza_dichiarata():
    r = assign_plugs_to_segments([], TREE)
    assert r['status'] == 'exploratory'
    assert r['method_id'] == 'mps_assign_airwaylab'


def test_copertura_di_assegnabilita():
    """Quanta parte dell'albero potrebbe contribuire allo score. Nel TREE solo
    B1_R e il suo sotto-segmentale sono sotto un segmento etichettato."""
    cov = assignable_coverage(TREE)
    assert cov['n_branches'] == 7
    assert cov['n_branches_assignable'] == 2        # br4 (B1_R) e br5 sotto di lui
    assert abs(cov['assignable_frac'] - 2 / 7) < 1e-9


def test_copertura_albero_vuoto():
    cov = assignable_coverage([])
    assert cov == {'n_branches': 0, 'n_branches_assignable': 0,
                   'assignable_frac': None}


def test_copertura_riportata_nell_assegnazione():
    r = assign_plugs_to_segments([{'pid': 'p1', 'branch': 4}], TREE)
    assert r['n_branches'] == 7
    assert r['n_branches_assignable'] == 2
    assert abs(r['assignable_frac'] - 2 / 7) < 1e-9


def test_copertura_indipendente_dai_tappi():
    """E' una proprieta' dell'ALBERO: non deve cambiare col numero di tappi."""
    a = assign_plugs_to_segments([], TREE)
    b = assign_plugs_to_segments([{'pid': 'x', 'branch': 5}] * 4, TREE)
    assert a['assignable_frac'] == b['assignable_frac']


def test_resolve_segment_e_la_sola_risalita():
    """Assegnazione e copertura devono dare lo stesso verdetto sullo stesso ramo."""
    by_id = {b['id']: b for b in TREE}
    parent = build_parent_map(TREE)
    seg, via, lv, reason = resolve_segment(5, by_id, parent)
    assert (seg, via, lv, reason) == ('RB1', 4, 1, REASON_OK)
    a = assign_plugs_to_segments([{'pid': 'p', 'branch': 5}], TREE)['assignments'][0]
    assert (a['segment'], a['via_branch'], a['levels_up'], a['reason']) == \
           (seg, via, lv, reason)
