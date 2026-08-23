"""Test del controllo di plausibilita' delle proporzioni lobari (plausibility_core).

'Completo' non vuol dire 'corretto': questi test coprono le partizioni implausibili
che la guardia di presenza (label_qc) NON puo' intercettare. Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from plausibility_core import (   # noqa: E402
    lobe_plausibility,
    plausibility_banner_html,
)


# proporzioni lobari fisiologiche approssimate (frazioni del volume polmonare)
NORMAL = {'RUL': 0.20, 'RML': 0.08, 'RLL': 0.25, 'LUL': 0.22, 'LING': 0.07, 'LLL': 0.18}


def test_normale_ok():
    res = lobe_plausibility(NORMAL)
    assert res['ok'] and res['flags'] == []
    assert plausibility_banner_html(NORMAL) == ''


def test_unita_arbitraria_conta_solo_il_rapporto():
    # stessi rapporti in conteggi voxel: nessun flag
    vox = {lb: int(v * 100000) for lb, v in NORMAL.items()}
    assert lobe_plausibility(vox)['ok']


def test_casoDAS_lobo_minuscolo_e_medio_dominante():
    # scenario reale pre-fix: RUL 0.2%, RML 22.5% (il medio quasi il piu' grande a dx)
    vol = {'RUL': 0.2, 'RML': 22.5, 'RLL': 30.0, 'LUL': 20.0, 'LING': 7.0, 'LLL': 20.3}
    res = lobe_plausibility(vol)
    assert not res['ok']
    codes = {f['code'] for f in res['flags']}
    assert 'tiny_lobe' in codes          # RUL trascurabile
    assert 'rml_not_smallest' in codes   # il medio non e' il piu' piccolo
    banner = plausibility_banner_html(vol)
    assert 'IMPLAUSIBILI' in banner and 'RUL' in banner


def test_medio_il_piu_grande_e_severita_alta():
    vol = {'RUL': 10.0, 'RML': 30.0, 'RLL': 12.0, 'LUL': 22.0, 'LING': 7.0, 'LLL': 19.0}
    res = lobe_plausibility(vol)
    rml = [f for f in res['flags'] if f['code'] == 'rml_not_smallest'][0]
    assert rml['severity'] == 'alto'   # RML e' il massimo a destra


def test_medio_piu_grande_del_sup_ma_non_massimo_e_severita_media():
    # RML > RUL ma RLL e' il maggiore -> RML non e' il minimo, ma nemmeno il massimo
    vol = {'RUL': 10.0, 'RML': 15.0, 'RLL': 25.0, 'LUL': 22.0, 'LING': 7.0, 'LLL': 21.0}
    res = lobe_plausibility(vol)
    rml = [f for f in res['flags'] if f['code'] == 'rml_not_smallest'][0]
    assert rml['severity'] == 'medio'


def test_emitorace_quasi_assente():
    # tutto il polmone destro schiacciato al 10%
    vol = {'RUL': 4.0, 'RML': 2.0, 'RLL': 4.0, 'LUL': 40.0, 'LING': 15.0, 'LLL': 35.0}
    res = lobe_plausibility(vol)
    codes = {f['code'] for f in res['flags']}
    assert 'side_imbalance' in codes


def test_lingula_piu_grande_del_lobo_superiore():
    vol = {'RUL': 20.0, 'RML': 8.0, 'RLL': 25.0, 'LUL': 8.0, 'LING': 20.0, 'LLL': 19.0}
    res = lobe_plausibility(vol)
    codes = {f['code'] for f in res['flags']}
    assert 'ling_gt_lul' in codes


def test_lobi_assenti_ignorati_nelle_quote():
    # solo lobi destri presenti: le quote si normalizzano tra i presenti,
    # la completezza la gestisce label_qc, non questo controllo
    vol = {'RUL': 20.0, 'RML': 8.0, 'RLL': 25.0}
    res = lobe_plausibility(vol)
    # RML minimo a destra, nessun lato sinistro da valutare -> ok
    assert res['ok']


def test_nessun_volume():
    assert not lobe_plausibility({})['ok']
    assert lobe_plausibility({})['flags'][0]['code'] == 'no_volume'
    assert not lobe_plausibility({'RUL': 0, 'RML': None})['ok']
