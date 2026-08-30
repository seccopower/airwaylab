"""Test del controllo di plausibilita' della geometria (geometry_qc_core).

Il caso che ha motivato il controllo e' reale: un CD con ogni fetta duplicata
(834 file su 417 posizioni) produceva un volume con spacing z 0.08 mm contro il
millimetro dichiarato, senza un solo errore. Questi test bloccano quel caso e le
sue varianti. Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from geometry_qc_core import (   # noqa: E402
    duplicate_positions,
    geometry_plausibility,
    geometry_banner,
    grids_match,
)


# geometria tipica di una TC torace sottile (il caso 'dina' dopo deduplica)
GOOD_SIZE = (512, 512, 417)
GOOD_SPACING = (0.744, 0.744, 0.700)


def test_geometria_buona_nessun_flag():
    res = geometry_plausibility(GOOD_SIZE, GOOD_SPACING, declared_thickness_mm=1.0)
    assert res['ok'] and res['flags'] == []
    assert round(res['coverage_mm']) == 292
    assert geometry_banner(GOOD_SIZE, GOOD_SPACING, 1.0) == ''


def test_caso_reale_fette_duplicate():
    """834 file su 417 posizioni -> spacing 0.08 mm con 1 mm dichiarato."""
    res = geometry_plausibility((512, 512, 834), (0.744, 0.744, 0.08),
                                declared_thickness_mm=1.0)
    assert not res['ok']
    codes = {f['code'] for f in res['flags']}
    assert 'z_spacing_troppo_piccolo' in codes
    assert 'spessore_incoerente' in codes
    assert all(f['severity'] == 'alto' for f in res['flags'])


def test_duplicate_positions_conta_le_ripetizioni():
    zs = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    d = duplicate_positions(zs)
    assert d['n_total'] == 6 and d['n_unique'] == 3
    assert d['n_duplicate'] == 3 and d['multiplicity'] == 2
    assert d['has_duplicates']


def test_duplicate_positions_serie_pulita():
    d = duplicate_positions([0.0, 0.7, 1.4, 2.1])
    assert not d['has_duplicates']
    assert d['n_duplicate'] == 0 and d['multiplicity'] == 1


def test_duplicate_positions_tolleranza_virgola_mobile():
    """Posizioni uguali a meno del rumore non devono contare come distinte."""
    d = duplicate_positions([1.0, 1.0000001, 2.0])
    assert d['n_unique'] == 2 and d['has_duplicates']


def test_duplicate_positions_tolleranza_non_valida():
    try:
        duplicate_positions([1.0], tol_mm=0)
    except ValueError:
        return
    raise AssertionError('tol_mm=0 deve sollevare ValueError')


def test_spacing_z_troppo_grande():
    res = geometry_plausibility((512, 512, 30), (0.7, 0.7, 12.0))
    assert 'z_spacing_troppo_grande' in {f['code'] for f in res['flags']}


def test_copertura_insufficiente():
    """Poche fette sottili: geometria coerente ma non e' un torace."""
    res = geometry_plausibility((512, 512, 40), (0.7, 0.7, 0.7))
    codes = {f['code'] for f in res['flags']}
    assert codes == {'copertura_insufficiente'}


def test_inplane_implausibile():
    res = geometry_plausibility(GOOD_SIZE, (3.0, 3.0, 0.7))
    codes = {f['code'] for f in res['flags']}
    assert 'inplane_x_implausibile' in codes and 'inplane_y_implausibile' in codes


def test_spessore_coerente_non_flagga():
    """Protocolli insoliti ma coerenti non devono essere segnalati."""
    res = geometry_plausibility((512, 512, 700), (0.6, 0.6, 0.4),
                                declared_thickness_mm=0.5)
    assert res['ok']


def test_spessore_dichiarato_assente_non_flagga():
    res = geometry_plausibility(GOOD_SIZE, GOOD_SPACING, declared_thickness_mm=None)
    assert res['ok']


def test_spacing_zero_non_esplode():
    """sz = 0 non deve sollevare ZeroDivisionError nel messaggio."""
    res = geometry_plausibility((512, 512, 100), (0.7, 0.7, 0.0))
    assert not res['ok']
    assert any(f['code'] == 'z_spacing_troppo_piccolo' for f in res['flags'])


def test_size_o_spacing_malformati():
    for bad in (((512, 512), GOOD_SPACING), (GOOD_SIZE, (0.7, 0.7))):
        try:
            geometry_plausibility(*bad)
        except ValueError:
            continue
        raise AssertionError('size/spacing a 2 componenti devono sollevare ValueError')


def test_banner_riporta_duplicati_e_geometria():
    dup = duplicate_positions([1.0, 1.0, 2.0, 2.0])
    txt = geometry_banner((512, 512, 834), (0.744, 0.744, 0.08), 1.0, dup=dup)
    assert 'posizione z ripetuta' in txt
    assert 'spessore dichiarato' in txt
    assert 'NON e\' affidabile' in txt


def test_banner_vuoto_se_tutto_a_posto():
    dup = duplicate_positions([0.0, 0.7, 1.4])
    assert geometry_banner(GOOD_SIZE, GOOD_SPACING, 1.0, dup=dup) == ''


def test_grids_match_stessa_griglia():
    r = grids_match(GOOD_SIZE, GOOD_SPACING, GOOD_SIZE, GOOD_SPACING)
    assert r['ok'] and r['reason'] is None


def test_grids_match_caso_reale_maschera_obsoleta():
    """Maschere della conversione rotta (834 @ 0.082) contro la CT corretta."""
    r = grids_match(GOOD_SIZE, GOOD_SPACING,
                    (512, 512, 834), (0.744, 0.744, 0.082))
    assert not r['ok']
    assert 'dimensioni' in r['reason']


def test_grids_match_spacing_diverso_stesse_dimensioni():
    r = grids_match(GOOD_SIZE, GOOD_SPACING, GOOD_SIZE, (0.744, 0.744, 1.25))
    assert not r['ok']
    assert 'spacing' in r['reason']


def test_grids_match_tollera_il_rumore_di_virgola_mobile():
    r = grids_match(GOOD_SIZE, GOOD_SPACING, GOOD_SIZE,
                    (0.7440001, 0.7439999, 0.7000001))
    assert r['ok']


def test_banner_duplicati_riparati_non_allarma_sulla_geometria():
    """Caso 'dina' dopo la deduplica: va detto cosa e' stato fatto, ma la
    geometria e' tornata corretta e non va dichiarata inaffidabile."""
    dup = duplicate_positions([1.0, 1.0, 2.0, 2.0])
    txt = geometry_banner(GOOD_SIZE, GOOD_SPACING, 1.0, dup=dup)
    assert 'posizione z ripetuta' in txt
    assert 'NON e\'' not in txt
