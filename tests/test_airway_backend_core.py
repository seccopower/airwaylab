"""Test della selezione pura del backend vie aeree (airway_backend_core). Puro."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from airway_backend_core import (   # noqa: E402
    DEFAULT_PREFERENCE,
    BackendError,
    choose_backend,
)


def test_auto_solo_totalsegmentator_e_il_default():
    # stato attuale (step 1): unico disponibile -> lo sceglie
    name, reason = choose_backend(["totalsegmentator"])
    assert name == "totalsegmentator"
    assert "unico disponibile" in reason


def test_auto_rispetta_la_preferenza_quando_ce_aeropath():
    # stato futuro (step 2): AeroPath davanti in preferenza -> vince in auto
    name, reason = choose_backend(["totalsegmentator", "aeropath_onnx"])
    assert name == "aeropath_onnx"
    assert "preferenza" in reason


def test_preferenza_ha_aeropath_prima_di_totalsegmentator():
    # il ribaltamento del default e' una riga: l'ordine e' gia' quello giusto
    assert DEFAULT_PREFERENCE.index("aeropath_onnx") < DEFAULT_PREFERENCE.index("totalsegmentator")


def test_richiesto_esplicito_disponibile():
    name, reason = choose_backend(["totalsegmentator", "aeropath_onnx"],
                                  requested="totalsegmentator")
    assert name == "totalsegmentator"
    assert "esplicit" in reason


def test_richiesto_esplicito_non_disponibile_fallisce():
    with pytest.raises(BackendError):
        choose_backend(["totalsegmentator"], requested="aeropath_onnx")


def test_nessun_backend_disponibile_fallisce():
    with pytest.raises(BackendError):
        choose_backend([])


def test_disponibile_fuori_preferenza():
    name, reason = choose_backend(["qualcosa_di_nuovo"])
    assert name == "qualcosa_di_nuovo"
    assert "fuori dalla lista" in reason


def test_dedup_preserva_ordine():
    name, _ = choose_backend(["totalsegmentator", "totalsegmentator"])
    assert name == "totalsegmentator"
