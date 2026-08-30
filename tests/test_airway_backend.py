"""Test dell'adapter I/O dei backend vie aeree (airway_backend).

Verifica la selezione end-to-end e l'adapter AeroPath come consumatore di maschera,
senza dipendenze pesanti (usa file finti + variabili d'ambiente)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

import airway_backend as ab   # noqa: E402
from airway_backend_core import BackendError   # noqa: E402


def _fake_mask(path, data=b"FAKENIfTI"):
    with open(path, "wb") as f:
        f.write(data)
    return path


def test_aeropath_non_disponibile_senza_maschera(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRWAYLAB_AEROPATH_MASK", raising=False)
    seg = tmp_path / "seg"
    seg.mkdir()
    assert ab.AeroPathOnnxBackend().is_available(str(tmp_path / "ct.nii.gz"), str(seg)) is False


def test_aeropath_adotta_maschera_da_env(tmp_path, monkeypatch):
    seg = tmp_path / "seg"
    seg.mkdir()
    ext = _fake_mask(str(tmp_path / "aero_out.nii.gz"))
    monkeypatch.setenv("AIRWAYLAB_AEROPATH_MASK", ext)
    b = ab.AeroPathOnnxBackend()
    assert b.is_available("ct.nii.gz", str(seg))
    out = b.run("ct.nii.gz", str(seg))
    assert os.path.exists(out) and out.endswith("aeropath_airways.nii.gz")


def test_selezione_preferisce_aeropath_quando_ce_la_maschera(tmp_path, monkeypatch):
    seg = tmp_path / "seg"
    seg.mkdir()
    # TS "disponibile" perche' ha gia' la maschera; AeroPath via env
    _fake_mask(str(seg / "lung_airways.nii.gz"))
    monkeypatch.setenv("AIRWAYLAB_AEROPATH_MASK", _fake_mask(str(tmp_path / "aero.nii.gz")))
    mask = ab.segment_airways(str(tmp_path / "ct.nii.gz"), str(seg))
    info = json.load(open(seg / "backend_info.json"))
    assert info["backend"] == "aeropath_onnx"        # preferenza: AeroPath davanti
    assert mask.endswith("aeropath_airways.nii.gz")
    assert len(info["airways_sha256"]) == 64


def test_selezione_ripiega_su_totalsegmentator(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRWAYLAB_AEROPATH_MASK", raising=False)
    seg = tmp_path / "seg"
    seg.mkdir()
    _fake_mask(str(seg / "lung_airways.nii.gz"))       # TS disponibile, AeroPath no
    mask = ab.segment_airways("ct.nii.gz", str(seg))
    info = json.load(open(seg / "backend_info.json"))
    assert info["backend"] == "totalsegmentator"
    assert mask.endswith("lung_airways.nii.gz")


def test_backend_esplicito_non_disponibile_fallisce(tmp_path, monkeypatch):
    monkeypatch.delenv("AIRWAYLAB_AEROPATH_MASK", raising=False)
    seg = tmp_path / "seg"
    seg.mkdir()
    _fake_mask(str(seg / "lung_airways.nii.gz"))
    with pytest.raises(BackendError):
        ab.segment_airways("ct.nii.gz", str(seg), requested="aeropath_onnx")


def _real_nifti(path, shape, spacing):
    """NIfTI vero e minimo: serve una geometria leggibile, non dei dati."""
    import numpy as np
    import nibabel as nib
    aff = np.diag(list(spacing) + [1.0])
    nib.save(nib.Nifti1Image(np.zeros(shape, dtype='uint8'), aff), path)
    return path


def test_maschera_su_griglia_diversa_viene_rigenerata(tmp_path, monkeypatch):
    """Caso reale: maschere di una conversione precedente (834 fette @ 0.082 mm)
    riproposte su una CT corretta (417 @ 0.700). Vanno viste QUI, non a valle."""
    ct = _real_nifti(str(tmp_path / "ct.nii.gz"), (4, 4, 417), (0.744, 0.744, 0.700))
    mask = _real_nifti(str(tmp_path / "m.nii.gz"), (4, 4, 834), (0.744, 0.744, 0.082))
    reason = ab._grid_mismatch(ct, mask)
    assert reason and 'dimensioni' in reason


def test_maschera_sulla_stessa_griglia_e_riusabile(tmp_path):
    ct = _real_nifti(str(tmp_path / "ct.nii.gz"), (4, 4, 8), (0.744, 0.744, 0.700))
    mask = _real_nifti(str(tmp_path / "m.nii.gz"), (4, 4, 8), (0.744, 0.744, 0.700))
    assert ab._grid_mismatch(ct, mask) is None


def test_geometria_non_verificabile_non_blocca(tmp_path):
    """Intestazioni illeggibili: non si dichiara un disallineamento non osservato."""
    assert ab._grid_mismatch("inesistente.nii.gz", str(_fake_mask(str(tmp_path / "m.nii.gz")))) is None
