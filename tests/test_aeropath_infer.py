"""Test della logica di file del ponte AeroPath (tools/aeropath_infer.py).

Non testa run_rads (libreria esterna): copre layout, pipeline, config e la raccolta
dell'output — cioe' tutto cio' che possiamo validare senza i pesi del modello."""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import aeropath_infer as ai   # noqa: E402


def test_stem_ext_nii_gz():
    assert ai._stem_ext("/a/b/casoDAS.nii.gz") == ("casoDAS", "nii.gz")
    assert ai._stem_ext("x.nii") == ("x", "nii")


def test_build_layout_crea_patient_t0(tmp_path):
    ct = tmp_path / "casoDAS.nii.gz"
    ct.write_bytes(b"CT")
    inp, out, stem = ai.build_layout(str(ct), str(tmp_path / "wk"))
    assert stem == "casoDAS"
    assert os.path.exists(os.path.join(inp, "T0", "casoDAS-t1gd.nii.gz"))
    assert os.path.isdir(out)


def test_write_pipeline_due_passi(tmp_path):
    p = ai.write_pipeline(str(tmp_path))
    d = json.load(open(p))
    assert d["1"]["model"] == "MRI_SequenceClassifier"
    assert d["2"]["model"] == "CT_Airways" and d["2"]["format"] == "thresholding"


def test_write_config_campi_cpu(tmp_path):
    p = ai.write_config(str(tmp_path), "/in", "/out", "/models", "/out/pipe.json")
    import configparser
    c = configparser.ConfigParser()
    c.read(p)
    assert c["System"]["gpu_id"] == "-1"                       # CPU
    assert c["Default"]["task"] == "mediastinum_diagnosis"
    assert c["Runtime"]["reconstruction_method"] == "thresholding"


def test_collect_airways_trova_e_copia(tmp_path):
    out = tmp_path / "output" / "T0"
    out.mkdir(parents=True)
    (out / "casoDAS-t1gd-Airways.nii.gz").write_bytes(b"MASK")
    dest = str(tmp_path / "final" / "aeropath_airways.nii.gz")
    got = ai.collect_airways(str(tmp_path / "output"), dest)
    assert got == dest and os.path.exists(dest)


def test_collect_airways_niente_output(tmp_path):
    (tmp_path / "output").mkdir()
    assert ai.collect_airways(str(tmp_path / "output"), str(tmp_path / "x.nii.gz")) is None


def test_place_extracted_normalizza_cartella(tmp_path):
    # simula uno zip estratto con la radice-modello annidata
    ex = tmp_path / "ex" / "Raidionics-CT_Airways-v13"
    ex.mkdir(parents=True)
    (ex / "pre_processing.ini").write_text("[pre]\n")
    (ex / "model.onnx").write_bytes(b"ONNX")
    models = tmp_path / "models"
    models.mkdir()
    dest = ai._place_extracted(str(tmp_path / "ex"), str(models), "CT_Airways")
    assert dest.endswith(os.path.join("models", "CT_Airways"))
    assert os.path.exists(os.path.join(dest, "pre_processing.ini"))
    assert ai._has_model(str(models), "CT_Airways")


def test_has_model_falso_se_vuoto(tmp_path):
    (tmp_path / "CT_Airways").mkdir()
    assert ai._has_model(str(tmp_path), "CT_Airways") is False


def test_urls_dei_due_modelli_presenti():
    assert set(ai.MODEL_URLS) == {"MRI_SequenceClassifier", "CT_Airways"}
    assert all(u.startswith("https://github.com/raidionics/Raidionics-models/releases/download/")
               for u in ai.MODEL_URLS.values())
