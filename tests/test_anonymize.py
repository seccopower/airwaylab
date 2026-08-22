"""Test della selezione automatica della serie (anonymize.py). Nessun DICOM reale:
metadati sintetici. Puro."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from anonymize import disqualify_reason, rank_series   # noqa: E402


def test_caso06_scenario_sceglie_la_sottile():
    # esattamente le 3 serie di caso06: la [2] e' un MIP 10 mm, va scartata
    metas = [
        {'modality': 'CT', 'desc': 'Torace HR Lung 1,00 Bl60 S3', 'thickness': '1.0',
         'image_type': 'ORIGINAL\\PRIMARY\\AXIAL', 'n_images': 556},
        {'modality': 'CT', 'desc': 'Torace HR Mediast 3,00 Br40 S3', 'thickness': '3.0',
         'image_type': 'ORIGINAL\\PRIMARY\\AXIAL', 'n_images': 129},
        {'modality': 'CT', 'desc': 'Torace HR Lung 10,00 Bl60 S3 ax MIP', 'thickness': '10.0',
         'image_type': 'DERIVED\\SECONDARY\\AXIAL\\MIP', 'n_images': 55},
    ]
    ranked = rank_series(metas)
    # la prima in classifica e' la serie sottile [0], qualificata
    assert ranked[0]['orig_index'] == 0 and ranked[0]['disq'] is None
    # il MIP e' squalificato
    mip = next(m for m in ranked if m['orig_index'] == 2)
    assert mip['disq'] is not None and 'derivata' in mip['disq'].lower()


def test_disqualify_reasons():
    assert disqualify_reason({'modality': 'MR', 'desc': 'x', 'n_images': 300}) == 'non CT'
    assert disqualify_reason({'modality': 'CT', 'desc': 'Scout', 'n_images': 300})
    assert disqualify_reason({'modality': 'CT', 'desc': 'lung', 'image_type': 'DERIVED\\MIP',
                              'n_images': 300})
    assert disqualify_reason({'modality': 'CT', 'desc': 'lung 1mm', 'image_type': 'ORIGINAL\\PRIMARY',
                              'n_images': 20}) == 'troppe poche immagini'
    # una buona serie assiale sottile non e' squalificata
    assert disqualify_reason({'modality': 'CT', 'desc': 'Chest 1.0 B31', 'image_type': 'ORIGINAL\\PRIMARY\\AXIAL',
                              'n_images': 400}) is None


def test_thin_preferred_over_thick_when_both_ok():
    metas = [
        {'modality': 'CT', 'desc': 'std 2mm', 'thickness': '2.0', 'image_type': 'ORIGINAL\\PRIMARY', 'n_images': 250},
        {'modality': 'CT', 'desc': 'thin 0.6mm', 'thickness': '0.6', 'image_type': 'ORIGINAL\\PRIMARY', 'n_images': 700},
    ]
    ranked = rank_series(metas)
    assert ranked[0]['desc'] == 'thin 0.6mm'


def test_missing_thickness_ranks_last_among_ok():
    metas = [
        {'modality': 'CT', 'desc': 'no thick', 'thickness': '?', 'image_type': 'ORIGINAL\\PRIMARY', 'n_images': 900},
        {'modality': 'CT', 'desc': 'thin 1mm', 'thickness': '1.0', 'image_type': 'ORIGINAL\\PRIMARY', 'n_images': 500},
    ]
    ranked = rank_series(metas)
    assert ranked[0]['desc'] == 'thin 1mm'   # strato noto e sottile batte spessore ignoto
