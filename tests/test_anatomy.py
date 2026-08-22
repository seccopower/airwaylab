"""Regression guards for anatomical identity and orientation handling.

These tests target the exact failure classes flagged in external review:
display-label/id drift (would silently disable ALR4) and non-LPS input
orientation (would silently flip left/right).
"""
import os
import sys

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "airwaylab", "pipeline"))

from anatomy import ALR4_AIDS, DISPLAY_TO_AID, to_aid  # noqa: E402


def test_alr4_aids_are_producible():
    """Every aid required by ALR4 must be producible by the label mapping."""
    producible = set(DISPLAY_TO_AID.values())
    assert ALR4_AIDS <= producible, ALR4_AIDS - producible


def test_segmental_aid_mapping():
    assert to_aid("B6 dx") == "B6_R"
    assert to_aid("B10 sx") == "B10_L"
    assert to_aid("B1+2 sx") == "B1+2_L"
    assert to_aid("") is None
    assert to_aid(None) is None


def test_dicom_orient_canonicalizes_ras_to_lps():
    """A RAS-oriented volume must come out of DICOMOrient as identity-LPS,
    with left/right voxel content swapped accordingly."""
    arr = np.zeros((4, 4, 4), dtype=np.float32)
    arr[:, :, 3] = 100.0  # marker at high-x voxels
    img = sitk.GetImageFromArray(arr)
    # RAS direction: x and y axes flipped w.r.t. LPS
    img.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    out = sitk.DICOMOrient(img, "LPS")
    assert np.allclose(out.GetDirection(), (1, 0, 0, 0, 1, 0, 0, 0, 1))
    out_arr = sitk.GetArrayFromImage(out)
    # the marker plane must now sit at LOW x indices (content actually moved)
    assert out_arr[:, :, 0].mean() == 100.0
    assert out_arr[:, :, 3].mean() == 0.0
