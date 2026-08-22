"""External segmentation backend (e.g. deep learning: TotalSegmentator, nnU-Net).

Takes the CT and an externally produced airway mask, brings both onto the
AirwayLab isotropic LPS grid, keeps the largest connected component, places
the trachea seed at the cranial end of the mask, and writes the same three
files segment.py would write — so the rest of the pipeline runs unchanged.

The mask is NOT trusted blindly: witness.py (air witness + resolution gate)
runs after measure.py and demotes branches with no visible lumen or below
the caliber resolution floor.

Usage: extmask.py <ct.nii.gz> <airway_mask.nii.gz>
"""
import json
import os
import sys

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from anatomy import QualityError

CT_IN = sys.argv[1]
MASK_IN = sys.argv[2]

img = sitk.ReadImage(CT_IN)
img = sitk.DICOMOrient(img, 'LPS')
img = sitk.Cast(img, sitk.sitkFloat32)

_env = float(os.environ.get('AIRWAYLAB_SPACING', 0) or 0)
ISO = round(_env if _env > 0 else max(min(img.GetSpacing()), 0.5), 3)

sp, sz = img.GetSpacing(), img.GetSize()
new_sz = [int(round(sz[i] * sp[i] / ISO)) for i in range(3)]
res = sitk.ResampleImageFilter()
res.SetOutputSpacing((ISO, ISO, ISO))
res.SetSize(new_sz)
res.SetOutputOrigin(img.GetOrigin())
res.SetOutputDirection(img.GetDirection())
res.SetInterpolator(sitk.sitkBSpline)
res.SetDefaultPixelValue(-1024)
est_vox = new_sz[0] * new_sz[1] * new_sz[2]
if est_vox > 4.5e8:
    raise QualityError('extmask', f'resampled volume would be {est_vox/1e6:.0f}M voxels; '
                       'set AIRWAYLAB_SPACING to a coarser value (e.g. 0.8) and retry')
iso = res.Execute(img)

m = sitk.ReadImage(MASK_IN)
m = sitk.DICOMOrient(m, 'LPS')
m = sitk.Resample(m, iso, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
arr = sitk.GetArrayFromImage(m) > 0
if arr.sum() == 0:
    raise QualityError('extmask', 'external airway mask is empty on the CT grid — '
                       'check that mask and CT come from the same exam')

lab, n = ndimage.label(arr)
if n > 1:
    sizes = ndimage.sum(arr, lab, range(1, n + 1))
    arr = lab == (int(np.argmax(sizes)) + 1)

vol_ml = float(arr.sum()) * ISO ** 3 / 1000.0
if not 5.0 < vol_ml < 600.0:
    raise QualityError('extmask', f'external airway mask volume {vol_ml:.0f} ml is '
                       'implausible for an airway tree — wrong label file?')

# trachea seed: centroid of the cranial end of the mask.
# On the LPS grid the z array index increases toward superior.
zs = np.nonzero(arr.any(axis=(1, 2)))[0]
z1 = int(zs.max())
band = arr[max(0, z1 - 4):z1 + 1]
cz, cy, cx = ndimage.center_of_mass(band)
seed = [max(0, z1 - 4) + int(round(cz)), int(round(cy)), int(round(cx))]

out = sitk.GetImageFromArray(arr.astype(np.uint8))
out.CopyInformation(iso)
os.makedirs('out', exist_ok=True)
sitk.WriteImage(out, 'out/airway_mask.nii.gz', True)
sitk.WriteImage(sitk.Cast(iso, sitk.sitkInt16), 'out/ct_iso.nii.gz')
json.dump({'seed_zyx': [int(seed[0]), int(seed[1]), int(seed[2])],
           'threshold': None, 'iso': ISO, 'volume_ml': vol_ml,
           'native_spacing': [round(s, 3) for s in sp],
           'backend': 'external mask: ' + os.path.basename(MASK_IN)},
          open('out/seg_info.json', 'w'), indent=1)
print(f'external mask on iso grid: {arr.shape}, {vol_ml:.1f} ml, seed {seed}')
