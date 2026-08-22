"""Airway segmentation: trachea seed detection + explosion-controlled region growing.

Input:  data/CT-chest.nrrd
Output: out/ct_iso.nii.gz      (CT resampled to isotropic spacing)
        out/airway_mask.nii.gz (binary airway lumen mask, isotropic)
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import os, sys, json

os.makedirs('out', exist_ok=True)

from anatomy import QualityError

INPUT = sys.argv[1] if len(sys.argv) > 1 else 'data/CT-chest.nrrd'

# ---------- load, canonicalize orientation, resample to isotropic ----------
img = sitk.ReadImage(INPUT)
# canonicalize to LPS: all downstream anatomy (left/right, ant/post, cranial)
# assumes +x=Left, +y=Posterior, +z=Superior with an identity-like direction.
img = sitk.DICOMOrient(img, 'LPS')
img = sitk.Cast(img, sitk.sitkFloat32)

# isotropic target: finest native spacing, floored at 0.5 mm, overridable
# via AIRWAYLAB_SPACING (mm) for memory-constrained machines
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
    raise QualityError('segment', f'resampled volume would be {est_vox/1e6:.0f}M voxels; '
                       'set AIRWAYLAB_SPACING to a coarser value (e.g. 0.8) and retry')
iso = res.Execute(img)
vol = sitk.GetArrayFromImage(iso)  # (z, y, x)
print('iso volume:', vol.shape, 'spacing', ISO)

# ---------- body mask (to exclude ambient air) ----------
body = vol > -500
lab, n = ndimage.label(body)
if n == 0:
    raise QualityError('segment', 'no body found (no voxels above -500 HU): not a CT?')
sizes = ndimage.sum(body, lab, range(1, n + 1))
body = lab == (1 + int(np.argmax(sizes)))
# fill internal holes slice by slice (lungs, airways become part of body)
body_filled = np.zeros_like(body)
for z in range(body.shape[0]):
    body_filled[z] = ndimage.binary_fill_holes(body[z])
print('body mask:', body_filled.sum() * ISO**3 / 1e6, 'l')

# ---------- candidate trachea seeds: persistent round air-blob chains ----------
# The kernel can be noisy (e.g. B60/B70): grow on a smoothed copy so parenchyma
# noise does not connect the airway lumen to the lung at low thresholds.
vs = ndimage.gaussian_filter(vol, sigma=1.2)

def find_seed_chains(vol):
    """Track round air blobs across cranial slices; return chains (candidates)."""
    Z, Y, X = vol.shape
    zrange = list(range(Z - 1, int(Z * 0.55), -2))
    chains = []   # each: list of (z, cy, cx, area)
    for z in zrange:
        sl = vol[z] < -900
        lab, n = ndimage.label(sl)
        for i in range(1, n + 1):
            m = lab == i
            area = m.sum() * ISO * ISO
            if not (60 < area < 900):
                continue
            ys, xs = np.nonzero(m)
            cy, cx = ys.mean(), xs.mean()
            if abs(cx - X / 2) > X * 0.20:
                continue
            h, w = int(ys.max() - ys.min()) + 1, int(xs.max() - xs.min()) + 1
            if max(h, w) / max(1, min(h, w)) > 2.2 or m.sum() / (h * w) < 0.5:
                continue
            for ch in chains:
                pz, py, px, _ = ch[-1]
                if abs(pz - z) <= 4 and abs(py - cy) < 12 and abs(px - cx) < 12:
                    ch.append((z, cy, cx, area))
                    break
            else:
                chains.append([(z, cy, cx, area)])
    chains = [c for c in chains if len(c) >= 6]
    chains.sort(key=lambda c: -len(c))
    return chains

chains = find_seed_chains(vs)
print(f'{len(chains)} candidate chains:',
      [(len(c), int(c[len(c)//2][1]), int(c[len(c)//2][2])) for c in chains[:4]])

# ---------- explosion-controlled region growing (on smoothed volume) ----------
def grow(seed, thr):
    mask = (vs < thr) & body_filled
    lab, _ = ndimage.label(mask)
    l = lab[seed]
    if l == 0:
        return None
    return lab == l

def grow_tree(seed):
    thrs = np.arange(-980, -694, 10)
    prev, prev_vox, chosen, chosen_thr = None, None, None, None
    for t in thrs:
        m = grow(seed, t)
        if m is None:
            continue
        v = int(m.sum())
        if v * ISO**3 / 1000 > 500 and prev is None:
            return None, None   # hopeless: already merged with lung

        if prev_vox is not None and prev_vox > 3000 and v > prev_vox * 1.9 \
           and (v - prev_vox) * ISO**3 / 1000 > 15:
            print(f'  explosion at {t}: {prev_vox} -> {v}, keeping {t-10}')
            return prev, t - 10
        prev, prev_vox = m, v
        print(f'  thr {t}: {v * ISO**3 / 1000:.1f} ml')
    return prev, thrs[-1] if prev is not None else None

best = None
for ci, ch in enumerate(chains[:3]):
    mid = ch[min(2, len(ch) - 1)]
    seed = (int(mid[0]), int(mid[1]), int(mid[2]))
    print(f'candidate {ci} seed {seed} (chain of {len(ch)} slices):')
    m, thr = grow_tree(seed)
    if m is None:
        continue
    vol_ml = m.sum() * ISO**3 / 1000
    xs = np.nonzero(m.any(axis=(0, 1)))[0]
    x_extent = (xs.max() - xs.min()) * ISO if len(xs) else 0
    zs = np.nonzero(m.any(axis=(1, 2)))[0]
    z_extent = (zs.max() - zs.min()) * ISO if len(zs) else 0
    # plausible airway tree: tens of ml, spans both lungs laterally, tall
    score = (10 < vol_ml < 300) * (x_extent > 80) * (z_extent > 80) * vol_ml
    print(f'  -> {vol_ml:.1f} ml, x-extent {x_extent:.0f} mm, z-extent {z_extent:.0f} mm, score {score:.1f}')
    if best is None or score > best[0]:
        best = (score, m, thr, seed)

if best is None or best[0] == 0:
    raise QualityError('segment', 'no plausible airway tree found: check that the input '
                       'is an inspiratory chest CT and that the trachea is in the field of view')
_, chosen, chosen_thr, seed = best
seed_z, seed_y, seed_x = seed
print('chosen seed', seed, 'thr', chosen_thr)

# refine on the unsmoothed volume: recover sharp lumen voxels near the mask.
# Conservative: only voxels below the SAME threshold, 1-voxel shell — on very
# dark lungs a looser refine pads the tree with a parenchyma halo and inflates
# every diameter.
REFINE = False   # off: on noisy/dark lungs it pads the tree and corrupts the skeleton
if REFINE:
    sharp = (vol < chosen_thr) & body_filled
    sharp &= ndimage.binary_dilation(chosen, iterations=1)
    chosen = chosen | sharp

# light morphological closing to smooth the lumen
chosen = ndimage.binary_closing(chosen, structure=np.ones((3, 3, 3)))
lab, _ = ndimage.label(chosen)
chosen = lab == lab[seed]

vol_ml = chosen.sum() * ISO**3 / 1000
print(f'final threshold {chosen_thr}, airway volume {vol_ml:.1f} ml')

out = sitk.GetImageFromArray(chosen.astype(np.uint8))
out.CopyInformation(iso)
sitk.WriteImage(out, 'out/airway_mask.nii.gz')
sitk.WriteImage(sitk.Cast(iso, sitk.sitkInt16), 'out/ct_iso.nii.gz')
json.dump({'seed_zyx': [int(seed_z), int(seed_y), int(seed_x)],
           'threshold': float(chosen_thr), 'iso': ISO,
           'native_spacing': [round(s, 3) for s in sp],
           'volume_ml': float(vol_ml)},
          open('out/seg_info.json', 'w'), indent=1)
print('saved out/airway_mask.nii.gz')
