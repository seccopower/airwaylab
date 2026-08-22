"""Snapshot di verifica per ogni ramo misurato.

Per ogni ramo: sezione CT perpendicolare al centerline nel punto centrale di
misura, finestra polmone, contorno del lume segmentato e barra di scala.
Output: out/snaps.json  {branch_id: dataURL PNG}
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from PIL import Image, ImageDraw
import json, io, base64
from lumen import analyze_section, pca_tangent, perp_basis

tree = json.load(open('out/tree_measured.json'))
ISO = tree['iso']
pts = np.array(tree['points'])

ct = sitk.GetArrayFromImage(sitk.ReadImage('out/ct_iso.nii.gz')).astype(np.float32)
mask = sitk.GetArrayFromImage(sitk.ReadImage('out/airway_mask.nii.gz')).astype(np.float32)
if 'bbox' in tree:
    (z0, z1), (y0, y1), (x0, x1) = tree['bbox']
    ct = ct[z0:z1, y0:y1, x0:x1].copy()
    mask = mask[z0:z1, y0:y1, x0:x1].copy()

maskb = mask > 0.5
edt = ndimage.distance_transform_edt(maskb, sampling=(ISO, ISO, ISO))

def tangent(p, i):
    a, b = max(0, i - 3), min(len(p) - 1, i + 3)
    t = p[b] - p[a]
    n = np.linalg.norm(t)
    return t / n if n > 0 else np.array([0, 0, 1.0])

import os as _os
_USE_F = _os.environ.get('AIRWAYLAB_REFINED') == '1'
_SH = np.array(edt.shape)

STEP = 0.3   # mm per pixel del piano
snaps = {}
for b in tree['branches']:
    if _USE_F and b.get('fpath'):
        path = np.array(b['fpath'], dtype=float)
        ridx = np.clip(np.round(path).astype(int), 0, _SH - 1)
        radii = np.array([edt[tuple(q)] for q in ridx])
    else:
        path = pts[b['path']].astype(float)
        radii = np.array([edt[tuple(pts[p])] for p in b['path']])
    n = len(path)
    margin = max(1, int(np.median(radii) / ISO))
    core = list(range(n))[margin:n - margin] or list(range(n))
    r_med = float(np.median(radii[core]))
    cands = sorted(core, key=lambda k: (abs(radii[k] - r_med), abs(k - core[len(core) // 2])))
    i = cands[0]
    best_sec = None
    for k in cands[:12]:
        sec_k = analyze_section(ct, path[k], pca_tangent(path, k), radii[k], ISO)
        if sec_k is not None and sec_k['quality']['ax_ratio'] <= 1.45:
            i = k
            best_sec = sec_k
            break
    center, t = path[i], pca_tangent(path, i)
    u, v = perp_basis(t)
    r = radii[i]
    half = float(np.clip(r * 3 + 8, 14, 42))       # semi-lato in mm
    npx = int(half / STEP)
    ax = np.arange(-npx, npx + 1) * (STEP / ISO)
    U, V = np.meshgrid(ax, ax)
    coords = center[None, :] + U.reshape(-1, 1) * u[None, :] + V.reshape(-1, 1) * v[None, :]
    plane_ct = ndimage.map_coordinates(ct, coords.T, order=1, mode='constant',
                                       cval=-1024).reshape(U.shape)
    plane_m = ndimage.map_coordinates(mask, coords.T, order=1, mode='constant',
                                      cval=0).reshape(U.shape) > 0.5
    # finestra polmone WL -600 / WW 1500
    g = np.clip((plane_ct + 1350) / 1500, 0, 1)
    rgb = (np.stack([g] * 3, -1) * 255).astype(np.uint8)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img)
    # bordo del lume (blu) + bordo esterno parete nei settori validi (arancio)
    sec = best_sec if best_sec is not None else analyze_section(ct, center, t, r, ISO)
    if sec is not None:
        inner_r, outer_r = sec['inner'], sec['outer']
        sh_u, sh_v = sec['shift']
        A = len(inner_r)
        angs = np.linspace(0, 2 * np.pi, A, endpoint=False)
        px = npx + (sh_u + inner_r * np.cos(angs)) / STEP
        py = npx + (sh_v + inner_r * np.sin(angs)) / STEP
        d.polygon(list(zip(px.tolist(), py.tolist())), outline=(42, 120, 214), width=2)
        ox = npx + (sh_u + outer_r * np.cos(angs)) / STEP
        oy = npx + (sh_v + outer_r * np.sin(angs)) / STEP
        okv = np.isfinite(outer_r)
        for a2 in range(A):
            b2 = (a2 + 1) % A
            if okv[a2] and okv[b2]:
                d.line([(ox[a2], oy[a2]), (ox[b2], oy[b2])], fill=(235, 104, 52), width=2)
    else:
        er = ndimage.binary_erosion(plane_m)
        contour = plane_m & ~er
        arr = np.array(img)
        arr[contour] = [42, 120, 214]
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img)
    # barra di scala da 10 mm in basso a destra
    L = int(10 / STEP)
    W, H = img.size
    d.line([(W - L - 8, H - 10), (W - 8, H - 10)], fill=(255, 255, 255), width=2)
    # badge NON REPORTABILE: lo snapshot resta (audit — ogni ramo ha la sua
    # immagine, anche quello escluso) ma dichiara che il numero non vale
    if b.get('qc') in ('no-lume', 'sotto-risoluzione', 'fuga-contorno'):
        d.rectangle([(0, 0), (W - 1, 14)], fill=(120, 20, 20))
        d.text((4, 2), f"NON REPORTABILE · {b['qc']}", fill=(255, 220, 220))
    # ridimensiona a 132 px per la tabella
    img = img.resize((132, 132), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    snaps[b['id']] = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

json.dump(snaps, open('out/snaps.json', 'w'))
tot = sum(len(s) for s in snaps.values()) / 1e6
print(f'{len(snaps)} snapshot, {tot:.1f} MB inline')
