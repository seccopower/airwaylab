"""Grafo dell'albero vascolare: scheletro, rami, diametri.

Output: out/vessel_tree.json {bbox, points, branches:[{id, path, length, d}]}
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize
import json

info = json.load(open('out/seg_info.json'))
ISO = info['iso']

mask = sitk.GetArrayFromImage(sitk.ReadImage('out/vessel_mask.nii.gz')).astype(bool)
nz = np.nonzero(mask)
M = 4
bbox = [[max(0, int(a.min()) - M), min(s, int(a.max()) + M + 1)]
        for a, s in zip(nz, mask.shape)]
mask = mask[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
del nz
print('crop', mask.shape)

edt = ndimage.distance_transform_edt(mask, sampling=(ISO, ISO, ISO)).astype(np.float32)
skel = skeletonize(mask)
print('skeleton voxels:', int(skel.sum()))

pts = np.argwhere(skel)
idx = {tuple(p): i for i, p in enumerate(pts)}
offs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)
        if (a, b, c) != (0, 0, 0)]
nbrs = [[] for _ in range(len(pts))]
for i, p in enumerate(pts):
    z, y, x = p
    for a, b, c in offs:
        j = idx.get((z + a, y + b, x + c))
        if j is not None:
            nbrs[i].append(j)
deg = np.array([len(n) for n in nbrs])

node_ids = set(np.nonzero(deg != 2)[0])
visited = set()
branches = []
for s in node_ids:
    for first in nbrs[s]:
        path = [s, first]
        prev, cur = s, first
        while cur not in node_ids:
            nxts = [k for k in nbrs[cur] if k != prev]
            if len(nxts) != 1:
                break
            prev, cur = cur, nxts[0]
            path.append(cur)
        key = (min(s, cur), max(s, cur), tuple(sorted(path[1:-1])[:3]))
        if key in visited:
            continue
        visited.add(key)
        length = sum(np.linalg.norm((pts[path[k + 1]] - pts[path[k]]) * ISO)
                     for k in range(len(path) - 1))
        radii = edt[tuple(pts[path].T)]
        n = len(path)
        m = max(1, n // 4)
        core = radii[m:n - m] if n > 2 * m + 1 else radii
        d = 2 * float(np.median(core))
        # spur cortissimi e strutture sotto-millimetriche: fuori
        if length < max(2.0, 1.5 * float(radii[0])) and (deg[s] == 1 or deg[cur] == 1):
            continue
        if d < 0.8:
            continue
        branches.append({'path': [int(p) for p in path], 'length': round(length, 1),
                         'd': round(d, 2)})

for i, b in enumerate(branches):
    b['id'] = f'v{i:04d}'
print(f'{len(branches)} rami vascolari; d mediano '
      f'{np.median([b["d"] for b in branches]):.1f} mm; lunghezza tot '
      f'{sum(b["length"] for b in branches)/10:.0f} cm')

json.dump({'iso': ISO, 'bbox': bbox, 'points': pts.tolist(), 'branches': branches},
          open('out/vessel_tree.json', 'w'))
print('saved out/vessel_tree.json')
