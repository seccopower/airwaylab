"""Skeletonize the airway mask, build the branch graph, assign generations.

Input:  out/airway_mask.nii.gz, out/seg_info.json
Output: out/tree.json  (nodes, branches with polylines, generation labels)
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize
import networkx as nx
import json

info = json.load(open('out/seg_info.json'))
ISO = info['iso']

from anatomy import QualityError

mask_img = sitk.ReadImage('out/airway_mask.nii.gz')
mask = sitk.GetArrayFromImage(mask_img).astype(bool)
if not mask.any():
    raise QualityError('tree', 'airway mask is empty')

# crop to airway bounding box (+margin) to bound memory on large volumes
nz = np.nonzero(mask)
M = 8
bbox = [[max(0, int(a.min()) - M), min(s, int(a.max()) + M + 1)]
        for a, s in zip(nz, mask.shape)]
mask = mask[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]]
seed_full = json.load(open('out/seg_info.json'))['seed_zyx']
info['seed_zyx'] = [seed_full[i] - bbox[i][0] for i in range(3)]
print('cropped to', mask.shape)

# Euclidean distance transform (mm) — lumen radius at every voxel
edt = ndimage.distance_transform_edt(mask, sampling=(ISO, ISO, ISO))

skel = skeletonize(mask)
print('skeleton voxels:', int(skel.sum()))

# ---------- graph over skeleton voxels ----------
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

# ---------- trace branches between junction/end voxels ----------
def build_graph():
    node_ids = set(np.nonzero((deg != 2))[0])
    G = nx.MultiGraph()
    visited_edges = set()
    for s in node_ids:
        for first in nbrs[s]:
            key = (min(s, first), max(s, first))
            path = [s, first]
            prev, cur = s, first
            while cur not in node_ids:
                nxts = [k for k in nbrs[cur] if k != prev]
                if len(nxts) != 1:
                    break
                prev, cur = cur, nxts[0]
                path.append(cur)
            e = (min(s, cur), max(s, cur), tuple(sorted(path[1:-1])[:3]))
            if e in visited_edges:
                continue
            visited_edges.add(e)
            length = sum(np.linalg.norm((pts[path[k + 1]] - pts[path[k]]) * ISO)
                         for k in range(len(path) - 1))
            G.add_edge(s, cur, path=path, length=length)
    return G

G = build_graph()
print('raw graph: %d nodes, %d edges' % (G.number_of_nodes(), G.number_of_edges()))

# ---------- root = skeleton voxel nearest the trachea seed (cranial end) ----------
seed = np.array(info['seed_zyx'])
d2seed = np.linalg.norm(pts - seed, axis=1)
root = int(np.argmin(d2seed))
if root not in G:
    # nearest graph node
    gn = np.array(list(G.nodes))
    root = int(gn[np.argmin(np.linalg.norm(pts[gn] - seed, axis=1))])

# ---------- prune spurious spurs (short leaf edges from skeleton noise) ----------
def prune(G):
    changed = True
    while changed:
        changed = False
        for n in list(G.nodes):
            if n == root or n not in G:
                continue
            if G.degree(n) == 1:
                (u, v, k) = list(G.edges(n, keys=True))[0]
                d = G.edges[u, v, k]
                other = v if u == n else u
                # spur if shorter than max(3 mm, 1.5x local radius at its junction)
                r_junc = edt[tuple(pts[other])]
                if d['length'] < max(3.0, 2.5 * r_junc):
                    G.remove_node(n)
                    changed = True
        # collapse degree-2 nodes created by pruning
        for n in list(G.nodes):
            if n == root or n not in G:
                continue
            if G.degree(n) == 2:
                edges = list(G.edges(n, keys=True))
                if len(edges) != 2:
                    continue
                (u1, v1, k1), (u2, v2, k2) = edges
                d1, d2 = G.edges[u1, v1, k1], G.edges[u2, v2, k2]
                a = v1 if u1 == n else u1
                b = v2 if u2 == n else u2
                if a == b:
                    continue
                p1 = d1['path'] if d1['path'][-1] == n else d1['path'][::-1]
                p2 = d2['path'] if d2['path'][0] == n else d2['path'][::-1]
                G.remove_node(n)
                G.add_edge(a, b, path=p1 + p2[1:], length=d1['length'] + d2['length'])
                changed = True
    return G

G = prune(G)
print('pruned graph: %d nodes, %d edges' % (G.number_of_nodes(), G.number_of_edges()))

# keep component containing root
comp = nx.node_connected_component(G, root)
G = G.subgraph(comp).copy()

# ---------- orient tree from root, assign generations ----------
# generation 0 = trachea (root -> carina); +1 at every bifurcation
T = nx.bfs_tree(G, root)  # directed
branches = []
gen_of_node = {root: 0}
# map each directed edge to its data in G
for u, v in nx.bfs_edges(G, root):
    # data
    dd = min(G.get_edge_data(u, v).values(), key=lambda d: -d['length'])
    g = gen_of_node[u]
    path = dd['path'] if dd['path'][0] == u else dd['path'][::-1]
    branches.append({'u': int(u), 'v': int(v), 'gen': g,
                     'path': [int(p) for p in path], 'length': dd['length']})
    # children of v are one generation deeper only if v is a real bifurcation
    nchild = len([w for w in G.neighbors(v) if w != u])
    gen_of_node[v] = g + 1 if nchild >= 2 else g
    # if v continues without branching, same generation continues

# merge chains: branch ending in a node with exactly one child = same airway
def merge_chains(branches):
    changed = True
    while changed:
        changed = False
        children = {}
        for b in branches:
            children.setdefault(b['u'], []).append(b)
        for b in branches:
            ch = children.get(b['v'], [])
            if len(ch) == 1 and ch[0] is not b:
                c = ch[0]
                b['path'] = b['path'] + c['path'][1:]
                b['length'] += c['length']
                b['v'] = c['v']
                branches.remove(c)
                changed = True
                break
    return branches

branches = merge_chains(branches)

# name the first branches anatomically
branches.sort(key=lambda b: b['gen'])
for b in branches:
    b['id'] = f"br{branches.index(b):03d}"

# naming is owned by labels.py (which also assigns stable anatomical ids)
for b in branches:
    b['name'] = ''

json.dump({
    'iso': ISO,
    'bbox': bbox,
    'root': int(root),
    'points': pts.tolist(),
    'branches': branches,
    'gen_max': max(b['gen'] for b in branches),
}, open('out/tree.json', 'w'))

from collections import Counter
c = Counter(b['gen'] for b in branches)
print('branches per generation:', dict(sorted(c.items())))
print('max generation:', max(c))
print('total branches:', len(branches))
