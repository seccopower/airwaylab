"""Etichettatura anatomica dell'albero bronchiale (lobari + segmentari B1-B10).

Regole geometrico-topologiche in coordinate LPS (+x sinistra paziente,
+y posteriore, +z craniale). Etichetta solo cio' di cui e' ragionevolmente
sicura; il resto resta senza nome. Da lanciare dopo tree.py (aggiorna
out/tree.json in place, prima di measure.py).
"""
import numpy as np
import json

from labels_core import choose_mains

tree = json.load(open('out/tree.json'))
pts = np.array(tree['points'])
ISO = tree['iso']
branches = tree['branches']

# idempotenza: azzera etichette e aid di corse precedenti, cosi' rilanciare
# labels.py su un albero gia' etichettato non lascia nomi stantii
for _b in branches:
    _b['name'] = ''
    _b.pop('aid', None)

# ---------- helpers ----------
by_id = {b['id']: b for b in branches}
children = {}
for b in branches:
    children.setdefault(b['u'], []).append(b)

def kids(b):
    return children.get(b['v'], [])

def direction(b, frac=1.0):
    """Mean direction (unit, LPS mm) of the first `frac` of the branch."""
    p = pts[b['path']].astype(float) * ISO
    n = max(2, int(len(p) * frac))
    d = p[n - 1] - p[0]          # (z, y, x)
    dz, dy, dx = d
    v = np.array([dx, dy, dz])   # -> (x, y, z) LPS
    nn = np.linalg.norm(v)
    return v / nn if nn > 0 else v

def endpoint(b):
    z, y, x = pts[b['path'][-1]].astype(float) * ISO
    return np.array([x, y, z])

def set_name(b, name):
    if b is not None:
        b['name'] = name

def pick(cands, score_fn):
    """Highest-scoring candidate, or None."""
    if not cands:
        return None
    s = sorted(cands, key=score_fn, reverse=True)
    return s[0]

# ---------- trachea and main bronchi ----------
gen0 = [b for b in branches if b['gen'] == 0]
trachea = max(gen0, key=lambda b: b['length']) if gen0 else None
set_name(trachea, 'trachea')
# --- bronchi principali: individuazione ROBUSTA per struttura, non per gen1 ---
# (lo scheletro puo' creare monconi spuri alla carena o saltare generazioni; i
#  principali sono i due rami grandi, ciascuno puro di un lato, piu' prossimali)
mid_x = endpoint(trachea)[0] if trachea is not None else 0.0
_depth = {}
def _set_depth(b, d):
    _depth[b['id']] = d
    for c in kids(b):
        _set_depth(c, d + 1)
if trachea is not None:
    import sys as _sys
    _sys.setrecursionlimit(20000)
    _set_depth(trachea, 0)
_subinfo = {}
def _calc(b):
    ex = endpoint(b)[0]
    nl = 1 if ex > mid_x else 0          # LPS: +x = sinistra paziente
    nr = 1 - nl
    n = 1
    for c in kids(b):
        cn, cl, cr = _calc(c)
        n += cn; nl += cl; nr += cr
    _subinfo[b['id']] = {'depth': _depth.get(b['id'], 0),
                         'n_sub': n, 'n_left': nl, 'n_right': nr}
    return n, nl, nr
if trachea is not None:
    _calc(trachea)

_rid, _lid = choose_mains(_subinfo)
RMB, LMB = by_id.get(_rid), by_id.get(_lid)
if RMB is None or LMB is None:
    # fallback storico: i due rami di generazione 1 per x
    g1 = sorted([b for b in branches if b['gen'] == 1], key=lambda b: endpoint(b)[0])
    if len(g1) >= 2:
        RMB, LMB = g1[0], g1[-1]
if RMB is None or LMB is None:
    json.dump(tree, open('out/tree.json', 'w'))
    print('WARNING [labels]: no main bifurcation found; anatomical labels not assigned')
    raise SystemExit(0)   # soft-fail: pipeline continues without labels
set_name(RMB, 'bronco principale dx')
set_name(LMB, 'bronco principale sx')

MIN_SEG_LEN = 5.0   # mm: sotto questa lunghezza niente etichetta segmentale

def label_upper_segments(seg_cands, side):
    """B1 apicale (piu' craniale), B2 posteriore, B3 anteriore."""
    got = {}
    c = [b for b in seg_cands if b['length'] >= MIN_SEG_LEN]
    b1 = pick(c, lambda b: direction(b)[2])
    if b1 is not None and direction(b1)[2] > 0.3:
        got['B1'] = b1; c.remove(b1)
    b2 = pick(c, lambda b: direction(b)[1])
    if b2 is not None and direction(b2)[1] > 0.1:
        got['B2'] = b2; c.remove(b2)
    b3 = pick(c, lambda b: -direction(b)[1])
    if b3 is not None and direction(b3)[1] < -0.1:
        got['B3'] = b3; c.remove(b3)
    for k, b in got.items():
        set_name(b, f'{k} {side}')
    return got

def label_basal(cands, side, medial_sign):
    """Piramide basale: B7 mediale, B8 anteriore, B9 laterale, B10 posteriore.
    medial_sign: +1 a destra (mediale = x crescente), -1 a sinistra."""
    c = [b for b in cands if direction(b)[2] < 0.2 and b['length'] >= MIN_SEG_LEN]
    got = {}
    b10 = pick(c, lambda b: direction(b)[1] - 0.5 * direction(b)[2])
    if b10 is not None and direction(b10)[1] > 0.15:
        got['B10'] = b10; c.remove(b10)
    b8 = pick(c, lambda b: -direction(b)[1])
    if b8 is not None and direction(b8)[1] < -0.1:
        got['B8'] = b8; c.remove(b8)
    b9 = pick(c, lambda b: -medial_sign * direction(b)[0])
    if b9 is not None and -medial_sign * direction(b9)[0] > 0.2:
        got['B9'] = b9; c.remove(b9)
    if side == 'dx':
        b7 = pick(c, lambda b: medial_sign * direction(b)[0])
        if b7 is not None and medial_sign * direction(b7)[0] > 0.25:
            got['B7'] = b7; c.remove(b7)
    for k, b in got.items():
        set_name(b, f'{k} {side}')
    return got

def descend_chain(b, depth):
    """Follow the largest child chain `depth` bifurcations down. Returns
    (final main segment, take-off branches, intermediate main segments).
    The intermediate mains are the CONTINUATION of the parent bronchus
    (e.g. the basal trunk below B6) — they are NOT segmental candidates."""
    takeoffs = []
    mains = []
    cur = b
    for _ in range(depth):
        ks = kids(cur)
        if not ks:
            break
        main = max(ks, key=lambda x: x['length'])
        takeoffs.extend([k for k in ks if k is not main])
        mains.append(main)
        cur = main
    if mains:
        mains = mains[:-1]          # the deepest main stays a segmental candidate
    return cur, takeoffs, mains


def label_lower_lobe(lobar, side, medial_sign):
    """Lower lobe: B6 takes off posteriorly, then the bronchus CONTINUES as
    the basal trunk before fanning into B7 (right only)/B8/B9/B10. The trunk
    segments must not be labelled as segmentals (clinical review, caso02:
    the segment after B6 was being called B9)."""
    cur, more, mains = descend_chain(lobar, 3)
    pool = [b for b in more + [cur]
            if b['length'] >= MIN_SEG_LEN and b not in mains]
    b6 = pick(pool, lambda b: direction(b)[1] + 0.4 * direction(b)[2])
    if b6 is not None and direction(b6)[1] > 0.25:
        set_name(b6, f'B6 {side}')
        pool = [b for b in pool if b is not b6]
    # the intermediate mains between the lobar bronchus and the basal fan-out
    # are the basal trunk (common inferior trunk)
    for m in mains:
        if not m.get('name'):
            set_name(m, f'tronco basale {side}')
    label_basal(pool, side, medial_sign=medial_sign)

# cranialita' del SOTTOALBERO (robusta a distorsioni del segmento di partenza)
_meanz_memo = {}
def _sub_z(b):
    zs = [endpoint(b)[2]]
    for c in kids(b):
        zs += _sub_z(c)
    return zs
def subtree_meanz(b):
    if b['id'] not in _meanz_memo:
        zs = _sub_z(b)
        _meanz_memo[b['id']] = sum(zs) / len(zs)
    return _meanz_memo[b['id']]

_meany_memo = {}
def _sub_y(b):
    ys = [endpoint(b)[1]]
    for c in kids(b):
        ys += _sub_y(c)
    return ys
def subtree_meany(b):
    if b['id'] not in _meany_memo:
        ys = _sub_y(b)
        _meany_memo[b['id']] = sum(ys) / len(ys)
    return _meany_memo[b['id']]

# ---------- lato destro ----------
rk = kids(RMB)
if rk:
    # lobare superiore dx = figlio col SOTTOALBERO piu' craniale, se chiaramente
    # piu' craniale della continuazione (soglia direzione del segmento troppo
    # fragile su anatomie distorte: usiamo la posizione del sottoalbero)
    rul = max(rk, key=subtree_meanz)
    _cont = max((b for b in rk if b is not rul), key=lambda b: b['length'], default=None)
    if _cont is not None and subtree_meanz(rul) > subtree_meanz(_cont) + 2.0:
        set_name(rul, 'lobare sup dx')
        label_upper_segments(kids(rul), 'dx')
    else:
        rul = None
    rest = [b for b in rk if b is not rul]
    bi = max(rest, key=lambda b: b['length']) if rest else None
    if bi is not None:
        set_name(bi, 'bronco intermedio')
        bik = kids(bi)
        # lobare medio = figlio col SOTTOALBERO piu' ANTERIORE (+y = posteriore),
        # se chiaramente piu' anteriore della continuazione (tronco/inferiore)
        ml = min(bik, key=subtree_meany) if bik else None
        _mlcont = max((b for b in bik if b is not ml), key=lambda b: b['length'], default=None)
        if ml is not None and _mlcont is not None and subtree_meany(ml) < subtree_meany(_mlcont) - 2.0:
            set_name(ml, 'lobare medio')
            mlk = sorted(kids(ml), key=lambda b: endpoint(b)[0])
            if len(mlk) >= 2:
                set_name(mlk[0], 'B4 dx')   # laterale = x minore a dx
                set_name(mlk[-1], 'B5 dx')
        rll_cands = [b for b in bik if b is not ml]
        rll = max(rll_cands, key=lambda b: b['length']) if rll_cands else None
        if rll is not None:
            set_name(rll, 'lobare inf dx')
            label_lower_lobe(rll, 'dx', medial_sign=+1)

# ---------- lato sinistro ----------
lk = kids(LMB)
if lk:
    lul = pick(lk, lambda b: direction(b)[2] + 0.3 * direction(b)[0])
    if lul is not None and direction(lul)[2] > -0.2:
        set_name(lul, 'lobare sup sx')
        uk = kids(lul)
        # lingula: caudale-anteriore; divisione superiore: craniale
        ling = pick(uk, lambda b: -direction(b)[2] - direction(b)[1])
        if ling is not None and direction(ling)[2] < 0.1:
            set_name(ling, 'lingulare')
            lgk = sorted(kids(ling), key=lambda b: -direction(b)[2])
            if len(lgk) >= 2:
                set_name(lgk[0], 'B4 sx')
                set_name(lgk[-1], 'B5 sx')
            upper = [b for b in uk if b is not ling]
        else:
            upper = uk
        segs = []
        for u in upper:
            segs.extend(kids(u)) if len(upper) == 1 else segs.append(u)
        got = label_upper_segments(segs or upper, 'sx')
        # a sinistra B1 e B2 sono di norma un tronco unico apico-posteriore
        if 'B1' in got and 'B2' not in got:
            set_name(got['B1'], 'B1+2 sx')
    rest = [b for b in lk if b is not lul]
    lll = max(rest, key=lambda b: b['length']) if rest else None
    if lll is not None:
        set_name(lll, 'lobare inf sx')
        label_lower_lobe(lll, 'sx', medial_sign=-1)

# stable anatomical ids alongside display labels (see anatomy.py)
from anatomy import to_aid
for b in branches:
    aid = to_aid(b.get('name'))
    if aid:
        b['aid'] = aid

json.dump(tree, open('out/tree.json', 'w'))
named = [(b['name'], b['id'], b['gen'], round(b['length'], 1)) for b in branches if b['name']]
print(f'{len(named)} rami etichettati:')
for n in named:
    print('  ', n)
