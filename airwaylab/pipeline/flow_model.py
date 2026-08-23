"""Nucleo IMPORTABILE del modello di flusso 1D (assemblaggio + soluzione).

Usato SIA da flow.py (produzione) SIA da uncertainty.py (ensemble di incertezza,
protocollo #6): stessa fisica, nessuna costante duplicata. La fisica pura
(Poiseuille, completamento periferico, solve serie/parallelo) sta in flow_core.py;
qui c'e' l'assemblaggio sull'albero reale, parametrizzato per permettere le
perturbazioni congiunte dell'ensemble senza toccare la produzione.

`run_model` con i default riproduce ESATTAMENTE la soluzione di flow.py. Gli
argomenti opzionali servono solo all'ensemble:
  d_meas_scale  : {bid -> fattore} sui diametri MISURATI (rumore per-misura);
  imp_scale     : {bid -> fattore} sui diametri IMPUTATI (perturbazione correlata
                  per sottoalbero, calcolata dal chiamante);
  territory_conditioned : se False, l'imputazione NON usa i territori (split
                  simmetrico di Murray) — ablazione strutturale #5.
"""
import numpy as np

from flow_core import mass_error, poiseuille_R, r_completion, solve_tree
from impute_core import imputed_diameter

MU = 1.81e-5          # Pa*s
RHO = 1.20            # kg/m^3
PA_PER_CMH2O = 98.0665

PARAMS = dict(
    Q_snapshot_Lps=0.5,
    murray_exp=3.0,
    completion_Ld=3.0,
    completion_dstop_mm=0.5,
    pedley=True,
    d_num_floor_mm=0.3,
)

LOBE_AID = {'RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL'}


def build_topology(tree, terr):
    """Costruisce topologia, territori di sottoalbero e helper. Chiamata una volta;
    il risultato (dict) si passa a run_model."""
    branches = tree['branches']
    by_id = {b['id']: b for b in branches}
    children = {}
    for b in branches:
        children.setdefault(b['u'], []).append(b)

    def kids(b):
        return children.get(b['v'], [])

    root = next((b for b in branches if b.get('aid') == 'TRACHEA'), None) or branches[0]
    parent = {}
    for p in branches:
        for c in kids(p):
            parent[c['id']] = p

    leaves = [b for b in branches if not kids(b)]
    terr_ml = {b['id']: max(0.05, float(terr.get(b['id'], 0.05))) for b in leaves}
    Sub = {}

    def subtree_terr(b):
        if b['id'] in Sub:
            return Sub[b['id']]
        ks = kids(b)
        s = terr_ml.get(b['id'], 0.0) if not ks else sum(subtree_terr(c) for c in ks)
        Sub[b['id']] = s
        return s

    subtree_terr(root)

    def lobe_of(b):
        cur = b
        for _ in range(40):
            if cur.get('aid') in LOBE_AID:
                return cur['aid']
            cur = parent.get(cur['id'])
            if cur is None:
                return 'CENTRAL'
        return 'CENTRAL'

    return dict(branches=branches, by_id=by_id, kids=kids, root=root, parent=parent,
                leaves=leaves, terr_ml=terr_ml, Sub=Sub, tot_ml=Sub[root['id']],
                lobe_of=lobe_of)


def run_model(topo, P, blocked=frozenset(), d_meas_scale=None, imp_scale=None,
              territory_conditioned=True):
    """Risolve il modello. Con i default identico a flow.py; gli override servono
    all'ensemble (vedi docstring del modulo)."""
    branches = topo['branches']; by_id = topo['by_id']; kids = topo['kids']
    root = topo['root']; parent = topo['parent']; leaves = topo['leaves']
    terr_ml = topo['terr_ml']; Sub = topo['Sub']; tot_ml = topo['tot_ml']

    dstop = P['completion_dstop_mm']
    Ld = P['completion_Ld']
    nexp = P['murray_exp']
    dfloor = P['d_num_floor_mm']
    Q_tot = P['Q_snapshot_Lps'] * 1e-3
    dms = d_meas_scale or {}
    ims = imp_scale or {}

    diam = {}

    def assign_d(b, d_par):
        if b.get('d_mean'):
            d = float(b['d_mean']) * dms.get(b['id'], 1.0)
        else:
            if territory_conditioned and b['id'] in parent:
                frac = Sub[b['id']] / max(1e-9, Sub[parent[b['id']]['id']])
            elif b['id'] in parent:
                # ablazione airway-only: split simmetrico (Murray), nessun territorio
                sibs = kids(parent[b['id']])
                frac = 1.0 / max(1, len(sibs))
            else:
                frac = 1.0
            d = imputed_diameter(d_par, frac, nexp, dfloor)
            d *= ims.get(b['id'], 1.0)
        diam[b['id']] = d
        for c in kids(b):
            assign_d(c, d)

    assign_d(root, float(root.get('d_mean') or 18.0))

    def R_completion(d_leaf):
        return r_completion(d_leaf, dstop, Ld, nexp, MU)

    keep = [b for b in branches if b['id'] not in blocked]
    def ch(bid):
        return [c['id'] for c in kids(by_id[bid]) if c['id'] not in blocked]
    lv = [b for b in keep if not ch(b['id'])]
    R_ext = {b['id']: R_completion(diam[b['id']]) for b in lv}

    diag = {}

    def branch_R(b, q):
        d_m = diam[b['id']] * 1e-3
        L_m = max(1e-3, float(b['length']) * 1e-3)
        R0 = poiseuille_R(L_m, d_m, MU)
        Re = Z = 0.0
        if P['pedley'] and q > 0:
            v = q / (np.pi * (d_m / 2) ** 2)
            Re = RHO * v * d_m / MU
            Z = 0.327 * np.sqrt(max(1e-9, Re * d_m / L_m))
            R = R0 * max(1.0, Z)
        else:
            R = R0
        diag[b['id']] = {'Re': Re, 'ReD_L': Re * d_m / L_m if Re else 0.0,
                         'Z': Z, 'pedley_attivo': bool(Z > 1.0)}
        return R

    Q = {b['id']: Q_tot * Sub[b['id']] / tot_ml for b in keep}
    Req, converged, iters, resid = {}, False, 0, float('inf')
    for it in range(1, 61):
        R = {b['id']: branch_R(b, Q[b['id']]) for b in keep}
        Req, newQ = solve_tree(root['id'], ch, R, R_ext, Q_tot)
        resid = max(abs(newQ[b['id']] - Q[b['id']]) for b in keep) / Q_tot
        Q, iters = newQ, it
        if resid < 1e-9:
            converged = True
            break

    R = {b['id']: branch_R(b, Q[b['id']]) for b in keep}
    Req, _ = solve_tree(root['id'], ch, R, R_ext, Q_tot)
    Raw = Req[root['id']] * 1e-3 / PA_PER_CMH2O
    mass_err = mass_error(root['id'], ch, Q, Q_tot)

    return dict(diam=diam, R=R, R_ext=R_ext, Req=Req, Q=Q, diag=diag,
                Raw=Raw, Q_tot=Q_tot, converged=converged, iters=iters,
                resid=resid, mass_err=mass_err, R_completion=R_completion)
