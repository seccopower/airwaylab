"""PROTOTIPO ESPLORATIVO — modello 1D di flusso sull'albero misurato.

*** SIMULAZIONE, NON MISURA CLINICA. *** Nessun numero prodotto qui e' una
misura del paziente: sono uscite di un modello a parametri, da CONGELARE e poi
validare contro pletismografia/spirometria/MBW prima di qualunque uso. Ogni
record esportato porta status="exploratory_simulation" e le chiavi numeriche
hanno prefisso model_.

Rete di resistenze risolta CORRETTAMENTE come albero serie/parallelo (risposta
alla 3a review avversariale, blocker #7): l'impedenza equivalente di un ramo e'
  R_eq(b) = R_b + ( sum_figli 1/R_eq(figlio) )^-1
con le foglie chiuse da una resistenza periferica di completamento. La Raw del
modello e' R_eq(trachea) — la vera resistenza passiva della rete a pressione
alveolare uniforme. I flussi si distribuiscono per conduttanza dei sottoalberi;
con la correzione di Pedley (non lineare) si itera fino a convergenza.

Diametri: half-max REPORTABILE dove esiste; altrove imputati per via
ASIMMETRICA usando i TERRITORI reali del paziente (d ∝ territorio^(1/3), Murray),
non un rapporto simmetrico fisso (blocker #4/major #3). Il completamento
periferico prosegue oltre la foglia senza plateau fino al diametro acinare.

Tutti i parametri fisici sono raccolti in PARAMS e sottoposti a una matrice di
sensibilita' (L/d, diametro di stop, esponente di Murray, Pedley on/off).

Run nel work dir dopo measure/witness/territory (+ plugs se presente).
"""
import json
import os

import numpy as np

import sys

from flow_model import (MU, PA_PER_CMH2O, PARAMS, RHO,  # noqa: F401
                        build_topology, run_model)

sys.setrecursionlimit(20000)

tree = json.load(open('out/tree_measured.json'))
terr = json.load(open('out/territories.json')) if os.path.exists('out/territories.json') else {}
plugs = json.load(open('out/plugs.json')) if os.path.exists('out/plugs.json') else []

# assemblaggio e soluzione vivono in flow_model (condivisi con uncertainty.py)
topo = build_topology(tree, terr)
branches = topo['branches']; by_id = topo['by_id']; kids = topo['kids']
root = topo['root']; parent = topo['parent']; leaves = topo['leaves']
terr_ml = topo['terr_ml']; Sub = topo['Sub']; tot_ml = topo['tot_ml']
lobe_of = topo['lobe_of']


# ---------- soluzione principale ---------------------------------------------
base = PARAMS.copy()
sol = run_model(topo, base)
Raw = sol['Raw']
Q, R, Req, diam = sol['Q'], sol['R'], sol['Req'], sol['diam']
n_meas = sum(1 for b in branches if b.get('d_mean'))
# quota di dissipazione basata su diametri MISURATI vs IMPUTATI (major #3)
Pd_meas = sum(R[b['id']] * Q[b['id']] ** 2 for b in branches if b.get('d_mean'))
Pd_all = sum(R[b['id']] * Q[b['id']] ** 2 for b in branches) \
    + sum(sol['R_ext'][b['id']] * Q[b['id']] ** 2 for b in leaves)
frac_meas = Pd_meas / Pd_all if Pd_all else 0.0

# caduta per generazione e per lobo (con Q coerente della rete)
by_gen, by_lobe = {}, {}
for b in branches:
    w = R[b['id']] * Q[b['id']] ** 2 / Pd_all * 100.0
    by_gen[b['gen']] = by_gen.get(b['gen'], 0.0) + w
    by_lobe[lobe_of(b)] = by_lobe.get(lobe_of(b), 0.0) + w

# ventilazione specifica: flusso terminale / territorio (media pesata = 1)
svent = {}
for b in leaves:
    q_frac = Q[b['id']] / sol['Q_tot']
    v_frac = terr_ml[b['id']] / tot_ml
    svent[b['id']] = q_frac / v_frac
sv = np.array(list(svent.values()))
wv = np.array([terr_ml[i] for i in svent]) / tot_ml
sv_mean = float(np.sum(sv * wv))
sv_cv = float(np.sqrt(np.sum(wv * (sv - sv_mean) ** 2)) / sv_mean)

# pressione cumulativa per ramo (per la visualizzazione)
P_end = {}


def cum_p(b, p_in):
    P_end[b['id']] = p_in + R[b['id']] * Q[b['id']]
    for c in kids(b):
        cum_p(c, P_end[b['id']])


cum_p(root, 0.0)

# ---------- esperimento occlusione plug (review r3, major #7) ----------------
# La rete potata viene RISOLTA DA CAPO con lo stesso punto fisso non lineare del
# baseline (run_model con blocked): le R di Pedley si riassestano sui flussi
# ridistribuiti, non si riusano le R del baseline.
plug_out = []
for p in plugs:
    bid = p.get('branch')
    if bid not in by_id:
        continue
    blocked = set()

    def mark(b):
        blocked.add(b['id'])
        for c in kids(b):
            mark(c)

    mark(by_id[bid])
    # chiusura verso l'alto: un ramo rimasto senza figli attivi e' un vicolo
    # cieco (nessun flusso oltre il plug) ed esce anch'esso dalla rete; cosi'
    # nessuna "nuova foglia" fittizia riceve completamento periferico.
    changed = True
    while changed:
        changed = False
        for b in branches:
            if b['id'] in blocked or not kids(b):
                continue
            if all(c['id'] in blocked for c in kids(b)):
                blocked.add(b['id'])
                changed = True
    if root['id'] in blocked:
        continue
    lost_ml = sum(terr_ml.get(i, 0.0) for i in blocked if i in terr_ml)
    s2 = run_model(topo, base, blocked=frozenset(blocked))
    Raw2 = s2['Raw']
    plug_out.append({
        'status': 'exploratory_simulation', 'plug': p.get('pid'),
        'branch': bid, 'zona': p.get('zona'),
        'ml_esclusi': round(lost_ml, 1),
        'model_raw_occlusa_cmH2O_s_L': round(Raw2, 3),
        'delta_raw_pct': round(100 * (Raw2 - Raw) / Raw, 1),
        'converged': s2['converged'], 'iterazioni': s2['iters'],
        'residuo_rel': float(f"{s2['resid']:.2e}")})

# ---------- sensibilita' one-at-a-time (major #5): Raw E CV -------------------
def cv_of(s):
    q = {b['id']: s['Q'][b['id']] / s['Q_tot'] for b in leaves}
    vv = {b['id']: terr_ml[b['id']] / tot_ml for b in leaves}
    a = np.array([q[b['id']] / vv[b['id']] for b in leaves])
    w = np.array([vv[b['id']] for b in leaves])
    m = float(np.sum(a * w))
    return round(float(np.sqrt(np.sum(w * (a - m) ** 2)) / m), 3)


sens = []
for key, values in (('completion_Ld', [2.0, 3.0, 4.0]),
                    ('completion_dstop_mm', [0.3, 0.5, 0.8]),
                    ('murray_exp', [2.6, 3.0, 3.3]),
                    ('pedley', [True, False])):
    for v in values:
        pp = base.copy()
        pp[key] = v
        s = run_model(topo, pp)
        sens.append({'parametro': key, 'valore': v,
                     'model_raw': round(s['Raw'], 3), 'model_cv_svent': cv_of(s),
                     'converged': s['converged']})

# ---------- dati per ramo (simulati) -----------------------------------------
per_ramo = {}
for b in branches:
    d_m = diam[b['id']] * 1e-3
    v = Q[b['id']] / (np.pi * (d_m / 2) ** 2)
    dg = sol['diag'].get(b['id'], {})
    per_ramo[b['id']] = {
        'status': 'exploratory_simulation',
        'model_d_mm': round(diam[b['id']], 2),
        'd_da_misura': bool(b.get('d_mean')),
        'model_q_ml_s': round(Q[b['id']] * 1e6, 2),
        'model_v_ms': round(v, 3),
        'model_dp_pa': round(R[b['id']] * Q[b['id']], 3),
        'model_p_cum_pa': round(P_end[b['id']], 3),
        'Re': round(dg.get('Re', 0.0), 1),
        'ReD_L': round(dg.get('ReD_L', 0.0), 1),
        'Z_pedley': round(dg.get('Z', 0.0), 3),
        'pedley_attivo': dg.get('pedley_attivo', False),
    }

out = {
    'status': 'exploratory_simulation',
    'DISCLAIMER': 'SIMULAZIONE, non misura clinica. Parametri non validati; '
                  'da congelare e confrontare con pletismografia/spirometria/MBW '
                  'prima di ogni interpretazione. Vedi protocollo di validazione.',
    'condizione': f"snapshot quasi-stazionario a {base['Q_snapshot_Lps']} L/s "
                  "sulla geometria TC a piena inspirazione (NON flusso medio; "
                  "la Raw pletismografica e' a volume/fase diversi e include le vie alte)",
    'model_raw_albero_cmH2O_s_L': round(Raw, 3),
    'quota_dissipazione_da_diametri_misurati_pct': round(100 * frac_meas, 1),
    'diametri_misurati': n_meas, 'diametri_imputati': len(branches) - n_meas,
    'imputazione': 'asimmetrica per territorio (d ∝ territorio^(1/murray_exp)); '
                   'nelle catene mono-figlio frac=1 -> nessun taper finche non '
                   'compare una biforcazione (dipende da topologia e pruning)',
    'convergenza': {'converged': sol['converged'], 'iterazioni': sol['iters'],
                    'residuo_rel': float(f"{sol['resid']:.2e}"),
                    'errore_conservazione_massa_rel': float(f"{sol['mass_err']:.2e}")},
    'sensibilita_tipo': 'one-at-a-time (non fattoriale completa)',
    'model_quota_caduta_pct_per_generazione': {int(g): round(v, 1) for g, v in sorted(by_gen.items())},
    'model_quota_caduta_pct_per_lobo': {k: round(v, 1) for k, v in sorted(by_lobe.items(), key=lambda x: -x[1])},
    'model_ventilazione_specifica': {'media_pesata': round(sv_mean, 3), 'cv': round(sv_cv, 3),
        'nota': 'CV ancora influenzato da profondita' + chr(39) + ' di segmentazione '
                'residua e imputazione: interpretare come confronto tra casi/tempi, non in assoluto'},
    'sensibilita_one_at_a_time': sens,
    'occlusione_plug': plug_out,
    'per_ramo': per_ramo,
    'model_svent_terminali': {i: round(v, 3) for i, v in svent.items()},
    'PARAMS': base,
}
json.dump(out, open('out/flow.json', 'w'), indent=1)
if not sol['converged']:
    print(f"ATTENZIONE: non convergenza (residuo {sol['resid']:.1e} in {sol['iters']} iter)")
print(f"model_raw = {Raw:.3f} cmH2O*s/L | conv={sol['converged']} ({sol['iters']} it, "
      f"resid {sol['resid']:.1e}, massa {sol['mass_err']:.1e}) | "
      f"diametri {n_meas} mis / {len(branches)-n_meas} imp | vent.spec CV {sv_cv:.2f}")
print('R_completion(0.6mm) =', round(sol['R_completion'](0.6) * 1e-3 / PA_PER_CMH2O, 4),
      'cmH2O*s/L (> 0 atteso)')
print('sensibilita:', {f"{s['parametro']}={s['valore']}": (s['model_raw'], s['model_cv_svent']) for s in sens})
for o in plug_out:
    print(' plug', o['plug'], o['zona'], '-', o['ml_esclusi'], 'ml, dRaw',
          o['delta_raw_pct'], '%', '(conv)' if o['converged'] else '(NON CONV)')
