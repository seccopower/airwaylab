"""Ensemble di incertezza sul modello di flusso (protocollo #6) — compute + pagina.

Propaga in modo CONGIUNTO le incertezze del modello (diametri misurati e imputati,
Murray, completamento, Pedley, topologia/pruning, territori) fino agli endpoint
LOBARI: quote di flusso q per lobo e indice I = media della frazione LAA pesata dalle
quote di flusso. Riporta mediana e 5-95%, probabilita' di rango peggiore, stabilita'
della classificazione e del rango, quota di varianza dai diametri imputati, e
l'ablazione strutturale #5 (q con vs senza condizionamento dal territorio). La
convergenza numerica e' riportata separatamente dall'incertezza epistemica.

Scenario e perturbazioni: protocollo v1.0 congelato. Riusa flow_model (stessa fisica
di produzione) e morphomap_core (LAA). Deterministico: seed fisso.

Run nel work dir dopo flow.py + territory.py + morphomap. Output: out/uncertainty.json
+ out/uncertainty.html. Si salta se mancano gli input.
"""
import json
import os

import nibabel as nib
import numpy as np

from flow_model import PARAMS, build_topology, run_model
from morphomap_core import classify_lobe, voxelwise_destruction
from uncertainty_core import (
    label_stability,
    quantiles,
    rank_stability,
    variance_share,
    worst_rank_prob,
)

SEED = 12345
N_MAIN = 300
N_ABL = 100          # sub-ensemble per la quota di varianza (imputati vs misurati)
LOBES = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')
STAB_THRESH = 0.8
LAA_HU = -950.0

need = ['out/tree_measured.json', 'out/territory_labels_ds.nii.gz',
        'out/territory_index.json', 'out/ct_iso.nii.gz']
if not all(os.path.exists(p) for p in need):
    print('uncertainty: input mancanti — salto')
    raise SystemExit(0)

tree = json.load(open('out/tree_measured.json'))
terr = json.load(open('out/territories.json')) if os.path.exists('out/territories.json') else {}

# --- LAA per lobo (fisso, dalla TC): non varia tra le repliche ---------------
lab = np.asarray(nib.load('out/territory_labels_ds.nii.gz').dataobj).astype(np.int32)
ti = {int(k): v for k, v in json.load(open('out/territory_index.json')).items()}
ct = np.asarray(nib.load('out/ct_iso.nii.gz').dataobj).astype(np.float32)
laa_vox, _ = voxelwise_destruction(ct, lab.shape, laa_hu=LAA_HU)
del ct
_sum, _n = {}, {}
for L, meta in ti.items():
    m = lab == L
    n = int(m.sum())
    if n == 0:
        continue
    lb = meta['lobe']
    _sum[lb] = _sum.get(lb, 0.0) + float(laa_vox[m].sum())
    _n[lb] = _n.get(lb, 0) + n
LAA_lobe = {lb: _sum[lb] / _n[lb] for lb in _n if lb in LOBES and _n[lb]}

base = PARAMS.copy()
rng = np.random.default_rng(SEED)


def lobe_flow_fracs(topo, sol):
    """Frazioni di flusso per lobo (foglie della rete tenuta), ristrette ai 6 lobi
    e rinormalizzate a somma 1."""
    Q, Qt, kids = sol['Q'], sol['Q_tot'], topo['kids']
    lobe_of = topo['lobe_of']
    ql = {}
    for b in topo['branches']:
        bid = b['id']
        if bid not in Q:
            continue
        if any(c['id'] in Q for c in kids(b)):
            continue                      # non e' foglia della rete tenuta
        lb = lobe_of(b)
        if lb in LOBES:
            ql[lb] = ql.get(lb, 0.0) + Q[bid] / Qt
    tot = sum(ql.values()) or 1.0
    return {lb: ql.get(lb, 0.0) / tot for lb in LAA_lobe}


def index_and_shares(ql):
    """Indice I = sum w_lobo*LAA_lobo e quota di ciascun lobo sul carico (ds_share)."""
    contrib = {lb: ql.get(lb, 0.0) * LAA_lobe[lb] for lb in LAA_lobe}
    tot = sum(contrib.values())
    I = tot
    ds_share = {lb: (contrib[lb] / tot if tot else 0.0) for lb in LAA_lobe}
    return I, ds_share, contrib


def draw_perturbation():
    """Un pacchetto di perturbazioni congiunte per una replica (protocollo v1.0)."""
    # territori: lognormale ~10% per foglia -> topo ricostruita
    terr_r = {k: float(v) * float(np.exp(rng.normal(0, 0.10))) for k, v in terr.items()}
    topo_r = build_topology(tree, terr_r)
    lobe_of = topo_r['lobe_of']
    # diametri misurati: rumore per-misura +-10%
    dms = {b['id']: float(np.exp(rng.normal(0, 0.10)))
           for b in topo_r['branches'] if b.get('d_mean')}
    # diametri imputati: fattore comune per lobo (correlazione entro sottoalbero)
    # + componente indipendente per ramo
    lobe_f = {lb: float(np.exp(rng.normal(0, 0.10))) for lb in LOBES + ('CENTRAL',)}
    ims = {}
    for b in topo_r['branches']:
        if not b.get('d_mean'):
            ims[b['id']] = lobe_f.get(lobe_of(b), 1.0) * float(np.exp(rng.normal(0, 0.05)))
    # parametri di scenario
    P = base.copy()
    P['murray_exp'] = float(rng.uniform(2.6, 3.3))
    P['completion_Ld'] = float(rng.uniform(2.0, 4.0))
    P['completion_dstop_mm'] = float(rng.uniform(0.3, 0.8))
    P['pedley'] = bool(rng.random() > 0.3)
    # pruning: blocca una piccola frazione di foglie (topologia incerta)
    leaves = topo_r['leaves']
    k = int(len(leaves) * rng.uniform(0.0, 0.05))
    blocked = frozenset(rng.choice([b['id'] for b in leaves], size=k, replace=False)) if k else frozenset()
    return topo_r, P, dms, ims, blocked


# --- baseline (scenario di riferimento, nessuna perturbazione) ----------------
topo0 = build_topology(tree, terr)
sol0 = run_model(topo0, base)
ql0 = lobe_flow_fracs(topo0, sol0)
I0, ds0, _ = index_and_shares(ql0)
base_labels = {lb: classify_lobe(LAA_lobe[lb], ds0[lb]) for lb in LAA_lobe}
base_order = sorted(ds0, key=lambda lb: -ds0[lb])   # peggiore -> migliore per ds_share

# --- ensemble principale ------------------------------------------------------
I_samples, conv = [], 0
q_reps, share_reps, label_reps = [], [], []
for _ in range(N_MAIN):
    topo_r, P, dms, ims, blocked = draw_perturbation()
    sol = run_model(topo_r, P, blocked=blocked, d_meas_scale=dms, imp_scale=ims)
    conv += 1 if sol['converged'] else 0
    ql = lobe_flow_fracs(topo_r, sol)
    I, ds, _ = index_and_shares(ql)
    I_samples.append(round(100 * I, 3))
    q_reps.append({lb: round(100 * ql.get(lb, 0.0), 3) for lb in LAA_lobe})
    share_reps.append({lb: ds[lb] for lb in LAA_lobe})
    label_reps.append({lb: classify_lobe(LAA_lobe[lb], ds[lb]) for lb in LAA_lobe})

# --- quota di varianza dai diametri IMPUTATI vs MISURATI ----------------------
def sub_ensemble(meas_only):
    xs = []
    for _ in range(N_ABL):
        terr_r = ({k: float(v) * float(np.exp(rng.normal(0, 0.10))) for k, v in terr.items()}
                  if not meas_only else dict(terr))
        topo_r = build_topology(tree, terr_r)
        lobe_of = topo_r['lobe_of']
        if meas_only:
            dms = {b['id']: float(np.exp(rng.normal(0, 0.10)))
                   for b in topo_r['branches'] if b.get('d_mean')}
            ims, P = None, base.copy()
        else:
            dms = None
            lobe_f = {lb: float(np.exp(rng.normal(0, 0.10))) for lb in LOBES + ('CENTRAL',)}
            ims = {b['id']: lobe_f.get(lobe_of(b), 1.0) * float(np.exp(rng.normal(0, 0.05)))
                   for b in topo_r['branches'] if not b.get('d_mean')}
            P = base.copy()
            P['murray_exp'] = float(rng.uniform(2.6, 3.3))
            P['completion_Ld'] = float(rng.uniform(2.0, 4.0))
            P['completion_dstop_mm'] = float(rng.uniform(0.3, 0.8))
        sol = run_model(topo_r, P, d_meas_scale=dms, imp_scale=ims)
        I, _, _ = index_and_shares(lobe_flow_fracs(topo_r, sol))
        xs.append(100 * I)
    return float(np.var(xs))


var_meas = sub_ensemble(meas_only=True)
var_imp = sub_ensemble(meas_only=False)
share_imp = variance_share(var_imp, var_imp + var_meas)

# --- ablazione strutturale #5: q con vs senza condizionamento dal territorio ---
sol_t = run_model(topo0, base, territory_conditioned=True)
sol_a = run_model(topo0, base, territory_conditioned=False)
order_t = sorted(index_and_shares(lobe_flow_fracs(topo0, sol_t))[1].items(), key=lambda x: -x[1])
order_a = sorted(index_and_shares(lobe_flow_fracs(topo0, sol_a))[1].items(), key=lambda x: -x[1])
rank_t = [lb for lb, _ in order_t]
rank_a = [lb for lb, _ in order_a]
ablation_changes_rank = rank_t != rank_a

# --- aggregazione -------------------------------------------------------------
I_q = quantiles(I_samples)
q_bands = {lb: quantiles([r[lb] for r in q_reps]) for lb in LAA_lobe}
worst = worst_rank_prob(share_reps, higher_is_worse=True)
lab_stab = label_stability(label_reps, base_labels)
rank_stab = rank_stability(share_reps, base_order, thresh=STAB_THRESH)

per_lobo = {}
for lb in LAA_lobe:
    frac = rank_stab[lb]['frac']
    stabile = rank_stab[lb]['stabile'] and (lab_stab[lb] or 0) >= STAB_THRESH
    per_lobo[lb] = {
        'laa': round(LAA_lobe[lb], 3),
        'q_frac_pct': q_bands[lb],
        'ds_share_baseline': round(ds0[lb], 3),
        'prob_rango_peggiore': worst.get(lb),
        'stabilita_rango': frac,
        'stabilita_classe': lab_stab.get(lb),
        'etichetta_baseline': base_labels[lb],
        'informativo': bool(stabile),
    }

out = {
    'status': 'exploratory',
    'protocollo': 'v1.0 congelato',
    'scenario': base,
    'n_repliche': N_MAIN,
    'seed': SEED,
    'indice_globale_pct': {'baseline': round(100 * I0, 3), **{f'p{k}': v for k, v in I_q.items()}},
    'convergenza_frazione': round(conv / N_MAIN, 3),
    'quota_varianza_indice_da_imputati': share_imp,
    'ablazione_territorio': {
        'rango_con_territorio': rank_t, 'rango_airway_only': rank_a,
        'cambia_il_rango': bool(ablation_changes_rank),
        'nota': 'se cambia il rango lobare e incertezza STRUTTURALE del modello '
                '(non collinearita)'},
    'per_lobo': per_lobo,
}
json.dump(out, open('out/uncertainty.json', 'w'), indent=1)

print(f"Ensemble incertezza ({N_MAIN} repliche, seed {SEED}):")
print(f"  indice globale: baseline {100*I0:.1f}%  ·  5-95% [{I_q[5]}, {I_q[95]}]  (mediana {I_q[50]})")
print(f"  convergenza: {100*conv/N_MAIN:.0f}%  ·  quota varianza da imputati: {share_imp}")
print(f"  ablazione territorio: rango {'CAMBIA' if ablation_changes_rank else 'stabile'} "
      f"({'->'.join(rank_t)} vs {'->'.join(rank_a)})")
for lb in base_order:
    s = per_lobo[lb]
    print(f"  {lb:5s} q {s['q_frac_pct'][50]}% [{s['q_frac_pct'][5]},{s['q_frac_pct'][95]}] "
          f"· P(peggiore) {s['prob_rango_peggiore']} · stab.rango {s['stabilita_rango']} "
          f"· {'INFORMATIVO' if s['informativo'] else 'non informativo'}")

# ---------------------------- pagina --------------------------------------------
lobi = list(base_order)
med = [per_lobo[lb]['q_frac_pct'][50] for lb in lobi]
lo = [round(per_lobo[lb]['q_frac_pct'][50] - per_lobo[lb]['q_frac_pct'][5], 3) for lb in lobi]
hi = [round(per_lobo[lb]['q_frac_pct'][95] - per_lobo[lb]['q_frac_pct'][50], 3) for lb in lobi]
cols = ['#c0392b' if per_lobo[lb]['informativo'] else '#9a9891' for lb in lobi]


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


rows = ''.join(
    f"<tr><td>{lb}</td><td>{round(100*per_lobo[lb]['laa'],1)}</td>"
    f"<td>{per_lobo[lb]['q_frac_pct'][50]} [{per_lobo[lb]['q_frac_pct'][5]}, {per_lobo[lb]['q_frac_pct'][95]}]</td>"
    f"<td>{per_lobo[lb]['prob_rango_peggiore']}</td>"
    f"<td>{per_lobo[lb]['stabilita_rango']}</td>"
    f"<td>{per_lobo[lb]['stabilita_classe']}</td>"
    f"<td><b>{'informativo' if per_lobo[lb]['informativo'] else 'non informativo'}</b></td></tr>"
    for lb in lobi)

name = os.environ.get('AIRWAYLAB_CASE', 'caso')
ig = out['indice_globale_pct']
abl = out['ablazione_territorio']
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — incertezza (ensemble, esplorativo)</title>
<script>__PLOTLY__</script>
<style>
  .viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --page:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781; --border:rgba(11,11,11,.10); }}
  @media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff; --text-secondary:#c3c2b7;
    --muted:#898781; --border:rgba(255,255,255,.10); }} }}
  * {{ box-sizing:border-box }} body {{ margin:0 }}
  .viz-root {{ font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    background:var(--page); color:var(--text-primary); min-height:100vh; padding:24px }}
  .wrap {{ max-width:1080px; margin:0 auto }}
  h1 {{ font-size:20px; font-weight:650; margin:0 0 4px }}
  .sub {{ color:var(--text-secondary); font-size:13px; margin:0 0 16px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin-bottom:16px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px }}
  .tile .k {{ font-size:12px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:22px; font-weight:700 }} .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  table {{ border-collapse:collapse; width:100%; font-size:13px }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--border); white-space:nowrap }}
  th {{ color:var(--text-secondary); font-weight:600 }}
  #bands {{ width:100%; height:340px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Incertezza dell'indice e dei ranghi (ensemble)</h1>
<p class="sub">{name} · {N_MAIN} repliche, seed {SEED} · protocollo v1.0 congelato · propagazione congiunta (diametri misurati/imputati, Murray, completamento, Pedley, pruning, territori)</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Esplorativo.</b> L'indice è valutato a <b>granularità lobare</b> (per la stabilità dei ranghi); il valore puntuale della scheda multi-asse è a granularità territoriale, quindi differisce di qualche punto. <b>Solo i lobi marcati "informativo"</b> hanno rango e classe stabili (≥{int(STAB_THRESH*100)}%): sugli altri l'incertezza del modello non permette conclusioni. La convergenza numerica è separata dall'incertezza epistemica.</div>
<div class="tiles">
  <div class="tile"><div class="k">Indice globale (LAA pesata dal modello resistivo)</div><div class="v">{ig['p50']}<span class="u">%</span></div><div class="u">5–95%: {ig['p5']} – {ig['p95']} · baseline {ig['baseline']}</div></div>
  <div class="tile"><div class="k">Varianza dell'indice da diametri IMPUTATI</div><div class="v">{int(round(100*(out['quota_varianza_indice_da_imputati'] or 0)))}<span class="u">%</span></div><div class="u">il resto dai diametri misurati</div></div>
  <div class="tile"><div class="k">Convergenza numerica</div><div class="v">{int(round(100*out['convergenza_frazione']))}<span class="u">%</span></div><div class="u">delle repliche</div></div>
</div>
<div class="card">
  <h2>Quota di flusso simulato per lobo — mediana e intervallo 5–95%</h2>
  <p class="note">Rosso = lobo <b>informativo</b> (rango e classe stabili); grigio = <b>non informativo</b> (l'intervallo si sovrappone agli altri). Le barre d'errore mostrano quanto l'allocazione del modello è incerta.</p>
  <div id="bands"></div>
</div>
<div class="card">
  <h2>Ablazione strutturale (territorio sì/no)</h2>
  <p class="note">Rango dei lobi per quota del carico: <b>con territorio</b> {_esc(' → '.join(abl['rango_con_territorio']))}; <b>airway-only</b> {_esc(' → '.join(abl['rango_airway_only']))}. {'<b>Il rango cambia</b>: incertezza STRUTTURALE del modello (non semplice collinearità).' if abl['cambia_il_rango'] else 'Il rango non cambia con questa ablazione.'}</p>
</div>
<div class="card">
  <h2>Tabella per lobo</h2>
  <table><thead><tr><th>Lobo</th><th>LAA %</th><th>q mediana [5–95] %</th><th>P(peggiore)</th><th>stab. rango</th><th>stab. classe</th><th>esito</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
const L={json.dumps(lobi)}, MED={json.dumps(med)}, LO={json.dumps(lo)}, HI={json.dumps(hi)}, C={json.dumps(cols)};
Plotly.newPlot('bands', [{{type:'bar', x:L, y:MED, marker:{{color:C}},
  error_y:{{type:'data', symmetric:false, array:HI, arrayminus:LO, color:'#52514e', thickness:1.5, width:5}},
  hovertemplate:'%{{x}} · mediana %{{y}}%<extra></extra>'}}], {{
  margin:{{l:44,r:10,t:8,b:30}}, yaxis:{{title:{{text:'quota flusso simulato (%)'}},color:'#898781'}},
  xaxis:{{color:'#898781'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'plotly.min.js'),
          encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/uncertainty.html', 'w', encoding='utf-8').write(html)
print(f"out/uncertainty.html ({len(html)//1024//1024} MB)")
