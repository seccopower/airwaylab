"""Ensemble di ROBUSTEZZA del modello di flusso (protocollo #6) — compute + pagina.

ATTENZIONE alla natura di questo ensemble (review GPT, riconsiderazione):
le distribuzioni degli input (±10%, Murray 2.6-3.3, Pedley on/off, pruning, territori)
sono SCENARI PLAUSIBILI PRESPECIFICATI, non distribuzioni probabilistiche calibrate
empiricamente. Quindi l'ensemble misura la ROBUSTEZZA degli output alle assunzioni
prespecificate; NON e' una quantificazione probabilistica calibrata dell'incertezza del
paziente. Gli intervalli riportati sono percentili dell'ensemble prespecificato.

Propaga in modo congiunto fino agli endpoint LOBARI: quote di flusso simulato per lobo e
indice I = media della frazione LAA pesata dalle quote di flusso. Riporta:
 - mediana e percentili 5-95 dell'ensemble, con CONTROLLO DI CONVERGENZA Monte Carlo
   (N=300/600/1200: stabilita' di mediana, code, frequenze di rango, classificazione);
 - frequenza con cui ogni lobo ha l'indice piu' alto (non "probabilita'");
 - robustezza del rango NELL'ensemble prespecificato e robustezza ALLA FORMA del modello
   (ablazione territorio on/off), tenute DISTINTE;
 - un'ablazione one-at-a-time sui soli diametri (misurati vs imputati), descritta
   letteralmente (NON una decomposizione di varianza completa: interazioni non assegnate).
Convergenza numerica del solver riportata separatamente dall'incertezza epistemica.

Riusa flow_model (stessa fisica di produzione) e morphomap_core (LAA). Deterministico: seed fisso.
Run nel work dir dopo flow.py + territory.py + morphomap. Output: uncertainty.json + uncertainty.html.
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
    worst_rank_prob,
)

SEED = 12345
N_SIZES = (300, 600, 1200)     # controllo di convergenza; il piu' grande e' il riferimento
N_MAIN = N_SIZES[-1]
N_ABL = 300                    # ablazione OAT sui diametri
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

# --- LAA per lobo (fisso, dalla TC) ------------------------------------------
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
    Q, Qt, kids = sol['Q'], sol['Q_tot'], topo['kids']
    lobe_of = topo['lobe_of']
    ql = {}
    for b in topo['branches']:
        bid = b['id']
        if bid not in Q or any(c['id'] in Q for c in kids(b)):
            continue
        lb = lobe_of(b)
        if lb in LOBES:
            ql[lb] = ql.get(lb, 0.0) + Q[bid] / Qt
    tot = sum(ql.values()) or 1.0
    return {lb: ql.get(lb, 0.0) / tot for lb in LAA_lobe}


def index_and_shares(ql):
    contrib = {lb: ql.get(lb, 0.0) * LAA_lobe[lb] for lb in LAA_lobe}
    tot = sum(contrib.values())
    return tot, {lb: (contrib[lb] / tot if tot else 0.0) for lb in LAA_lobe}


def draw_replicate():
    terr_r = {k: float(v) * float(np.exp(rng.normal(0, 0.10))) for k, v in terr.items()}
    topo_r = build_topology(tree, terr_r)
    lobe_of = topo_r['lobe_of']
    dms = {b['id']: float(np.exp(rng.normal(0, 0.10)))
           for b in topo_r['branches'] if b.get('d_mean')}
    lobe_f = {lb: float(np.exp(rng.normal(0, 0.10))) for lb in LOBES + ('CENTRAL',)}
    ims = {b['id']: lobe_f.get(lobe_of(b), 1.0) * float(np.exp(rng.normal(0, 0.05)))
           for b in topo_r['branches'] if not b.get('d_mean')}
    P = base.copy()
    P['murray_exp'] = float(rng.uniform(2.6, 3.3))
    P['completion_Ld'] = float(rng.uniform(2.0, 4.0))
    P['completion_dstop_mm'] = float(rng.uniform(0.3, 0.8))
    P['pedley'] = bool(rng.random() > 0.3)
    leaves = topo_r['leaves']
    k = int(len(leaves) * rng.uniform(0.0, 0.05))
    blocked = frozenset(rng.choice([b['id'] for b in leaves], size=k, replace=False)) if k else frozenset()
    return topo_r, P, dms, ims, blocked


# --- baseline (scenario prespecificato, nessuna perturbazione) ----------------
topo0 = build_topology(tree, terr)
sol0 = run_model(topo0, base)
I0, ds0 = index_and_shares(lobe_flow_fracs(topo0, sol0))
base_labels = {lb: classify_lobe(LAA_lobe[lb], ds0[lb]) for lb in LAA_lobe}
base_order = sorted(ds0, key=lambda lb: -ds0[lb])

# --- ensemble principale (N_MAIN) --------------------------------------------
I_samples, conv = [], 0
q_reps, share_reps, label_reps = [], [], []
for _ in range(N_MAIN):
    topo_r, P, dms, ims, blocked = draw_replicate()
    sol = run_model(topo_r, P, blocked=blocked, d_meas_scale=dms, imp_scale=ims)
    conv += 1 if sol['converged'] else 0
    I, ds = index_and_shares(lobe_flow_fracs(topo_r, sol))
    ql = lobe_flow_fracs(topo_r, sol)
    I_samples.append(I)
    q_reps.append({lb: round(100 * ql.get(lb, 0.0), 3) for lb in LAA_lobe})
    share_reps.append({lb: ds[lb] for lb in LAA_lobe})
    label_reps.append({lb: classify_lobe(LAA_lobe[lb], ds[lb]) for lb in LAA_lobe})

# --- controllo di convergenza Monte Carlo su prefissi -------------------------
conv_table = []
for n in N_SIZES:
    Iq = quantiles([100 * x for x in I_samples[:n]])
    top = worst_rank_prob(share_reps[:n], higher_is_worse=True)
    rll_top = top.get('RLL')
    conv_table.append({'N': n, 'mediana': Iq[50], 'p5': Iq[5], 'p95': Iq[95],
                       'freq_RLL_indice_max': rll_top})

# --- ablazione OAT sui diametri (binaria pulita: tutto il resto FISSO) --------
def oat_diam(measured):
    xs = []
    for _ in range(N_ABL):
        if measured:
            dms = {b['id']: float(np.exp(rng.normal(0, 0.10)))
                   for b in topo0['branches'] if b.get('d_mean')}
            ims = None
        else:
            lobe_of = topo0['lobe_of']
            lobe_f = {lb: float(np.exp(rng.normal(0, 0.10))) for lb in LOBES + ('CENTRAL',)}
            ims = {b['id']: lobe_f.get(lobe_of(b), 1.0) * float(np.exp(rng.normal(0, 0.05)))
                   for b in topo0['branches'] if not b.get('d_mean')}
            dms = None
        sol = run_model(topo0, base, d_meas_scale=dms, imp_scale=ims)  # scenario e territori FISSI
        I, _ = index_and_shares(lobe_flow_fracs(topo0, sol))
        xs.append(100 * I)
    return float(np.std(xs))


sd_meas = oat_diam(measured=True)
sd_imp = oat_diam(measured=False)

# --- ablazione strutturale (forma del modello): territorio on/off -------------
sol_t = run_model(topo0, base, territory_conditioned=True)
sol_a = run_model(topo0, base, territory_conditioned=False)
_, ds_t = index_and_shares(lobe_flow_fracs(topo0, sol_t))
_, ds_a = index_and_shares(lobe_flow_fracs(topo0, sol_a))
rank_t = sorted(ds_t, key=lambda lb: -ds_t[lb])
rank_a = sorted(ds_a, key=lambda lb: -ds_a[lb])
pos_t = {lb: i for i, lb in enumerate(rank_t)}
pos_a = {lb: i for i, lb in enumerate(rank_a)}
rank_unchanged_ablation = {lb: (pos_t[lb] == pos_a[lb]) for lb in LAA_lobe}

# --- aggregazione -------------------------------------------------------------
I_q = quantiles([100 * x for x in I_samples])
q_bands = {lb: quantiles([r[lb] for r in q_reps]) for lb in LAA_lobe}
freq_top = worst_rank_prob(share_reps, higher_is_worse=True)
lab_stab = label_stability(label_reps, base_labels)
rank_stab = rank_stability(share_reps, base_order, thresh=STAB_THRESH)

per_lobo = {}
for lb in LAA_lobe:
    rango_robusto_ens = rank_stab[lb]['stabile']
    robusto_forma = bool(rank_unchanged_ablation[lb])
    per_lobo[lb] = {
        'laa': round(LAA_lobe[lb], 3),
        'q_frac_pct': q_bands[lb],
        'ds_share_baseline': round(ds0[lb], 3),
        'freq_indice_max': freq_top.get(lb),
        'rango_robusto_ensemble': bool(rango_robusto_ens),
        'freq_rango_baseline': rank_stab[lb]['frac'],
        'freq_permanenza_classe': lab_stab.get(lb),
        'rango_robusto_forma_modello': robusto_forma,
        'etichetta_baseline_espl': base_labels[lb],
    }

out = {
    'status': 'exploratory',
    'natura': 'ensemble di ROBUSTEZZA a scenari prespecificati (input NON calibrati '
              'empiricamente): non una quantificazione probabilistica dell incertezza del '
              'paziente. Intervalli = percentili dell ensemble prespecificato.',
    'protocollo': 'v1.1 (chiarimenti di terminologia + convergenza; estimando invariato da v1.0)',
    'scenario': base,
    'seed': SEED,
    'convergenza_montecarlo': conv_table,
    'indice_LAA_pesata': {
        'baseline': round(I0, 4),
        'mediana_ensemble': round(I_q[50] / 100, 4),
        'p5': round(I_q[5] / 100, 4), 'p95': round(I_q[95] / 100, 4),
        'nota': 'frazione di LAA pesata dalle quote di flusso; equivalente in punti '
                'percentuali di LAA. NON e una % di flusso o ventilazione.'},
    'convergenza_solver_frazione': round(conv / N_MAIN, 3),
    'ablazione_diametri_OAT': {
        'sd_indice_pp_solo_misurati': round(sd_meas / 100, 4),
        'sd_indice_pp_solo_imputati': round(sd_imp / 100, 4),
        'rapporto_imputati_su_misurati': round(sd_imp / sd_meas, 2) if sd_meas else None,
        'nota': 'deviazione standard dell indice sotto perturbazione ONE-AT-A-TIME dei soli '
                'diametri (misurati vs imputati), TUTTI gli altri fattori fissi. NON e una '
                'decomposizione di varianza: interazioni e altri fattori non assegnati; input '
                'trattati come scenari, non distribuzioni calibrate.'},
    'ablazione_forma_modello': {
        'rango_con_territorio': rank_t, 'rango_airway_only': rank_a,
        'rango_lobare_complessivo_cambia': bool(rank_t != rank_a),
        'nota': 'se il rango di un lobo cambia tra i due scenari, quel lobo NON e robusto '
                'alla forma del modello (incertezza strutturale, non collinearita)'},
    'per_lobo': per_lobo,
}
json.dump(out, open('out/uncertainty.json', 'w'), indent=1)

print(f"Ensemble di robustezza ({N_MAIN} repliche, seed {SEED}):")
print(f"  indice LAA pesata: baseline {I0:.3f} · mediana {I_q[50]/100:.3f} · "
      f"p5-p95 [{I_q[5]/100:.3f}, {I_q[95]/100:.3f}] (percentili ensemble prespecificato)")
print("  convergenza MC:", [(c['N'], c['p5'], c['p95'], c['freq_RLL_indice_max']) for c in conv_table])
print(f"  ablazione diametri OAT: sd imputati {sd_imp/100:.4f} vs misurati {sd_meas/100:.4f} "
      f"(rapporto {sd_imp/sd_meas:.1f}x)")
print(f"  ablazione forma: territorio {'->'.join(rank_t)} | airway-only {'->'.join(rank_a)}")
for lb in base_order:
    s = per_lobo[lb]
    print(f"  {lb:5s} q {s['q_frac_pct'][50]}% [{s['q_frac_pct'][5]},{s['q_frac_pct'][95]}] · "
          f"freq indice max {s['freq_indice_max']} · rango robusto ens {s['rango_robusto_ensemble']} · "
          f"robusto forma {s['rango_robusto_forma_modello']}")

# ---------------------------- pagina --------------------------------------------
lobi = list(base_order)
med = [per_lobo[lb]['q_frac_pct'][50] for lb in lobi]
lo = [round(per_lobo[lb]['q_frac_pct'][50] - per_lobo[lb]['q_frac_pct'][5], 3) for lb in lobi]
hi = [round(per_lobo[lb]['q_frac_pct'][95] - per_lobo[lb]['q_frac_pct'][50], 3) for lb in lobi]
# rosso solo se robusto SIA nell'ensemble SIA alla forma del modello
cols = ['#c0392b' if (per_lobo[lb]['rango_robusto_ensemble'] and per_lobo[lb]['rango_robusto_forma_modello'])
        else '#9a9891' for lb in lobi]


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _rob(lb):
    e = per_lobo[lb]['rango_robusto_ensemble']
    f = per_lobo[lb]['rango_robusto_forma_modello']
    if e and f:
        return 'robusto (ensemble + forma)'
    if e and not f:
        return 'robusto nell\'ensemble, non alla forma del modello'
    return 'rango non robusto'


rows = ''.join(
    f"<tr><td>{lb}</td><td>{round(100*per_lobo[lb]['laa'],1)}</td>"
    f"<td>{per_lobo[lb]['q_frac_pct'][50]} [{per_lobo[lb]['q_frac_pct'][5]}, {per_lobo[lb]['q_frac_pct'][95]}]</td>"
    f"<td>{per_lobo[lb]['freq_indice_max']}</td>"
    f"<td>{per_lobo[lb]['freq_rango_baseline']}</td>"
    f"<td>{per_lobo[lb]['freq_permanenza_classe']}</td>"
    f"<td>{_esc(_rob(lb))}</td></tr>"
    for lb in lobi)

conv_rows = ''.join(
    f"<tr><td>{c['N']}</td><td>{round(c['mediana']/100,3)}</td>"
    f"<td>{round(c['p5']/100,3)}</td><td>{round(c['p95']/100,3)}</td>"
    f"<td>{c['freq_RLL_indice_max']}</td></tr>" for c in conv_table)

name = os.environ.get('AIRWAYLAB_CASE', 'caso')
ix = out['indice_LAA_pesata']
abl = out['ablazione_forma_modello']
ad = out['ablazione_diametri_OAT']
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — robustezza (ensemble, esplorativo)</title>
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
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; margin-bottom:16px }}
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
<h1>Robustezza dell'indice e dei ranghi (ensemble)</h1>
<p class="sub">{name} · {N_MAIN} repliche, seed {SEED} · protocollo v1.1 · propagazione congiunta di scenari prespecificati</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Ensemble di ROBUSTEZZA a scenari prespecificati, non incertezza calibrata.</b> Gli input (±10%, Murray 2.6–3.3, Pedley on/off, pruning, territori) sono <b>scenari plausibili</b>, non distribuzioni empiriche: gli intervalli sono <b>percentili dell'ensemble prespecificato</b>, non intervalli di confidenza. Un lobo è marcato robusto solo se il suo rango è stabile <b>sia</b> nell'ensemble <b>sia</b> alla forma del modello (ablazione territorio). La convergenza numerica del solver è separata da questa robustezza.</div>
<div class="tiles">
  <div class="tile"><div class="k">Indice LAA pesata dal modello resistivo</div><div class="v">{ix['mediana_ensemble']}</div><div class="u">frazione (= {round(100*ix['mediana_ensemble'],1)} punti perc. di LAA) · p5–p95 {ix['p5']}–{ix['p95']} · baseline {ix['baseline']}</div></div>
  <div class="tile"><div class="k">Ablazione diametri (OAT): sd indice</div><div class="v">{ad['rapporto_imputati_su_misurati']}×</div><div class="u">imputati {ad['sd_indice_pp_solo_imputati']} vs misurati {ad['sd_indice_pp_solo_misurati']} (frazione) — non una decomposizione di varianza</div></div>
  <div class="tile"><div class="k">Convergenza numerica solver</div><div class="v">{int(round(100*out['convergenza_solver_frazione']))}<span class="u">%</span></div><div class="u">delle repliche</div></div>
</div>
<div class="card">
  <h2>Quota di flusso simulato per lobo — mediana e percentili 5–95 dell'ensemble</h2>
  <p class="note">Rosso = lobo con rango <b>robusto sia nell'ensemble sia alla forma del modello</b>; grigio = rango non robusto (intervalli sovrapposti o dipendente dalla forma del modello). Le barre d'errore sono l'ampiezza p5–p95 dell'ensemble prespecificato, non un intervallo di confidenza.</p>
  <div id="bands"></div>
</div>
<div class="card">
  <h2>Convergenza Monte Carlo</h2>
  <p class="note">Stabilita' di mediana, code p5/p95 e frequenza del RLL come indice più alto, all'aumentare delle repliche. Un seed fisso dà riproducibilità, non accuratezza: i valori devono stabilizzarsi.</p>
  <table><thead><tr><th>N repliche</th><th>mediana</th><th>p5</th><th>p95</th><th>freq. RLL indice max</th></tr></thead>
  <tbody>{conv_rows}</tbody></table>
</div>
<div class="card">
  <h2>Ablazione strutturale (forma del modello: territorio sì/no)</h2>
  <p class="note">Rango dei lobi per quota del carico: <b>con territorio</b> {_esc(' → '.join(abl['rango_con_territorio']))}; <b>airway-only</b> {_esc(' → '.join(abl['rango_airway_only']))}. I lobi che cambiano posizione tra i due scenari <b>non sono robusti alla forma del modello</b> (incertezza strutturale, non collinearità). Pedley e territorio andrebbero visti come scenari separati: fonderli in un'unica distribuzione assegnerebbe implicitamente un peso ai modelli.</p>
</div>
<div class="card">
  <h2>Tabella per lobo</h2>
  <table><thead><tr><th>Lobo</th><th>LAA %</th><th>q mediana [p5–p95] %</th><th>freq. indice max</th><th>freq. rango baseline</th><th>freq. permanenza classe</th><th>robustezza del rango</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
const L={json.dumps(lobi)}, MED={json.dumps(med)}, LO={json.dumps(lo)}, HI={json.dumps(hi)}, C={json.dumps(cols)};
Plotly.newPlot('bands', [{{type:'bar', x:L, y:MED, marker:{{color:C}},
  error_y:{{type:'data', symmetric:false, array:HI, arrayminus:LO, color:'#52514e', thickness:1.5, width:5}},
  hovertemplate:'%{{x}} · mediana %{{y}}% (p5–p95)<extra></extra>'}}], {{
  margin:{{l:44,r:10,t:8,b:30}}, yaxis:{{title:{{text:'quota flusso simulato (%)'}},color:'#898781'}},
  xaxis:{{color:'#898781'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'plotly.min.js'),
          encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/uncertainty.html', 'w', encoding='utf-8').write(html)
print(f"out/uncertainty.html ({len(html)//1024//1024} MB)")
