"""Visualizzazione del modello di flusso 1D (ESPLORATIVO) — pagina standalone.

L'albero 3D del paziente colorato dalla fisica:
  - colore del ramo   = caduta di pressione cumulativa dall'ingresso (sequenziale,
                        un solo colore chiaro->scuro: magnitudo)
  - spessore del ramo = flusso che lo attraversa (radice quadrata, per leggibilita')
  - punti terminali   = ventilazione specifica del territorio (divergente attorno a 1:
                        blu = ipo-ventilato, arancio = iper-ventilato, grigio = atteso)
  - hover             = ogni ramo dichiara d (misurato/imputato), Q, v, dP, Re; watermark 'SIMULAZIONE'

Run nel work dir dopo flow.py. Output: out/flow_viz.html (autonomo, plotly inline).
"""
import json
import os

import numpy as np

flow = json.load(open('out/flow.json'))

# --- gate di convergenza (review r3, major #8): mai presentare come normale ---
# una soluzione numerica non convergente. Se il punto fisso non e' arrivato a
# convergenza si emette SOLO una pagina di errore, senza alcun valore del modello.
conv = flow.get('convergenza', {})
if not conv.get('converged', False):
    name = os.environ.get('AIRWAYLAB_CASE', 'caso')
    err = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — flusso: soluzione non valida</title>
<style>body{{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
background:#f9f9f7;color:#0b0b0b;min-height:100vh;display:grid;place-items:center;padding:24px}}
.box{{max-width:640px;background:#7a1f1f;color:#fff;border-radius:12px;padding:24px 28px}}
h1{{font-size:18px;margin:0 0 10px}} p{{font-size:14px;line-height:1.5;margin:0 0 8px}}
code{{background:rgba(255,255,255,.15);border-radius:4px;padding:1px 5px}}</style>
</head><body><div class="box">
<h1>⚠ SOLUZIONE NUMERICA NON VALIDA</h1>
<p>Il punto fisso del modello di flusso (correzione di Pedley) <b>non è arrivato a
convergenza</b> dopo {conv.get('iterazioni', '?')} iterazioni
(residuo relativo {conv.get('residuo_rel', '?')}).</p>
<p>Nessun valore del modello viene mostrato: i numeri di una soluzione non
convergente non sono interpretabili nemmeno come simulazione esplorativa.</p>
<p>Rieseguire <code>flow.py</code> dopo aver verificato geometria e parametri
(<code>out/flow.json → convergenza</code>).</p>
</div></body></html>"""
    open('out/flow_viz.html', 'w', encoding='utf-8').write(err)
    print('out/flow_viz.html: SOLUZIONE NON CONVERGENTE — emessa pagina di errore')
    raise SystemExit(0)

data = json.load(open('out/map_data.json'))
PR = flow['per_ramo']
SV = flow.get('model_svent_terminali', {})

branches = data['branches']
pmax = max(v['model_p_cum_pa'] for v in PR.values()) or 1.0
qmax = max(v['model_q_ml_s'] for v in PR.values()) or 1.0

# sequenziale blu (un solo hue, chiaro->scuro) per la pressione cumulativa
SEQ = ['#dbe9f9', '#b3d0f0', '#86b6ef', '#5598e7', '#2a78d6', '#1c5cab', '#0d366b']
# divergente per la ventilazione specifica: blu <- grigio -> arancio
DIV_LO, DIV_MID, DIV_HI = '#2a78d6', '#9a9891', '#eb6834'


def seq_color(frac):
    i = min(len(SEQ) - 1, int(frac * len(SEQ)))
    return SEQ[i]


def lerp(a, b, t):
    ah = [int(a[i:i+2], 16) for i in (1, 3, 5)]
    bh = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(int(ah[k] + (bh[k]-ah[k]) * t) for k in range(3))


def div_color(sv):
    x = float(np.clip(np.log2(max(1e-3, sv)), -2, 2)) / 2.0   # [-1..1]
    return lerp(DIV_MID, DIV_HI, x) if x >= 0 else lerp(DIV_MID, DIV_LO, -x)


traces = []
for b in branches:
    pr = PR.get(b['id'])
    if not pr:
        continue
    frac = pr['model_p_cum_pa'] / pmax
    w = 1.5 + 6.5 * np.sqrt(pr['model_q_ml_s'] / qmax)
    est = '' if pr['d_da_misura'] else ' (imputato)'
    traces.append({
        'type': 'scatter3d', 'mode': 'lines',
        'x': b['x'], 'y': b['y'], 'z': b['z'],
        'line': {'color': seq_color(frac), 'width': float(w)},
        'hovertemplate': (f"{b.get('name') or b['id']}<br>"
                          f"Ø {pr['model_d_mm']} mm{est} · Q {pr['model_q_ml_s']:.1f} ml/s · "
                          f"v {pr['model_v_ms']:.2f} m/s<br>"
                          f"ΔP {pr['model_dp_pa']:.2f} Pa · P cum {pr['model_p_cum_pa']:.2f} Pa · "
                          f"Re {pr['Re']:.0f}{' · Pedley' if pr['pedley_attivo'] else ''}<br>"
                          f"<b>SIMULAZIONE — non misura</b><extra></extra>"),
        'showlegend': False,
    })

# terminali: ventilazione specifica (divergente)
tx, ty, tz, tc, tt = [], [], [], [], []
for b in branches:
    if b['id'] in SV:
        tx.append(b['x'][-1]); ty.append(b['y'][-1]); tz.append(b['z'][-1])
        tc.append(div_color(SV[b['id']]))
        tt.append(f"{b.get('name') or b['id']}<br>vent. specifica {SV[b['id']]:.2f}"
                  "<extra></extra>")
traces.append({'type': 'scatter3d', 'mode': 'markers',
               'x': tx, 'y': ty, 'z': tz,
               'marker': {'size': 3.5, 'color': tc},
               'hovertemplate': tt, 'showlegend': False})

gen_share = flow['model_quota_caduta_pct_per_generazione']
raw = flow['model_raw_albero_cmH2O_s_L']
cv = flow['model_ventilazione_specifica']['cv']

plugs = flow.get('occlusione_plug', [])
plug_txt = ' · '.join(
    f"{p['plug']} ({p['zona']}): −{p['ml_esclusi']} ml, ΔRaw {p['delta_raw_pct']:+}%"
    + ('' if p.get('converged', True) else ' ⚠ scenario non convergente')
    for p in plugs) or '—'
sens = flow.get('sensibilita_one_at_a_time', [])
frac_meas = flow.get('quota_dissipazione_da_diametri_misurati_pct', '?')

html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>__NAME__ — flusso simulato (esplorativo)</title>
<script>__PLOTLY__</script>
<style>
  .viz-root {{ color-scheme: light;
    --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
    --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
    --border:rgba(11,11,11,.10); --series-1:#2a78d6; }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
      --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff;
      --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
      --border:rgba(255,255,255,.10); --series-1:#3987e5; }} }}
  * {{ box-sizing:border-box }} body {{ margin:0 }}
  .viz-root {{ font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    background:var(--page); color:var(--text-primary); min-height:100vh; padding:24px }}
  .wrap {{ max-width:1100px; margin:0 auto }}
  h1 {{ font-size:20px; font-weight:650; margin:0 0 4px }}
  .sub {{ color:var(--text-secondary); font-size:13px; margin:0 0 18px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
    gap:12px; margin-bottom:16px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:14px 16px }}
  .tile .k {{ font-size:12px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:24px; font-weight:650 }}
  .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .card {{ background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12px;
    color:var(--text-secondary); margin-top:8px }}
  .legend span {{ display:inline-flex; align-items:center; gap:6px }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block }}
  #tree3d {{ width:100%; height:600px }} #gen {{ width:100%; height:280px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Flusso d'aria simulato sull'albero misurato</h1>
<p class="sub">__NAME__ · rete di resistenze 1D risolta come albero serie/parallelo
 (Poiseuille+Pedley) · calibri half-max dove reportabili, imputazione asimmetrica per
 territorio altrove · {flow['condizione']}</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>SIMULAZIONE — non è una misura del paziente.</b> Parametri non ancora validati;
  ogni numero va confrontato con pletismografia/spirometria/MBW prima di qualunque interpretazione.</div>
<div class="tiles">
  <div class="tile"><div class="k">Raw modello (tracheobronchiale)</div>
    <div class="v">{raw} <span class="u">cmH₂O·s/L</span></div>
    <div class="u">simulata · vie alte escluse</div></div>
  <div class="tile"><div class="k">Diametri misurati / imputati</div>
    <div class="v">{flow['diametri_misurati']} <span class="u">/ {flow['diametri_imputati']}</span></div>
    <div class="u">{frac_meas}% della dissipazione da misura</div></div>
  <div class="tile"><div class="k">Eterogeneità ventilazione (CV)</div>
    <div class="v">{cv}</div><div class="u">confronto tra casi/tempi, non assoluto</div></div>
  <div class="tile"><div class="k">Sensibilità Raw (Pedley on/off)</div>
    <div class="v">{next((s['model_raw'] for s in sens if s['parametro']=='pedley' and s['valore']),raw)}
      <span class="u">/ {next((s['model_raw'] for s in sens if s['parametro']=='pedley' and not s['valore']),'?')}</span></div>
    <div class="u">Poiseuille puro molto più basso</div></div>
</div>
<div class="card">
  <h2>Albero colorato dalla fisica</h2>
  <p class="note">Colore = caduta di pressione cumulativa dall'ingresso (chiaro→scuro).
     Spessore = flusso. Punti terminali = ventilazione specifica del territorio.
     Passa il mouse su un ramo per Ø (misurato o stimato), flusso, velocità e ΔP.</p>
  <div id="tree3d"></div>
  <div class="legend">
    <span><span class="sw" style="background:linear-gradient(90deg,{SEQ[0]},{SEQ[-1]})"></span> P cumulativa (0 → {pmax:.1f} Pa)</span>
    <span><span class="sw" style="background:{DIV_LO}"></span> territorio ipo-ventilato</span>
    <span><span class="sw" style="background:{DIV_MID}"></span> come atteso (=1)</span>
    <span><span class="sw" style="background:{DIV_HI}"></span> iper-ventilato</span>
  </div>
</div>
<div class="card">
  <h2>Dove cade la pressione, per generazione</h2>
  <p class="note">Quota % della dissipazione totale. Nel soggetto normale il grosso sta nelle vie di conduzione centrali e medie.</p>
  <div id="gen"></div>
</div>
<div class="card">
  <h2>Esperimento di occlusione (candidati plug)</h2>
  <p class="note">{plug_txt}</p>
  <p class="note">Nel modello un singolo plug periferico sposta poco la Raw globale; l'effetto è
     regionale (territorio de-ventilato). Osservazione del modello, NON claim clinico validato.</p>
</div>
</div></div>
<script>
const TR = __TRACES__;
Plotly.newPlot('tree3d', TR, {{
  margin:{{l:0,r:0,t:0,b:0}}, showlegend:false,
  scene:{{ xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}},
          aspectmode:'data', camera:{{eye:{{x:0,y:-1.7,z:0.15}},up:{{x:0,y:0,z:1}}}},
          bgcolor:'rgba(0,0,0,0)' }},
  paper_bgcolor:'rgba(0,0,0,0)',
}}, {{displayModeBar:false, responsive:true}});
const G = __GEN__;
const gl = Object.keys(G), gv = gl.map(k=>G[k]);
Plotly.newPlot('gen', [{{ type:'bar', x:gl, y:gv,
  marker:{{color:'#2a78d6'}},
  hovertemplate:'gen %{{x}} · %{{y:.1f}}%<extra></extra>' }}], {{
  margin:{{l:44,r:10,t:8,b:36}},
  xaxis:{{title:{{text:'generazione'}},color:'#898781'}},
  yaxis:{{title:{{text:'% della caduta di pressione'}},color:'#898781',gridcolor:'rgba(137,135,129,.25)'}},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
}}, {{displayModeBar:false, responsive:true}});
</script></body></html>"""

html = html.replace('__TRACES__', json.dumps(traces))
html = html.replace('__GEN__', json.dumps(gen_share))
name = os.environ.get('AIRWAYLAB_CASE', 'caso')
html = html.replace('__NAME__', name)
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets', 'plotly.min.js'), encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/flow_viz.html', 'w', encoding='utf-8').write(html)
print(f'out/flow_viz.html ({len(html)//1024//1024} MB)')
