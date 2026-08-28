"""Visualizzazione del modello di flusso 1D (ESPLORATIVO) — pagina standalone.

L'albero 3D del paziente colorato dalla fisica, PIU' quattro pannelli di lettura
(pure CSS/SVG, nessun grafico Plotly aggiuntivo):
  - albero 3D   : colore = caduta di pressione cumulativa; spessore = flusso;
                  punti terminali = ventilazione specifica (divergente attorno a 1)
  - per lobo    : dove cade la pressione, vista anatomica (barre orizzontali)
  - misura/modello: quota della dissipazione da diametri MISURATI vs imputati
  - ventilazione: distribuzione della ventilazione specifica (istogramma log)
  - occlusione  : esperimento plug prima/dopo (Raw globale vs territorio perso)

Tutti i numeri restano SIMULAZIONE. Run nel work dir dopo flow.py.
Output: out/flow_viz.html (autonomo, plotly inline).
"""
import json
import math
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


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


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

raw = flow['model_raw_albero_cmH2O_s_L']
cv = flow['model_ventilazione_specifica']['cv']
sens = flow.get('sensibilita_one_at_a_time', [])
frac_meas = flow.get('quota_dissipazione_da_diametri_misurati_pct', 0.0)

# ---------- pannello 01: caduta di pressione per lobo -------------------------
LOBE_IT = {'CENTRAL': 'Centrali', 'LING': 'Lingula'}
lobe_share = flow.get('model_quota_caduta_pct_per_lobo', {})
periph_share = round(100.0 - sum(lobe_share.values()), 1)
items = [(LOBE_IT.get(k, k), v, False) for k, v in lobe_share.items()]
if periph_share > 0.05:
    items.append(('Periferia*', periph_share, True))     # completamento imputato
items.sort(key=lambda t: -t[1])
vmax_l = max([v for _, v, _ in items]) or 1.0
_lrows = []
for lab, v, hatched in items:
    w = 100.0 * v / vmax_l
    if hatched:
        fill = 'class="lfill lhatch"'
    else:
        col = SEQ[min(len(SEQ) - 1, int(v / vmax_l * len(SEQ)))]
        fill = f'class="lfill" style="width:{w:.1f}%;background:{col}"'
    if hatched:
        fill = f'class="lfill lhatch" style="width:{w:.1f}%"'
    _lrows.append(f'<div class="lrow"><div class="llab">{_esc(lab)}</div>'
                  f'<div class="ltrack"><div {fill}></div></div>'
                  f'<div class="lval">{v}%</div></div>')
lobe_bars = ''.join(_lrows)

# ---------- pannello 03: misura vs modello -----------------------------------
n_meas = flow.get('diametri_misurati', 0)
n_imp = flow.get('diametri_imputati', 0)
fm = float(frac_meas) if frac_meas not in (None, '?') else 0.0
fi = round(100.0 - fm, 1)

# ---------- pannello 05: distribuzione della ventilazione specifica -----------
EDGES = [0, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 8.0, 16.0, float('inf')]
XLAB = ['&lt;.5', '.5–.7', '.7–1', '1–1.4', '1.4–2', '2–2.8', '2.8–4', '4–8', '8–16', '&gt;16']
counts = [0] * (len(EDGES) - 1)
for v in SV.values():
    for i in range(len(EDGES) - 1):
        if EDGES[i] <= v < EDGES[i + 1]:
            counts[i] += 1
            break
cmax = max(counts) or 1
_BLUE3 = ['#2a78d6', '#5598e7', '#86b6ef']    # i<=2 (<1), saturo lontano da 1
_OR_OP = [0.5, 0.62, 0.72, 0.82, 0.9, 0.96, 1.0]   # i>=3 (>1), opacita' crescente


def _hcolor(i):
    if i <= 2:
        return _BLUE3[i]
    return f'rgba(235,104,52,{_OR_OP[i - 3]})'


_hbars = ''.join(
    f'<div class="hbar" style="height:{max(2.0, 100.0 * c / cmax):.1f}%;'
    f'background:{_hcolor(i)}" title="{XLAB[i]}: {c}"></div>'
    for i, c in enumerate(counts))
_hxlab = ''.join(f'<div>{lab}</div>' for lab in XLAB)

# ---------- pannello 06: occlusione plug prima/dopo ---------------------------
plugs = sorted(flow.get('occlusione_plug', []), key=lambda p: -p.get('ml_esclusi', 0))
if plugs:
    p0 = plugs[0]
    after = p0.get('model_raw_occlusa_cmH2O_s_L', raw)
    dpct = p0.get('delta_raw_pct', 0.0)
    ml = p0.get('ml_esclusi', 0.0)
    mx = max(raw, after) or 1.0
    hb = max(6.0, 100.0 * raw / mx)
    ha = max(6.0, 100.0 * after / mx)
    extra = ''
    if len(plugs) > 1:
        extra = ('<p class="note" style="margin-top:10px">altri ' + str(len(plugs) - 1)
                 + ' plug: ' + ' · '.join(
                     f"{_esc(p.get('zona', p.get('plug')))} −{p.get('ml_esclusi')} ml "
                     f"(ΔRaw {p.get('delta_raw_pct'):+}%)" for p in plugs[1:4]) + '</p>')
    plug_panel = f"""
    <div class="plugwrap">
      <div class="pba">
        <div class="pcap">Raw globale</div>
        <div class="pbars">
          <div class="pcol"><div class="pbar" style="height:{hb:.0f}%"></div><div class="ptag">prima</div></div>
          <div class="pcol"><div class="pbar" style="height:{ha:.0f}%"></div><div class="ptag">dopo</div></div>
        </div>
        <div class="pnum">{raw} → {after} · <b>{dpct:+}%</b></div>
      </div>
      <div class="pcallout">
        <div class="pbig">−{ml} ml</div>
        <div class="note" style="margin:0">territorio de-ventilato a valle del plug
          (<b>{_esc(p0.get('zona', p0.get('plug', '')))}</b>) — invisibile nella Raw
          globale, visibile nella mappa regionale.</div>
      </div>
    </div>{extra}
    <p class="note" style="margin-top:8px">Il punto onesto è il contrasto: la Raw globale
       non si muove, ma un territorio si perde. Osservazione del modello, non claim clinico.</p>"""
else:
    plug_panel = ('<p class="note">Nessun candidato mucus plug in questo caso: '
                  'niente esperimento di occlusione da mostrare.</p>')

# Pedley on/off per la tile di sensibilita'
pedley_on = next((s['model_raw'] for s in sens if s['parametro'] == 'pedley' and s['valore']), raw)
pedley_off = next((s['model_raw'] for s in sens if s['parametro'] == 'pedley' and not s['valore']), '?')

html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>__NAME__ — flusso simulato (esplorativo)</title>
<script>__PLOTLY__</script>
<style>
  .viz-root {{ color-scheme: light;
    --surface-1:#fcfcfb; --page:#f9f9f7; --text-primary:#0b0b0b;
    --text-secondary:#52514e; --muted:#898781; --grid:#e1e0d9;
    --border:rgba(11,11,11,.10); --series-1:#2a78d6; --orange:#eb6834; --neutral:#9a9891; }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark;
      --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff;
      --text-secondary:#c3c2b7; --muted:#898781; --grid:#2c2c2a;
      --border:rgba(255,255,255,.10); --series-1:#3987e5; --orange:#f0793f; --neutral:#8a8880; }} }}
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
  .cap {{ font-size:10.5px; color:var(--muted); margin-top:8px }}
  .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:16px }}
  @media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr }} }}
  .legend {{ display:flex; flex-wrap:wrap; gap:16px; font-size:12px;
    color:var(--text-secondary); margin-top:8px }}
  .legend span {{ display:inline-flex; align-items:center; gap:6px }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block }}
  #tree3d {{ width:100%; height:600px }}
  /* 01 barre per lobo */
  .lrow {{ display:flex; align-items:center; gap:9px; margin:5px 0; font-size:12px }}
  .lrow .llab {{ width:78px; text-align:right; color:var(--text-secondary); flex:none }}
  .lrow .ltrack {{ flex:1; background:var(--grid); border-radius:5px; height:16px; overflow:hidden }}
  .lrow .lfill {{ height:100%; border-radius:5px }}
  .lrow .lval {{ width:40px; color:var(--text-secondary); flex:none; font-variant-numeric:tabular-nums }}
  .lhatch {{ background-image:repeating-linear-gradient(45deg,var(--neutral),var(--neutral) 3px,transparent 3px,transparent 6px);
    border:1px solid var(--neutral) }}
  /* 03 barra misura/modello */
  .split {{ display:flex; height:30px; border-radius:7px; overflow:hidden; border:1px solid var(--border); margin-top:4px }}
  .split .seg {{ display:flex; align-items:center; justify-content:center; font-size:11.5px; font-weight:600; color:#fff }}
  /* 05 istogramma */
  .hist {{ display:flex; align-items:flex-end; gap:4px; height:130px; padding-top:6px }}
  .hist .hbar {{ flex:1; border-radius:4px 4px 0 0 }}
  .xlab {{ display:flex; gap:4px; margin-top:3px }}
  .xlab div {{ flex:1; text-align:center; font-size:8.5px; color:var(--muted) }}
  /* 06 occlusione */
  .plugwrap {{ display:flex; gap:18px; align-items:flex-end; flex-wrap:wrap }}
  .pba {{ text-align:center }}
  .pcap {{ font-size:10px; color:var(--muted); margin-bottom:4px }}
  .pbars {{ display:flex; gap:8px; align-items:flex-end; height:70px }}
  .pcol {{ width:34px; display:flex; flex-direction:column; justify-content:flex-end; height:100% }}
  .pbar {{ background:var(--series-1); border-radius:5px 5px 0 0 }}
  .ptag {{ font-size:9px; color:var(--muted); margin-top:3px }}
  .pnum {{ font-size:11px; color:var(--text-secondary); margin-top:5px }}
  .pcallout {{ flex:1; min-width:180px; background:color-mix(in srgb,var(--orange) 10%,var(--surface-1));
    border:1px solid var(--orange); border-radius:9px; padding:11px 13px }}
  .pbig {{ font-size:22px; font-weight:700; color:var(--orange) }}
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
    <div class="v">{n_meas} <span class="u">/ {n_imp}</span></div>
    <div class="u">{fm}% della dissipazione da misura</div></div>
  <div class="tile"><div class="k">Eterogeneità ventilazione (CV)</div>
    <div class="v">{cv}</div><div class="u">confronto tra casi/tempi, non assoluto</div></div>
  <div class="tile"><div class="k">Sensibilità Raw (Pedley on/off)</div>
    <div class="v">{pedley_on} <span class="u">/ {pedley_off}</span></div>
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
<div class="grid2">
  <div class="card">
    <h2>Dove cade la pressione — per lobo</h2>
    <p class="note">Quota % della dissipazione totale, vista anatomica. Nel soggetto
       normale il grosso sta nelle vie centrali; i lobi seguono.</p>
    {lobe_bars}
    <p class="cap">*Periferia = completamento oltre le foglie (imputato), tratteggiata perché non è un lobo misurato.</p>
  </div>
  <div class="card">
    <h2>Quanto è misura, quanto è modello</h2>
    <p class="note">Quota della dissipazione che poggia su diametri realmente misurati
       (half-max) contro quelli imputati per territorio.</p>
    <div class="split">
      <div class="seg" style="width:{fm}%;background:{SEQ[5]}">{fm:.0f}% misurato</div>
      <div class="seg" style="width:{fi}%;background:var(--neutral);background-image:repeating-linear-gradient(45deg,rgba(255,255,255,.25),rgba(255,255,255,.25) 3px,transparent 3px,transparent 7px)">{fi:.0f}% imputato</div>
    </div>
    <div class="legend">
      <span><span class="sw" style="background:{SEQ[5]}"></span> da calibro half-max ({n_meas} rami)</span>
      <span><span class="sw lhatch"></span> imputato per territorio ({n_imp} rami)</span>
    </div>
    <p class="cap">~metà del risultato dipende da un'assunzione: migliorare la segmentazione sposta la barra a sinistra.</p>
  </div>
</div>
<div class="grid2">
  <div class="card">
    <h2>Eterogeneità della ventilazione</h2>
    <p class="note">La distribuzione che il singolo CV nasconde: {len(SV)} territori,
       ventilazione specifica su scala log. Blu = ipo (&lt;1), arancio = iper (&gt;1).</p>
    <div class="hist">{_hbars}</div>
    <div class="xlab">{_hxlab}</div>
    <div class="legend" style="margin-top:10px">
      <span><span class="sw" style="background:{DIV_LO}"></span> ipo-ventilato (&lt;1)</span>
      <span><span class="sw" style="background:{DIV_HI}"></span> iper-ventilato (&gt;1)</span>
    </div>
    <p class="cap">La coda a destra (pochi territori "iper") è ciò che gonfia il CV; resta
       influenzata da imputazione/pruning — leggere come confronto, non in assoluto.</p>
  </div>
  <div class="card">
    <h2>Esperimento di occlusione (candidati plug)</h2>
    {plug_panel}
  </div>
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
</script></body></html>"""

html = html.replace('__TRACES__', json.dumps(traces))
name = os.environ.get('AIRWAYLAB_CASE', 'caso')
html = html.replace('__NAME__', name)
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets', 'plotly.min.js'), encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/flow_viz.html', 'w', encoding='utf-8').write(html)
print(f'out/flow_viz.html ({len(html)//1024//1024} MB)')
