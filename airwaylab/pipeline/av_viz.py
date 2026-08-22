"""Visualizzazione arteria/vena (ESPLORATIVA) — pagina standalone.
  - nuvole 3D separate: arterie (rosso) e vene (blu), con ancore fisse D/S che
    ruotano con l'anatomia (riferimento destra/sinistra);
  - QC visivo della separazione A/V (su TC senza contrasto va verificata);
  - tabella per lobo: volumi della maschera vascolare + rapporto A/V.
Volumi di MASCHERA, non ematici. Il pruning/BV5 e' ritirato (vedi vasculature.py).

Run nel work dir dopo vasculature.py + viz.py. Output: out/av_viz.html.
"""
import json
import os

import numpy as np

av = json.load(open('out/vascular_av.json'))
per = av.get('per_lobo', {})
ca, cv = av['cloud_art'], av['cloud_vein']


def ds_anchors(cloud):
    """Ancore testo D (destra) / S (sinistra) alle estremita' in x (LPS: +x =
    sinistra). Ruotano con la scena: riferimento sempre presente."""
    if not cloud['x']:
        return None
    x = cloud['x']; y = cloud['y']; z = cloud['z']
    ymid = float(np.median(y)); zmid = float(np.median(z))
    return {'type': 'scatter3d', 'mode': 'text',
            'x': [max(x) + 8, min(x) - 8], 'y': [ymid, ymid], 'z': [zmid, zmid],
            'text': ['S', 'D'], 'textfont': {'size': 20, 'color': '#0b0b0b'},
            'hoverinfo': 'skip', 'showlegend': False}


traces = [
    {'type': 'scatter3d', 'mode': 'markers', 'name': 'arterie',
     'x': ca['x'], 'y': ca['y'], 'z': ca['z'],
     'marker': {'size': 1.8, 'color': '#d1443f', 'opacity': 0.45},
     'hovertemplate': 'arteria<extra></extra>'},
    {'type': 'scatter3d', 'mode': 'markers', 'name': 'vene',
     'x': cv['x'], 'y': cv['y'], 'z': cv['z'],
     'marker': {'size': 1.8, 'color': '#2a78d6', 'opacity': 0.35},
     'hovertemplate': 'vena<extra></extra>'},
]
anch = ds_anchors(ca) or ds_anchors(cv)
if anch:
    traces.append(anch)

lobi = [k for k in per if k != 'CENTRAL']
lobi.sort(key=lambda k: -(per[k].get('art_ml') or 0))
rows = ''.join(
    f"<tr><td>{k}</td><td>{per[k].get('art_ml')}</td><td>{per[k].get('vein_ml')}</td>"
    f"<td>{per[k].get('av_ratio')}</td></tr>"
    for k in lobi)

name = os.environ.get('AIRWAYLAB_CASE', 'caso')
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — arterie/vene (esplorativa)</title>
<script>__PLOTLY__</script>
<style>
  .viz-root {{ color-scheme: light; --surface-1:#fcfcfb; --page:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --border:rgba(11,11,11,.10); }}
  @media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{
    --surface-1:#1a1a19; --page:#0d0d0d; --text-primary:#fff;
    --text-secondary:#c3c2b7; --muted:#898781; --border:rgba(255,255,255,.10); }} }}
  * {{ box-sizing:border-box }} body {{ margin:0 }}
  .viz-root {{ font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    background:var(--page); color:var(--text-primary); min-height:100vh; padding:24px }}
  .wrap {{ max-width:1100px; margin:0 auto }}
  h1 {{ font-size:20px; font-weight:650; margin:0 0 4px }}
  .sub {{ color:var(--text-secondary); font-size:13px; margin:0 0 16px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin-bottom:16px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px }}
  .tile .k {{ font-size:12px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:22px; font-weight:650 }} .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  table {{ border-collapse:collapse; width:100%; font-size:13px }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--border) }}
  th {{ color:var(--text-secondary); font-weight:600 }}
  #av3d {{ width:100%; height:600px }} #bars {{ width:100%; height:300px }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin-top:8px }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block; vertical-align:middle; margin-right:5px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Arterie e vene polmonari</h1>
<p class="sub">{name} · maschere DL separate (TotalSegmentator) · le ancore <b>D</b>/<b>S</b> ruotano con l'anatomia</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Esplorativo. Volumi della MASCHERA vascolare, non volume ematico né perfusione</b> (TC senza contrasto). I volumi <b>dipendono dalla segmentazione e dalla profondità raggiunta</b>, e il segmentatore può avere <b>sensibilità diversa per arterie e vene</b>. Separazione A/V da verificare: <b>guarda prima questa vista come QC</b>. Il pruning / BV5 (piccoli vasi) è stato <b>ritirato</b>: la stima voxelwise da EDT non è valida (serve un metodo di calibro segmentale/scale-space; sarebbe una misura nuova, non una calibrazione).</div>
<div class="tiles">
  <div class="tile"><div class="k">Volume maschera arteriosa</div><div class="v">{av['arterie_ml']} <span class="u">ml</span></div></div>
  <div class="tile"><div class="k">Volume maschera venosa</div><div class="v">{av['vene_ml']} <span class="u">ml</span></div></div>
  <div class="tile"><div class="k">Rapporto A/V</div><div class="v">{av['av_ratio']}</div><div class="u">volumi maschera</div></div>
</div>
<div class="card">
  <h2>Arterie (rosso) e vene (blu) — ruotabile, con riferimento D/S</h2>
  <p class="note">QC della separazione: le arterie devono correre coi bronchi dagli ili, le vene interdigitare. Trascina per ruotare; le ancore D/S restano attaccate ai lati dell'anatomia.</p>
  <div id="av3d"></div>
  <div class="legend"><span><span class="sw" style="background:#d1443f"></span>arterie</span>
    <span><span class="sw" style="background:#2a78d6"></span>vene</span></div>
</div>
<div class="card">
  <h2>Tabella regionale</h2>
  <p class="note">Volumi della maschera vascolare per lobo (non ematici). Il pruning / BV5 è stato ritirato in attesa di un metodo di calibro validato.</p>
  <table><thead><tr><th>Lobo</th><th>arterie (ml)</th><th>vene (ml)</th><th>A/V</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
Plotly.newPlot('av3d', __TRACES__, {{
  margin:{{l:0,r:0,t:0,b:0}}, showlegend:false,
  scene:{{ xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}},
          aspectmode:'data', camera:{{eye:{{x:0,y:-1.7,z:0.15}},up:{{x:0,y:0,z:1}}}},
          bgcolor:'rgba(0,0,0,0)' }}, paper_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""

html = html.replace('__TRACES__', json.dumps(traces))
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets', 'plotly.min.js'), encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/av_viz.html', 'w', encoding='utf-8').write(html)
print(f'out/av_viz.html ({len(html)//1024//1024} MB)')
