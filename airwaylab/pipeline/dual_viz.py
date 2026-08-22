"""Visualizzazione della discordanza aereo-vascolare (ESPLORATIVA) — pagina
standalone. Due assi separati:
  - albero 3D colorato dal BA ratio (bronco/arteria): asse DILATAZIONE
    (rosso = via piu' larga dell'arteria, sospetto rimodellamento/bronchiectasia);
  - barre per lobo: frazione di mismatch (asse OCCLUSIONE) e frazione BA>1
    (asse dilatazione), regionalizzate — si confronta lobo con lobo.

Run nel work dir dopo dual.py + viz.py. Output: out/dual_viz.html (autonomo).
"""
import json
import os

import numpy as np

data = json.load(open('out/map_data.json'))
reg = json.load(open('out/discordance_regional.json')) if os.path.exists('out/discordance_regional.json') else {'per_lobo': {}}
per_lobo = reg.get('per_lobo', {})
cloud = reg.get('cloud', {'x': [], 'y': [], 'z': [], 'delta': []})
coronal_png = data.get('meta', {}).get('dual_png')
branches = data['branches']

# colormap del delta (coerente con la mappa coronale): blu(basso) -> grigio -> rosso(alto)
DELTA_SCALE = [[0, '#2a78d6'], [0.5, '#f0efec'], [1.0, '#e34948']]
DELTA_LO, DELTA_HI = -10.0, 60.0

# diverging attorno a BA=1: blu (via stretta vs arteria) .. grigio .. rosso
DIV_LO, DIV_MID, DIV_HI = '#2a78d6', '#9a9891', '#eb6834'


def lerp(a, b, t):
    ah = [int(a[i:i+2], 16) for i in (1, 3, 5)]
    bh = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    return '#%02x%02x%02x' % tuple(int(ah[k] + (bh[k]-ah[k]) * t) for k in range(3))


def ba_color(ba):
    # 0.6 -> blu, 1.0 -> grigio, 1.6 -> rosso (clip)
    x = float(np.clip((ba - 1.0) / 0.6, -1, 1))
    return lerp(DIV_MID, DIV_HI, x) if x >= 0 else lerp(DIV_MID, DIV_LO, -x)


traces = []
for b in branches:
    ba = b.get('ba')
    col = ba_color(ba) if ba else '#c9c8c2'
    w = 2.0 + (3.5 if ba else 0.0)
    ht = (f"{b.get('name') or b['id']}<br>BA ratio {ba}" if ba
          else f"{b.get('name') or b['id']}<br>(non accoppiato)")
    traces.append({'type': 'scatter3d', 'mode': 'lines',
                   'x': b['x'], 'y': b['y'], 'z': b['z'],
                   'line': {'color': col, 'width': float(w)},
                   'hovertemplate': ht + '<extra></extra>', 'showlegend': False})

# barre per lobo: mismatch (occlusione) e BA>1 (dilatazione)
lobi = [k for k in per_lobo if k != 'CENTRAL']
lobi.sort(key=lambda k: -(per_lobo[k].get('mismatch_frac') or 0))
mm = [round(100 * (per_lobo[k].get('mismatch_frac') or 0), 1) for k in lobi]
bg = [round(100 * (per_lobo[k].get('ba_gt1_frac') or 0), 1) for k in lobi]

rows = ''.join(
    f"<tr><td>{k}</td><td>{per_lobo[k].get('delta_med_mm')}</td>"
    f"<td>{per_lobo[k].get('mismatch_frac')}</td>"
    f"<td>{per_lobo[k].get('ba_med')}</td>"
    f"<td>{per_lobo[k].get('ba_gt1_frac')}</td>"
    f"<td><b>{per_lobo[k].get('prevalenza')}</b></td></tr>"
    for k in lobi)

# ancore D/S che ruotano con l'anatomia (riferimento destra/sinistra; LPS: +x = S)
def _ds_anchors(c):
    if not c.get('x'):
        return None
    x, y, z = c['x'], c['y'], c['z']
    ym = float(np.median(y)); zm = float(np.median(z))
    return {'type': 'scatter3d', 'mode': 'text', 'x': [max(x) + 8, min(x) - 8],
            'y': [ym, ym], 'z': [zm, zm], 'text': ['S', 'D'],
            'textfont': {'size': 20, 'color': '#0b0b0b'},
            'hoverinfo': 'skip', 'showlegend': False}


# nuvola 3D del delta
cloud_trace = {
    'type': 'scatter3d', 'mode': 'markers',
    'x': cloud['x'], 'y': cloud['y'], 'z': cloud['z'],
    'marker': {'size': 2, 'color': cloud['delta'], 'colorscale': DELTA_SCALE,
               'cmin': DELTA_LO, 'cmax': DELTA_HI, 'opacity': 0.35,
               'colorbar': {'title': {'text': 'delta (mm)'}, 'thickness': 10}},
    'hovertemplate': 'delta %{marker.color:.1f} mm<extra></extra>',
    'showlegend': False,
}
coronal_card = (
    f'<div class="card"><h2>Mappa coronale del delta</h2>'
    f'<p class="note">Proiezione coronale mediana del delta (blu vicino alle vie → '
    f'rosso lontano dalle vie a parità di vasi). Contorno polmone e albero aereo per '
    f'riferimento. Leggere i gradienti e le asimmetrie, non i valori assoluti.</p>'
    f'<img src="{coronal_png}" style="max-width:100%;border-radius:8px"></div>'
    if coronal_png else '')

name = os.environ.get('AIRWAYLAB_CASE', 'caso')
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — discordanza aereo-vascolare (esplorativa)</title>
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
  .card {{ background:var(--surface-1); border:1px solid var(--border);
    border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  table {{ border-collapse:collapse; width:100%; font-size:13px }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--border) }}
  th {{ color:var(--text-secondary); font-weight:600 }}
  #tree3d,#cloud3d {{ width:100%; height:560px }} #bars {{ width:100%; height:300px }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin-top:8px }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block; vertical-align:middle; margin-right:5px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Discordanza aereo-vascolare</h1>
<p class="sub">{name} · due assi separati: BA ratio (via vs arteria) e mismatch regionale (parenchima vascolarizzato senza via aerea)</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Indici ESPLORATIVI, non validati.</b> La mappa voxel ha un bias di visibilità bronchi/vasi:
  si legge il confronto <b>tra lobi</b>, non i valori assoluti.</div>
<div class="card">
  <h2>Albero colorato dal BA ratio (asse dilatazione)</h2>
  <p class="note">Rosso = bronco più largo dell'arteria satellite (BA&gt;1, sospetto rimodellamento/bronchiectasia); blu = più stretto; grigio ≈ 1. I rami non accoppiati sono grigio chiaro e sottili.</p>
  <div id="tree3d"></div>
  <div class="legend">
    <span><span class="sw" style="background:{DIV_LO}"></span>BA &lt; 1</span>
    <span><span class="sw" style="background:{DIV_MID}"></span>BA ≈ 1</span>
    <span><span class="sw" style="background:{DIV_HI}"></span>BA &gt; 1</span>
  </div>
</div>
<div class="card">
  <h2>Nuvola 3D del delta (ruotabile)</h2>
  <p class="note">Ogni punto è parenchima polmonare, colorato dal delta = distanza dalla via aerea − distanza dal vaso. Rosso = vascolarizzato ma lontano da una via aerea (asse occlusione). Trascina per ruotare.</p>
  <div id="cloud3d"></div>
</div>
{coronal_card}
<div class="card">
  <h2>Per lobo: occlusione vs BA&gt;1</h2>
  <p class="note">Blu = % di parenchima in mismatch (asse occlusione/de-ventilazione). Arancio = % di bronchi con BA&gt;1 (dilatazione bronchiale <b>o</b> assottigliamento arterioso/pruning: il rapporto non li distingue). Regionalizzato: confronta i lobi tra loro.</p>
  <div id="bars"></div>
</div>
<div class="card">
  <h2>Tabella regionale</h2>
  <table><thead><tr><th>Lobo</th><th>delta med (mm)</th><th>mismatch</th><th>BA med</th><th>BA&gt;1</th><th>prevalenza</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
Plotly.newPlot('tree3d', __TRACES__, {{
  margin:{{l:0,r:0,t:0,b:0}}, showlegend:false,
  scene:{{ xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}},
          aspectmode:'data', camera:{{eye:{{x:0,y:-1.7,z:0.15}},up:{{x:0,y:0,z:1}}}},
          bgcolor:'rgba(0,0,0,0)' }}, paper_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
Plotly.newPlot('cloud3d', __CLOUDS__, {{
  margin:{{l:0,r:0,t:0,b:0}}, showlegend:false,
  scene:{{ xaxis:{{visible:false}}, yaxis:{{visible:false}}, zaxis:{{visible:false}},
          aspectmode:'data', camera:{{eye:{{x:0,y:-1.7,z:0.15}},up:{{x:0,y:0,z:1}}}},
          bgcolor:'rgba(0,0,0,0)' }}, paper_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
const L={json.dumps(lobi)}, MM={json.dumps(mm)}, BG={json.dumps(bg)};
Plotly.newPlot('bars', [
  {{type:'bar', name:'mismatch %', x:L, y:MM, marker:{{color:'#2a78d6'}}}},
  {{type:'bar', name:'BA>1 %', x:L, y:BG, marker:{{color:'#eb6834'}}}}
], {{ barmode:'group', margin:{{l:44,r:10,t:8,b:30}},
  legend:{{orientation:'h',y:1.15}}, yaxis:{{title:{{text:'%'}},color:'#898781'}},
  xaxis:{{color:'#898781'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""

_anch = _ds_anchors(cloud)
_cloud_traces = [cloud_trace] + ([_anch] if _anch else [])
# ancore D/S anche sull'albero BA
_tanch = _ds_anchors({'x': [v for b in branches for v in b['x']],
                      'y': [v for b in branches for v in b['y']],
                      'z': [v for b in branches for v in b['z']]})
if _tanch:
    traces.append(_tanch)
html = html.replace('__TRACES__', json.dumps(traces))
html = html.replace('__CLOUDS__', json.dumps(_cloud_traces))
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets', 'plotly.min.js'), encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/dual_viz.html', 'w', encoding='utf-8').write(html)
print(f'out/dual_viz.html ({len(html)//1024//1024} MB)')
