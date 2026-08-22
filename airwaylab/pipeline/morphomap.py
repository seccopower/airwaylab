"""Mappa strutturale multi-asse (ESPLORATIVA) — compute + pagina standalone.

Incrocia, per lobo/territorio, DUE assi tenuti separati:
  - conduttanza bronchiale  = flusso modellato che raggiunge il territorio
                              (flow.json, modello di flusso esplorativo)
  - distruzione parenchimale = enfisema (LAA, %voxel < -950 HU) e f_tessuto,
                              campionati dalla TC dentro ogni territorio.
Indice derivato: quota di conduttanza diretta a parenchima distrutto
(= candidato spazio morto strutturale). NON e' ventilazione ne' V/Q misurati.

Dipendenze nel work dir: territory_labels_ds.nii.gz, territory_index.json,
flow.json, ct_iso.nii.gz. Opzionale: vascular_av.json (BV5 per lobo, solo tabella).
Si salta se manca il flusso o le territorie. Output: out/morphomap.json + morphomap.html.

Run nel work dir dopo territory.py + flow.py (+ vasculature.py, opz.).
"""
import json
import os

import nibabel as nib
import numpy as np

from morphomap_core import SCHEMA_VERSION, aggregate_lobes, classify_lobe

FLOW = 'out/flow.json'
LABELS = 'out/territory_labels_ds.nii.gz'
INDEX = 'out/territory_index.json'
CT = 'out/ct_iso.nii.gz'
LAA_HU = -950.0

if not (os.path.exists(FLOW) and os.path.exists(LABELS) and os.path.exists(INDEX)):
    print('morphomap: mancano flow.json / territorie — salto')
    raise SystemExit(0)

flow = json.load(open(FLOW)).get('per_ramo', {})
ti = {int(k): v for k, v in json.load(open(INDEX)).items()}
lab = np.asarray(nib.load(LABELS).dataobj).astype(np.int32)

# --- HU a risoluzione iso, accumulato per cella ds (blocchi 3x3x3), frugale ---
# La griglia ds (territorie) e' esattamente 3x piu' grossa della iso, stessa
# origine: ogni voxel ds = blocco 3x3x3 di voxel iso. Accumuliamo LAA e f_tessuto
# su tutti i sotto-voxel senza materializzare il volume iso intero.
ct = np.asarray(nib.load(CT).dataobj).astype(np.float32)
S = lab.shape[0]
ctp = np.pad(ct, [(0, S * 3 - s) for s in ct.shape], constant_values=-1000.0)
del ct
laa = np.zeros(lab.shape, np.float32)     # frazione voxel < LAA_HU per cella ds
ftis = np.zeros(lab.shape, np.float32)    # f_tessuto medio per cella ds
for a in range(3):
    for b in range(3):
        for c in range(3):
            sub = ctp[a::3, b::3, c::3][:S, :S, :S]
            laa += (sub < LAA_HU)
            ftis += np.clip(1.0 + sub / 1000.0, 0.0, 1.0)
del ctp
laa /= 27.0
ftis /= 27.0

# --- per territorio ---
territories = []
for L, meta in ti.items():
    m = lab == L
    n = int(m.sum())
    if n == 0:
        continue
    q = flow.get(meta['branch_id'], {}).get('model_q_ml_s', 0.0) or 0.0
    territories.append({
        'lobe': meta['lobe'], 'q': float(q), 'n': n,
        'laa': float(laa[m].mean()), 'f_tissue': float(ftis[m].mean()),
    })

if not territories:
    print('morphomap: nessun territorio con voxel — salto')
    raise SystemExit(0)

per, glob = aggregate_lobes(territories)

# BV5 per lobo (solo per la tabella) se disponibile
bv5 = {}
if os.path.exists('out/vascular_av.json'):
    for lb, s in json.load(open('out/vascular_av.json')).get('per_lobo', {}).items():
        if s.get('bv5_ml') is not None:
            bv5[lb] = s['bv5_ml']

for lb, s in per.items():
    s['prevalenza'] = classify_lobe(s['laa'], s['ds_share'])
    s['bv5_ml'] = bv5.get(lb)

json.dump({'schema_version': SCHEMA_VERSION, 'status': 'exploratory',
           'laa_threshold_hu': LAA_HU, 'globale': glob, 'per_lobo': per},
          open('out/morphomap.json', 'w'), indent=1)

print('Mappa strutturale multi-asse per lobo:')
print(f"  quota conduttanza -> parenchima distrutto (globale): {100 * glob['cond_to_destroyed']:.1f}%")
for lb, s in sorted(per.items(), key=lambda x: -(x[1]['ds_share'] or 0)):
    print(f"  {lb:5s} cond {100 * s['cond_frac']:4.1f}%  LAA {100 * s['laa']:4.1f}%  "
          f"f_tes {s['f_tissue']:.3f}  quota-mismatch {100 * (s['ds_share'] or 0):4.1f}%  {s['prevalenza']}")

# --- pagina ---
lobi = [lb for lb in per]
lobi.sort(key=lambda lb: -(per[lb]['ds_share'] or 0))
ds_share = [round(100 * (per[lb]['ds_share'] or 0), 1) for lb in lobi]
cond = [round(100 * per[lb]['cond_frac'], 1) for lb in lobi]
laa_pct = [round(100 * per[lb]['laa'], 1) for lb in lobi]


def _bar_col(v):
    return '#c0392b' if v >= 30 else '#eb6834' if v >= 15 else '#9a9891'


ds_cols = [_bar_col(v) for v in ds_share]


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


rows = ''.join(
    f"<tr><td>{lb}</td><td>{per[lb]['cond_q']}</td><td>{round(100 * per[lb]['cond_frac'], 1)}</td>"
    f"<td>{round(100 * per[lb]['laa'], 1)}</td><td>{per[lb]['f_tissue']}</td>"
    f"<td>{per[lb]['bv5_ml'] if per[lb]['bv5_ml'] is not None else '—'}</td>"
    f"<td>{round(100 * (per[lb]['ds_share'] or 0), 1)}</td>"
    f"<td>{_esc(per[lb]['prevalenza'])}</td></tr>"
    for lb in lobi)

name = os.environ.get('AIRWAYLAB_CASE', 'caso')
head = round(100 * glob['cond_to_destroyed'], 1)
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — mappa strutturale multi-asse (esplorativa)</title>
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
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:16px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px }}
  .tile.hero {{ border-color:#c0392b }}
  .tile .k {{ font-size:12px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:26px; font-weight:700 }} .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .tile.hero .v {{ color:#c0392b }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  table {{ border-collapse:collapse; width:100%; font-size:13px }}
  th,td {{ text-align:left; padding:6px 10px; border-bottom:1px solid var(--border); white-space:nowrap }}
  th {{ color:var(--text-secondary); font-weight:600 }}
  td:last-child {{ white-space:normal; color:var(--text-secondary) }}
  #ds {{ width:100%; height:320px }} #grp {{ width:100%; height:340px }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin-top:6px }}
  .sw {{ width:14px; height:14px; border-radius:4px; display:inline-block; vertical-align:middle; margin-right:5px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Mappa strutturale multi-asse</h1>
<p class="sub">{name} · conduttanza bronchiale (modello di flusso) <b>×</b> distruzione parenchimale (enfisema TC), per lobo — assi tenuti separati</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>NON è ventilazione né V/Q misurati.</b> È conduttanza <i>modellata</i> (simulazione di flusso) incrociata con la distruzione <i>strutturale</i> (LAA &lt;−950&nbsp;HU dalla TC). L'indice segnala <b>rischio strutturale di spazio morto</b>: bronchi pervi che ventilano parenchima distrutto. Confronto tra lobi, esplorativo.</div>
<div class="tiles">
  <div class="tile hero"><div class="k">Conduttanza → parenchima distrutto</div><div class="v">{head}<span class="u">%</span></div><div class="u">flusso modellato verso polmone &lt;−950 HU</div></div>
  <div class="tile"><div class="k">Enfisema polmone (LAA −950)</div><div class="v">{round(100 * glob['laa_lung'], 1)}<span class="u">%</span></div></div>
  <div class="tile"><div class="k">f_tessuto medio polmone</div><div class="v">{glob['f_tissue_lung']}</div><div class="u">frazione non-aria</div></div>
</div>
<div class="card">
  <h2>Quota del mismatch per lobo — dove va la conduttanza sprecata</h2>
  <p class="note">Ogni barra = quota del lobo sul totale della conduttanza diretta a parenchima distrutto (q × enfisema). Pesa insieme <b>quanto flusso</b> riceve il lobo e <b>quanto è distrutto</b>: il lobo più alto è il primo candidato a spazio morto.</p>
  <div id="ds"></div>
</div>
<div class="card">
  <h2>I due assi, affiancati per lobo</h2>
  <p class="note">Blu = quota di conduttanza (flusso modellato). Rosso = enfisema del lobo (LAA). Il rischio di spazio morto nasce dove le due barre sono <b>entrambe</b> alte: molto flusso verso molto parenchima distrutto.</p>
  <div id="grp"></div>
  <div class="legend"><span><span class="sw" style="background:#2a78d6"></span>conduttanza (quota %)</span>
    <span><span class="sw" style="background:#c0392b"></span>enfisema LAA (%)</span></div>
</div>
<div class="card">
  <h2>Tabella regionale</h2>
  <table><thead><tr><th>Lobo</th><th>conduttanza (ml/s)</th><th>quota cond. %</th><th>LAA %</th><th>f_tessuto</th><th>BV5 (ml)</th><th>quota mismatch %</th><th>pattern</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
const L={json.dumps(lobi)}, DS={json.dumps(ds_share)}, DSC={json.dumps(ds_cols)};
const COND={json.dumps(cond)}, LAA={json.dumps(laa_pct)};
Plotly.newPlot('ds', [{{type:'bar', x:L, y:DS, marker:{{color:DSC}},
  hovertemplate:'%{{x}} · %{{y}}% del mismatch totale<extra></extra>'}}], {{
  margin:{{l:44,r:10,t:8,b:30}},
  yaxis:{{title:{{text:'quota del mismatch (%)'}},color:'#898781'}}, xaxis:{{color:'#898781'}},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
Plotly.newPlot('grp', [
  {{type:'bar', name:'conduttanza', x:L, y:COND, marker:{{color:'#2a78d6'}},
    hovertemplate:'%{{x}} · conduttanza %{{y}}%<extra></extra>'}},
  {{type:'bar', name:'enfisema LAA', x:L, y:LAA, marker:{{color:'#c0392b'}},
    hovertemplate:'%{{x}} · LAA %{{y}}%<extra></extra>'}}
], {{ barmode:'group', showlegend:false,
  margin:{{l:44,r:10,t:8,b:30}}, yaxis:{{title:{{text:'%'}},color:'#898781'}}, xaxis:{{color:'#898781'}},
  paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""

pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       '..', 'assets', 'plotly.min.js'), encoding='utf-8').read()
html = html.replace('__PLOTLY__', pl)
open('out/morphomap.html', 'w', encoding='utf-8').write(html)
print(f'out/morphomap.html ({len(html) // 1024 // 1024} MB)')
