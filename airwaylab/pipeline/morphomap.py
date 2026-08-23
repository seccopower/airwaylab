"""Mappa strutturale multi-asse (ESPLORATIVA) — compute + pagina standalone.

Incrocia, per lobo/territorio, DUE assi tenuti separati:
  - conduttanza bronchiale  = flusso modellato che raggiunge il territorio
                              (flow.json, modello di flusso esplorativo)
  - bassa attenuazione inspiratoria = LAA (%voxel < -950 HU) e f_tessuto,
                              campionati dalla TC dentro ogni territorio.
Indice derivato: media della LAA territoriale PESATA per le quote di conduttanza.
NON e' ventilazione, perfusione, spazio morto ne distruzione (vedi morphomap_core
per i limiti; su singola inspiratoria la LAA non equivale a distruzione, specie
nell'asma). Descrittore strutturale esplorativo.

Dipendenze nel work dir: territory_labels_ds.nii.gz, territory_index.json,
flow.json, ct_iso.nii.gz.
Si salta se manca il flusso o le territorie. Output: out/morphomap.json + morphomap.html.

Run nel work dir dopo territory.py + flow.py (+ vasculature.py, opz.).
"""
import json
import os

import nibabel as nib
import numpy as np

from label_qc import labeling_banner_html, labeling_status
from plausibility_core import lobe_plausibility, plausibility_banner_html
from morphomap_core import (
    SCHEMA_VERSION,
    aggregate_lobes,
    classify_lobe,
    voxelwise_destruction,
)

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
laa, ftis = voxelwise_destruction(ct, lab.shape, laa_hu=LAA_HU)
del ct

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

for lb, s in per.items():
    s['prevalenza'] = classify_lobe(s['laa'], s['ds_share'])

lab_banner = labeling_banner_html(per.keys())
if lab_banner:
    print('  ATTENZIONE: etichettatura lobare incompleta —',
          labeling_status(per.keys())['missing'], 'mancanti')

# --- controllo di plausibilita' delle proporzioni lobari (completo != corretto) ---
# usa il volume lobare (conteggio voxel del territorio): intercetta le partizioni
# implausibili che la guardia di presenza non puo' vedere.
vol_by_lobe = {lb: s['n_vox'] for lb, s in per.items()}
plaus = lobe_plausibility(vol_by_lobe)
plaus_banner = plausibility_banner_html(vol_by_lobe)
if not plaus['ok']:
    print('  ATTENZIONE: proporzioni lobari implausibili (controllo plausibilita\'):')
    for f in plaus['flags']:
        print(f"    [{f['severity']}] {f['msg']}")

json.dump({'schema_version': SCHEMA_VERSION, 'status': 'exploratory',
           'labeling': labeling_status(per.keys()),
           'plausibilita_lobare': plaus,
           'metrica': 'media della frazione LAA inspiratoria (<-950 HU) pesata per le '
                      'quote di flusso del modello resistivo: w_i=q_i/sum(q), I=sum(w_i*LAA_i). '
                      'I pesi sono allocazione relativa del modello nello scenario '
                      'prespecificato, NON misure regionali di ventilazione. NON e\' '
                      'ventilazione, perfusione, spazio morto ne distruzione',
           'laa_threshold_hu': LAA_HU, 'globale': glob, 'per_lobo': per},
          open('out/morphomap.json', 'w'), indent=1)

print('Mappa strutturale multi-asse per lobo:')
print(f"  LAA inspiratoria pesata dal modello resistivo (globale): {100 * glob['cond_to_destroyed']:.1f}%")
for lb, s in sorted(per.items(), key=lambda x: -(x[1]['ds_share'] or 0)):
    print(f"  {lb:5s} q-flusso {100 * s['cond_frac']:4.1f}%  LAA {100 * s['laa']:4.1f}%  "
          f"f_tes {s['f_tissue']:.3f}  quota-indice {100 * (s['ds_share'] or 0):4.1f}%  {s['prevalenza']}")

# --- pagina ---
lobi = [lb for lb in per]
lobi.sort(key=lambda lb: -(per[lb]['ds_share'] or 0))
ds_share = [round(100 * (per[lb]['ds_share'] or 0), 1) for lb in lobi]
cond = [round(100 * per[lb]['cond_frac'], 1) for lb in lobi]
laa_pct = [round(100 * per[lb]['laa'], 1) for lb in lobi]


def _bar_col(laa_v):
    # colore per DISTRUZIONE reale del lobo (LAA %), non per la quota relativa:
    # su un polmone sano le quote sommano comunque a 100% ma restano grigie.
    return '#c0392b' if laa_v >= 40 else '#eb6834' if laa_v >= 25 else '#9a9891'


ds_cols = [_bar_col(v) for v in laa_pct]


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


rows = ''.join(
    f"<tr><td>{lb}</td><td>{per[lb]['cond_q']}</td><td>{round(100 * per[lb]['cond_frac'], 1)}</td>"
    f"<td>{round(100 * per[lb]['laa'], 1)}</td><td>{per[lb]['f_tissue']}</td>"
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
<p class="sub">{name} · quota di flusso simulato (modello resistivo) <b>×</b> bassa attenuazione inspiratoria (LAA −950 HU), per lobo — assi tenuti separati</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Descrittore strutturale, NON funzionale.</b> Indice = <b>media della frazione LAA inspiratoria pesata dalle quote di flusso del modello resistivo</b>: w<sub>i</sub> = q<sub>i</sub>/Σq,&nbsp; I = Σ w<sub>i</sub>·LAA<sub>i</sub>. I pesi w<sub>i</sub> rappresentano <b>l'allocazione relativa ottenuta dal modello nello scenario prespecificato</b> (flusso totale fisso, Pedley attivo) — <b>non</b> sono misure regionali di ventilazione. <b>Non</b> è ventilazione, perfusione, spazio morto né distruzione. La bassa attenuazione inspiratoria può riflettere <b>iperinflazione o altre cause non enfisematose</b>; <b>l'air-trapping non è identificabile da questa acquisizione e richiede dati espiratori</b>. Dipende da soglia HU e volume inspiratorio (confronto solo a parità di protocollo). Le quote q vengono dal modello e sono in gran parte da diametri imputati. Esplorativo.</div>
{lab_banner}
{plaus_banner}
<div class="tiles">
  <div class="tile hero"><div class="k">LAA inspiratoria pesata dal modello resistivo</div><div class="v">{round(head/100, 3)}</div><div class="u">frazione = {head} punti percentuali di LAA · I = Σ (q<sub>i</sub>/Σq)·LAA<sub>i</sub> · NON una % di flusso/ventilazione</div></div>
  <div class="tile"><div class="k">Bassa attenuazione inspiratoria (LAA −950)</div><div class="v">{round(100 * glob['laa_lung'], 1)}<span class="u">%</span></div><div class="u">media polmone</div></div>
  <div class="tile"><div class="k">f_tessuto medio polmone</div><div class="v">{glob['f_tissue_lung']}</div><div class="u">frazione non-aria</div></div>
</div>
<div class="card">
  <h2>Quota per lobo dell'indice (flusso simulato × LAA)</h2>
  <p class="note">Ogni barra = quota del lobo sul totale di (quota di flusso simulato × bassa attenuazione). Pesa insieme <b>quanto flusso simulato</b> riceve il lobo e <b>quanta bassa attenuazione</b> ha. <b>Colore = bassa attenuazione del lobo</b> (LAA): rosso ≥40%, arancio ≥25%, grigio sotto — su un polmone con LAA bassa le barre restano grigie anche se una quota è alta (le quote sommano comunque a 100%). Leggi l'altezza <i>col</i> valore globale in testa.</p>
  <div id="ds"></div>
</div>
<div class="card">
  <h2>I due assi, affiancati per lobo</h2>
  <p class="note">Blu = quota di flusso simulato del lobo (allocazione del modello resistivo). Rosso = bassa attenuazione del lobo (LAA). Le due barre sono descrittori distinti e non indipendenti; nessuna delle due è una misura funzionale.</p>
  <div id="grp"></div>
  <div class="legend"><span><span class="sw" style="background:#2a78d6"></span>quota flusso simulato (%)</span>
    <span><span class="sw" style="background:#c0392b"></span>bassa attenuazione LAA (%)</span></div>
</div>
<div class="card">
  <h2>Tabella regionale</h2>
  <table><thead><tr><th>Lobo</th><th>flusso sim. (ml/s)</th><th>quota flusso sim. %</th><th>LAA %</th><th>f_tessuto</th><th>quota indice %</th><th>etichetta (soglie espl.)</th></tr></thead>
  <tbody>{rows}</tbody></table>
</div>
</div></div>
<script>
const L={json.dumps(lobi)}, DS={json.dumps(ds_share)}, DSC={json.dumps(ds_cols)};
const COND={json.dumps(cond)}, LAA={json.dumps(laa_pct)};
Plotly.newPlot('ds', [{{type:'bar', x:L, y:DS, marker:{{color:DSC}},
  hovertemplate:'%{{x}} · %{{y}}% del carico totale<extra></extra>'}}], {{
  margin:{{l:44,r:10,t:8,b:30}},
  yaxis:{{title:{{text:'quota del carico (%)'}},color:'#898781'}}, xaxis:{{color:'#898781'}},
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
