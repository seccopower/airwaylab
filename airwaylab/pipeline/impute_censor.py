"""Test di censura sull'imputazione dei diametri (validazione tecnica interna).

Prende i rami REALMENTE MISURATI, li censura uno alla volta (leave-one-out) e
lascia che il modello li re-imputi dal diametro del genitore (reale) e dalla
frazione di territorio; confronta imputato vs misurato. Da' una base EMPIRICA
all'errore di imputazione (finora assunto). Confronta anche l'imputazione
CONDIZIONATA DAL TERRITORIO con quella SIMMETRICA (Murray), per giudicare se il
condizionamento territoriale riduce davvero l'errore (ablazione #5).

LIMITE (dichiararlo): il test misura l'errore solo dove la misura ERA possibile
(rami piu' prossimali/grandi); i rami realmente imputati sono piu' periferici, dove
l'errore puo' essere maggiore. E' un limite inferiore realistico, non l'errore sui
rami periferici.

Run nel work dir dopo measure/territory. Output: out/impute_censor.json + .html.
"""
import json
import os

from flow_model import PARAMS, build_topology
from impute_core import error_stats, imputed_diameter

tree = json.load(open('out/tree_measured.json'))
terr = json.load(open('out/territories.json')) if os.path.exists('out/territories.json') else {}
topo = build_topology(tree, terr)
by_id, kids, parent, Sub = topo['by_id'], topo['kids'], topo['parent'], topo['Sub']
NEXP = PARAMS['murray_exp']
DFLOOR = PARAMS['d_num_floor_mm']

# leave-one-out sui rami misurati con genitore misurato
rec = []   # (gen, measured, imp_terr, imp_sym)
for b in tree['branches']:
    if not b.get('d_mean') or b.get('qc') not in (None, 'ok'):
        continue
    p = parent.get(b['id'])
    if p is None or not p.get('d_mean'):
        continue
    d_par = float(p['d_mean'])
    sub_p = Sub.get(p['id'], 0.0)
    frac_terr = (Sub.get(b['id'], 0.0) / sub_p) if sub_p > 0 else 1.0
    sibs = kids(p)
    frac_sym = 1.0 / max(1, len(sibs))
    rec.append((int(b.get('gen', 0)), float(b['d_mean']),
                imputed_diameter(d_par, frac_terr, NEXP, DFLOOR),
                imputed_diameter(d_par, frac_sym, NEXP, DFLOOR)))

if not rec:
    print('impute_censor: nessun ramo misurato con genitore misurato — salto')
    raise SystemExit(0)

terr_pairs = [(m, it) for _, m, it, _ in rec]
sym_pairs = [(m, isy) for _, m, _, isy in rec]
overall = {'territorio': error_stats(terr_pairs), 'simmetrico': error_stats(sym_pairs)}

# per generazione (solo bin con n>=5)
gens = sorted(set(g for g, *_ in rec))
per_gen = {}
for g in gens:
    tp = [(m, it) for gg, m, it, _ in rec if gg == g]
    if len(tp) >= 5:
        per_gen[g] = error_stats(tp)

out = {
    'status': 'exploratory',
    'nota': 'leave-one-out sui rami MISURATI (con genitore misurato): errore relativo '
            '(imputato-misurato)/misurato. Misura l\'imputazione dove la misura era '
            'possibile (rami prossimali); i rami realmente imputati sono piu\' periferici, '
            'errore potenzialmente maggiore. NEXP=%.1f, dfloor=%.1f mm.' % (NEXP, DFLOOR),
    'n_rami_testati': len(rec),
    'overall': overall,
    'per_generazione_territorio': per_gen,
}
json.dump(out, open('out/impute_censor.json', 'w'), indent=1)

print(f"Censura imputazione — {len(rec)} rami misurati testati (leave-one-out):")
for k in ('territorio', 'simmetrico'):
    s = overall[k]
    print(f"  {k:11s}: err.ass.mediano {s['errore_assoluto_mediano']} · bias {s['bias_mediano']} · "
          f"entro20% {s['entro_20pct']} · p5-p95 [{s['p5']},{s['p95']}]")
best = 'territorio' if overall['territorio']['errore_assoluto_mediano'] <= overall['simmetrico']['errore_assoluto_mediano'] else 'simmetrico'
print(f"  -> imputazione migliore: {best}")

# --- pagina ---
gl = list(per_gen.keys())
e_terr = [round(100 * per_gen[g]['errore_assoluto_mediano'], 1) for g in gl]
name = os.environ.get('AIRWAYLAB_CASE', 'caso')
ot, os_ = overall['territorio'], overall['simmetrico']
html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<title>{name} — censura imputazione diametri</title>
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
  .wrap {{ max-width:1000px; margin:0 auto }}
  h1 {{ font-size:20px; font-weight:650; margin:0 0 4px }}
  .sub {{ color:var(--text-secondary); font-size:13px; margin:0 0 16px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin-bottom:16px }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:14px 16px }}
  .tile .k {{ font-size:12px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:22px; font-weight:700 }} .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:10px; padding:16px; margin-bottom:16px }}
  .card h2 {{ font-size:14px; font-weight:600; margin:0 0 2px }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:0 0 10px }}
  #bars {{ width:100%; height:320px }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Censura dell'imputazione dei diametri</h1>
<p class="sub">{name} · {len(rec)} rami misurati censurati uno alla volta (leave-one-out) · errore relativo imputato vs misurato</p>
<div style="background:#7a1f1f;color:#fff;border-radius:8px;padding:8px 14px;margin-bottom:16px;font-size:13px">
  ⚠ <b>Validazione tecnica interna.</b> Misura l'errore di imputazione <b>dove la misura era possibile</b> (rami prossimali): i rami realmente imputati sono più periferici, dove l'errore può essere maggiore. È un limite inferiore realistico.</div>
<div class="tiles">
  <div class="tile"><div class="k">Errore assoluto mediano (territorio)</div><div class="v">{round(100*ot['errore_assoluto_mediano'],1)}<span class="u">%</span></div><div class="u">entro ±20%: {round(100*ot['entro_20pct'])}% dei rami</div></div>
  <div class="tile"><div class="k">Bias mediano (territorio)</div><div class="v">{round(100*ot['bias_mediano'],1)}<span class="u">%</span></div><div class="u">+ = sovrastima · p5–p95 {round(100*ot['p5'])}…{round(100*ot['p95'])}%</div></div>
  <div class="tile"><div class="k">Simmetrico (Murray), err.ass.mediano</div><div class="v">{round(100*os_['errore_assoluto_mediano'],1)}<span class="u">%</span></div><div class="u">entro ±20%: {round(100*os_['entro_20pct'])}% · confronto col territorio</div></div>
</div>
<div class="card">
  <h2>Errore assoluto mediano per generazione (imputazione territoriale)</h2>
  <p class="note">Come cresce l'errore man mano che si scende nell'albero. Le generazioni profonde sono quelle dove in pratica si imputa di più.</p>
  <div id="bars"></div>
</div>
</div></div>
<script>
const G={json.dumps(['gen '+str(g) for g in gl])}, E={json.dumps(e_terr)};
Plotly.newPlot('bars', [{{type:'bar', x:G, y:E, marker:{{color:'#2a78d6'}},
  hovertemplate:'%{{x}} · err %{{y}}%<extra></extra>'}}], {{
  margin:{{l:44,r:10,t:8,b:40}}, yaxis:{{title:{{text:'errore assoluto mediano (%)'}},color:'#898781'}},
  xaxis:{{color:'#898781'}}, paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)' }},
  {{displayModeBar:false, responsive:true}});
</script></body></html>"""
pl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets', 'plotly.min.js'),
          encoding='utf-8').read()
open('out/impute_censor.html', 'w', encoding='utf-8').write(html.replace('__PLOTLY__', pl))
print(f"out/impute_censor.html")
