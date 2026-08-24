"""Scheda "Sintesi quantitativa": raccoglie i descrittori numerici (Pi10, tapering,
morfometria dell'albero, parenchima, pruning vascolare, densitometria) in un'unica
pagina a riquadri. Legge i JSON gia' prodotti in out/; mostra solo cio' che esiste.

Run nel work dir dopo gli step di misura. Output: out/summary.html.
"""
import json
import os


def _load(name):
    p = os.path.join('out', name)
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except (ValueError, OSError):
            return None
    return None


pi10 = _load('pi10.json') or {}
tap = _load('tapering.json') or {}
tree = _load('treestats.json') or {}
par = _load('parenchyma.json') or {}
vg = _load('vascular_gradient.json') or {}
lm = _load('lung_metrics.json') or {}
vm = _load('vessel_metrics.json') or {}

name = os.environ.get('AIRWAYLAB_CASE', 'caso')


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def tile(label, value, unit='', note='', hero=False):
    """Un riquadro-metrica; ritorna '' se il valore e' None."""
    if value is None:
        return ''
    u = f'<span class="u">{_esc(unit)}</span>' if unit else ''
    n = f'<div class="n">{_esc(note)}</div>' if note else ''
    cls = 'tile hero' if hero else 'tile'
    return (f'<div class="{cls}"><div class="k">{_esc(label)}</div>'
            f'<div class="v">{_esc(value)}{u}</div>{n}</div>')


def section(title, note, tiles):
    body = ''.join(t for t in tiles if t)
    if not body:
        return ''
    nt = f'<p class="note">{_esc(note)}</p>' if note else ''
    return f'<div class="card"><h2>{_esc(title)}</h2>{nt}<div class="tiles">{body}</div></div>'


# --- Vie aeree ---
hg = par.get('histogram', {})
airways = section(
    'Vie aeree — rimodellamento e struttura',
    'Descrittori per soggetto. Confrontabili solo a parita di protocollo.',
    [
        tile('Pi10', pi10.get('pi10'), ' mm', f"√WA a Pi=10 · n={pi10.get('n')} · R²={pi10.get('r2')}", hero=True),
        tile('Tapering (figlio/genitore)', tap.get('taper_ratio_med'), '',
             'sano ~0.79; →1 = rastremazione persa'),
        tile('Gradiente calibro', tap.get('taper_rate_pct_per_cm'), ' %/cm',
             'riduzione del calibro per cm'),
        tile('AFD', (tree.get('fractal') or {}).get('afd'), '',
             'dimensione frattale dell albero (robusta al taglio)'),
        tile('Rami / terminali',
             (f"{(tree.get('counts') or {}).get('n_branches')} / {(tree.get('counts') or {}).get('n_terminals')}"
              if tree.get('counts') else None), '',
             'dipende dalla profondita di segmentazione'),
    ])

# --- Parenchima ---
het = par.get('heterogeneity', {})
clu = par.get('laa_clusters', {})
parench = section(
    'Parenchima',
    'Oltre la densita media: disomogeneita e organizzazione della bassa attenuazione.',
    [
        tile('MLD', lm.get('mld_hu'), ' HU', 'densita media polmone'),
        tile('LAA-950', lm.get('laa950_pct'), ' %', '<-950 HU (enfisema/iperinflazione)'),
        tile('Perc15', lm.get('perc15_hu'), ' HU', '15° percentile'),
        tile('Eterogeneita (mosaic)', het.get('het_sd_hu'), ' HU',
             'SD regionale della densita'),
        tile('Cluster LAA — D', clu.get('D'), '',
             f"esponente Mishima · maggiore {clu.get('largest_frac')}"),
        tile('Skewness / Kurtosi',
             (f"{hg.get('skewness')} / {hg.get('kurtosis')}" if hg.get('skewness') is not None else None),
             '', 'forma dell istogramma HU'),
    ])

# --- Vasi ---
vessels = section(
    'Vasi (TC senza contrasto → volume di vaso)',
    'Volume vascolare e pruning periferico dei piccoli vasi.',
    [
        tile('TBV', vm.get('tbv_ml'), ' ml', 'volume vascolare totale'),
        tile('BV5', (round(100 * vm['bv5_frac'], 0) if vm.get('bv5_frac') is not None else None), ' %',
             'quota nei vasi < 5 mm²'),
        tile('Pruning ratio', vg.get('pruning_ratio'), '',
             'densita piccoli vasi periferia/centro (<1 = pruning)'),
        tile('Gradiente pruning', vg.get('gradient_per_mm'), '/mm',
             'densita piccoli vasi vs distanza dalla pleura'),
    ])

cards = ''.join([airways, parench, vessels]) or '<p class="note">Nessun descrittore disponibile.</p>'

html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — sintesi quantitativa</title>
<style>
  .viz-root {{ color-scheme: light dark; --surface-1:#fcfcfb; --page:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --border:rgba(11,11,11,.10); --accent:#0c6f79; }}
  @media (prefers-color-scheme: dark) {{ .viz-root {{
    --surface-1:#161a1a; --page:#0d0d0d; --text-primary:#fff;
    --text-secondary:#c3c2b7; --muted:#898781; --border:rgba(255,255,255,.12); --accent:#3bb4c0; }} }}
  * {{ box-sizing:border-box }} body {{ margin:0 }}
  .viz-root {{ font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    background:var(--page); color:var(--text-primary); min-height:100vh; padding:22px }}
  .wrap {{ max-width:1000px; margin:0 auto }}
  h1 {{ font-size:19px; font-weight:650; margin:0 0 3px }}
  .sub {{ color:var(--text-secondary); font-size:12.5px; margin:0 0 18px }}
  .card {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px;
    padding:16px 18px; margin-bottom:14px }}
  .card h2 {{ font-size:13px; font-weight:600; margin:0 0 2px; letter-spacing:.01em }}
  .note {{ font-size:11.5px; color:var(--muted); margin:0 0 12px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:11px }}
  .tile {{ background:var(--page); border:1px solid var(--border); border-radius:10px; padding:11px 13px }}
  .tile.hero {{ border-color:var(--accent) }}
  .tile .k {{ font-size:11px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:22px; font-weight:680; font-variant-numeric:tabular-nums }}
  .tile.hero .v {{ color:var(--accent) }}
  .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .tile .n {{ font-size:10.5px; color:var(--muted); margin-top:5px; line-height:1.35 }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Sintesi quantitativa · {_esc(name)}</h1>
<p class="sub">Descrittori numerici del report. Ogni valore è misurato e ha i suoi limiti (vedi dizionario). Confronti solo a parità di protocollo.</p>
{cards}
</div></div></body></html>"""

open('out/summary.html', 'w', encoding='utf-8').write(html)
print('out/summary.html — sintesi quantitativa')
