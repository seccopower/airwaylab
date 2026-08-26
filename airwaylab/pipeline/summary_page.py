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


def _load_bodycomp():
    p = os.environ.get('AIRWAYLAB_BODYCOMP') or os.path.join('out', 'bodycomp.json')
    if p and os.path.exists(p):
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
bc = _load_bodycomp() or {}

name = os.environ.get('AIRWAYLAB_CASE', 'caso')

# provenienza: la prendo dal primo descrittore che ce l'ha (tutti timbrano lo stesso
# blocco). Serve al footer per dichiarare backend + versione sotto ai numeri.
prov = {}
for _d in (pi10, tap, tree, par, vg):
    if isinstance(_d, dict) and _d.get('provenance'):
        prov = _d['provenance']
        break


def _esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def tile(label, value, unit='', note='', hero=False, warn=False):
    """Un riquadro-metrica; ritorna '' se il valore e' None."""
    if value is None:
        return ''
    u = f'<span class="u">{_esc(unit)}</span>' if unit else ''
    n = f'<div class="n">{_esc(note)}</div>' if note else ''
    cls = 'tile hero' if hero else ('tile warn' if warn else 'tile')
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
# nota Pi10: n, R², CI bootstrap e — se 10 mm cade fuori dai perimetri osservati —
# l'avviso di estrapolazione (in quel caso la tile diventa 'warn', non 'hero').
_ci = pi10.get('ci95') or {}
_ci_txt = (f" · CI95 {_ci['ci_lo']}–{_ci['ci_hi']}"
           if _ci.get('ci_lo') is not None else '')
_pi10_extrap = bool(pi10.get('extrapolation'))
_pi10_note = f"√WA a Pi=10 · n={pi10.get('n')} · R²={pi10.get('r2')}{_ci_txt}"
if _pi10_extrap:
    _pi10_note += ' · estrapolato oltre i perimetri osservati (cautela)'
airways = section(
    'Vie aeree — rimodellamento e struttura',
    'Descrittori per soggetto. Confrontabili solo a parita di protocollo.',
    [
        tile('Pi10', pi10.get('pi10'), ' mm', _pi10_note,
             hero=not _pi10_extrap, warn=_pi10_extrap),
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
        tile('Eterogeneità (SD regionale)', het.get('het_sd_hu'), ' HU',
             'disomogeneità regionale — non specifica per mosaic/air-trapping'),
        tile('Cluster LAA — D', clu.get('D'), '',
             f"esponente rango-dimensione · maggiore {clu.get('largest_frac')} · "
             f"non specifico per enfisema"),
        tile('Skewness / Kurtosi',
             (f"{hg.get('skewness')} / {hg.get('kurtosis')}" if hg.get('skewness') is not None else None),
             '', 'forma dell istogramma HU'),
    ])

# --- Vasi ---
vessels = section(
    'Strutture vascolari (TC senza contrasto)',
    'Volume di strutture dense candidate a vaso e loro distribuzione periferica. Proxy morfologici, non calibro vero.',
    [
        tile('TBV', vm.get('tbv_ml'), ' ml',
             'volume maschera strutture dense (candidate vaso)'),
        tile('Piccole strutture <5mm²',
             (round(100 * vm['bv5_frac'], 0) if vm.get('bv5_frac') is not None else None), ' %',
             'residuo di apertura r≈1.26 mm (proxy, non BV5 validato)'),
        tile('Pruning ratio', vg.get('pruning_ratio'), '',
             'densità piccole strutture periferia/centro (ridotto = meno in periferia; esplorativo)'),
        tile('Gradiente pruning', vg.get('gradient_per_mm'), '/mm',
             'densità piccole strutture vs distanza dalla pleura'),
    ])

# --- Biomarcatori opportunistici (opt-in) ---
bone = bc.get('bone', {})
mus = bc.get('muscle', {})
fat = bc.get('fat', {})
opportun = section(
    'Biomarcatori opportunistici — stesso torace',
    'Opt-in, dalle maschere di composizione corporea. Screening, non diagnosi.',
    [
        tile('Osso — HU vertebrale', bone.get('mean_hu'), ' HU',
             f"min {bone.get('min_hu')} · n {bone.get('n')} · esplorativo, non diagnostico"),
        tile('Muscolo', mus.get('muscle_ml'), ' ml',
             f"HU {mus.get('muscle_hu')} (bassa = infiltrazione adiposa)"),
        tile('Grasso interno/SAT', fat.get('internal_sat_ratio'), '',
             f"SAT {fat.get('sat_ml')} · interno tronco {fat.get('internal_fat_ml')} ml "
             f"(torso_fat, non VAT segmentato)"),
    ])

cards = ''.join([airways, parench, vessels, opportun]) or '<p class="note">Nessun descrittore disponibile.</p>'

# --- banner esplorativo (il caveat che il report NON deve cancellare) ---
banner = (
    '<div class="banner"><strong>Descrittori esplorativi, non endpoint validati.</strong> '
    'Ogni riquadro è una misura per soggetto con i suoi limiti (definizione, denominatori '
    'ed esclusioni nel dizionario dei parametri). Non sono biomarcatori clinici validati: '
    'servono per confronti <em>a parità di protocollo</em> — tipicamente pre/post dello '
    'stesso paziente, stesso backend e stessa versione — non come valori assoluti né come '
    'diagnosi.</div>')

# --- footer di provenienza: sotto ai numeri, da dove vengono ---
_bits = []
if prov.get('backend'):
    _bits.append(f"backend {_esc(prov['backend'])}")
if prov.get('airwaylab_version'):
    _bits.append(f"AirwayLab v{_esc(prov['airwaylab_version'])}")
if prov.get('iso_mm'):
    _bits.append(f"ricostruzione {_esc(prov['iso_mm'])} mm iso")
foot = (f'<p class="foot">Provenienza: {" · ".join(_bits)}. '
        f'Stato: esplorativo.</p>' if _bits
        else '<p class="foot">Provenienza non disponibile (seg_info assente).</p>')

html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(name)} — sintesi quantitativa</title>
<style>
  .viz-root {{ color-scheme: light dark; --surface-1:#fcfcfb; --page:#f9f9f7;
    --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
    --border:rgba(11,11,11,.10); --accent:#0c6f79; --warn:#a86a12; }}
  @media (prefers-color-scheme: dark) {{ .viz-root {{
    --surface-1:#161a1a; --page:#0d0d0d; --text-primary:#fff;
    --text-secondary:#c3c2b7; --muted:#898781; --border:rgba(255,255,255,.12);
    --accent:#3bb4c0; --warn:#d69a3e; }} }}
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
  .banner {{ background:color-mix(in srgb, var(--warn) 12%, var(--surface-1));
    border:1px solid var(--warn); border-left-width:3px; border-radius:10px;
    padding:11px 14px; margin:0 0 16px; font-size:12px; line-height:1.5;
    color:var(--text-secondary) }}
  .banner strong {{ color:var(--warn) }}
  .foot {{ font-size:11px; color:var(--muted); margin:16px 2px 0;
    border-top:1px solid var(--border); padding-top:10px }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:11px }}
  .tile {{ background:var(--page); border:1px solid var(--border); border-radius:10px; padding:11px 13px }}
  .tile.hero {{ border-color:var(--accent) }}
  .tile.warn {{ border-color:var(--warn) }} .tile.warn .v {{ color:var(--warn) }}
  .tile .k {{ font-size:11px; color:var(--text-secondary); margin-bottom:4px }}
  .tile .v {{ font-size:22px; font-weight:680; font-variant-numeric:tabular-nums }}
  .tile.hero .v {{ color:var(--accent) }}
  .tile .u {{ font-size:12px; color:var(--muted); font-weight:400 }}
  .tile .n {{ font-size:10.5px; color:var(--muted); margin-top:5px; line-height:1.35 }}
</style></head><body><div class="viz-root"><div class="wrap">
<h1>Sintesi quantitativa · {_esc(name)}</h1>
<p class="sub">Descrittori numerici del report. Ogni valore è misurato e ha i suoi limiti (vedi dizionario). Confronti solo a parità di protocollo.</p>
{banner}
{cards}
{foot}
</div></div></body></html>"""

open('out/summary.html', 'w', encoding='utf-8').write(html)
print('out/summary.html — sintesi quantitativa')
