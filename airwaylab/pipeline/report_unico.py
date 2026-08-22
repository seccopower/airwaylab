"""Report UNICO a schede: raccoglie in un solo file HTML tutte le viste prodotte
(mappa quantitativa + discordanza + arterie/vene + flusso simulato).

Ogni scheda e' la pagina gia' generata, incapsulata in un iframe isolato (niente
collisioni di id/JS/stili tra le viste). Plotly e' memorizzato UNA sola volta e
reiniettato in ogni iframe a runtime: il file resta della dimensione della somma
dei contenuti, senza duplicare la libreria. Gli iframe si costruiscono in modo
pigro alla prima apertura della scheda (avvio rapido).

Le schede esplorative (flusso) restano etichettate SIMULAZIONE dalla pagina
stessa; il V/Q e' escluso per scelta (in pausa).

Run nel work dir dopo le viste. Output: out/report_unico.html.
"""
import os

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'assets')
PL = open(os.path.join(ASSETS, 'plotly.min.js'), encoding='utf-8').read()
TOKEN = '@@AIRWAYLAB_PLOTLY@@'

# (titolo, file, nota) — inclusa solo se il file esiste
TABS = [
    ('Mappa quantitativa', 'out/report_main.html'),
    ('Discordanza aereo-vascolare', 'out/dual_viz.html'),
    ('Mappa strutturale multi-asse', 'out/morphomap.html'),
    ('Arterie / vene', 'out/av_viz.html'),
    ('Flusso simulato', 'out/flow_viz.html'),
]


def _esc_js(s):
    """Rende una stringa sicura dentro un template-literal JS (backtick)."""
    return (s.replace('\\', '\\\\').replace('`', '\\`')
            .replace('${', '\\${').replace('</script>', '<\\/script>'))


present = [(t, p) for t, p in TABS if os.path.exists(p)]
if not present:
    print('report_unico: nessuna vista trovata — salto')
    raise SystemExit(0)

docs = []
for _, p in present:
    html = open(p, encoding='utf-8').read().replace(PL, TOKEN)
    docs.append(_esc_js(html))

buttons = ''.join(
    f'<button class="tab{" active" if i == 0 else ""}" data-i="{i}">{t}</button>'
    for i, (t, _) in enumerate(present))
docs_js = ',\n'.join('`' + d + '`' for d in docs)
name = os.environ.get('AIRWAYLAB_CASE', 'caso')

html = f"""<!DOCTYPE html><html lang="it"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{name} — report AirwayLab</title>
<style>
  :root {{ color-scheme: light dark; }}
  * {{ box-sizing:border-box }} html,body {{ margin:0; height:100% }}
  body {{ font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
    background:#f9f9f7; color:#0b0b0b; display:flex; flex-direction:column; height:100vh }}
  @media (prefers-color-scheme: dark) {{ body {{ background:#0d0d0d; color:#fff }} }}
  header {{ padding:12px 20px 0 }}
  h1 {{ font-size:17px; font-weight:650; margin:0 }}
  .subt {{ font-size:12px; color:#898781; margin:2px 0 8px }}
  .tabs {{ display:flex; gap:4px; flex-wrap:wrap; border-bottom:1px solid rgba(128,128,128,.25); padding:0 12px }}
  .tab {{ font:inherit; font-size:13px; font-weight:550; cursor:pointer; border:none;
    background:none; color:#52514e; padding:9px 14px; border-bottom:2px solid transparent; margin-bottom:-1px }}
  @media (prefers-color-scheme: dark) {{ .tab {{ color:#c3c2b7 }} }}
  .tab.active {{ color:#2a78d6; border-bottom-color:#2a78d6 }}
  .frame-wrap {{ flex:1; min-height:0 }}
  iframe {{ width:100%; height:100%; border:none; display:block }}
</style></head><body>
<header><h1>AirwayLab · {name}</h1>
  <div class="subt">Report unico — mappa quantitativa e analisi esplorative. Le schede esplorative sono marcate nella pagina stessa.</div>
</header>
<div class="tabs">{buttons}</div>
<div class="frame-wrap"><iframe id="frame" title="vista"></iframe></div>
<script>
const PL = `{_esc_js(PL)}`;
const DOCS = [
{docs_js}
];
const built = {{}};
const frame = document.getElementById('frame');
function show(i) {{
  document.querySelectorAll('.tab').forEach(b => b.classList.toggle('active', +b.dataset.i === i));
  if (!built[i]) {{
    const doc = DOCS[i].split('{TOKEN}').join(PL);
    built[i] = URL.createObjectURL(new Blob([doc], {{type: 'text/html'}}));
  }}
  frame.src = built[i];
}}
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => show(+b.dataset.i)));
show(0);
</script></body></html>"""

open('out/report_unico.html', 'w', encoding='utf-8').write(html)
print(f"report_unico.html ({len(html)//1024//1024} MB) — schede: "
      + ', '.join(t for t, _ in present))
