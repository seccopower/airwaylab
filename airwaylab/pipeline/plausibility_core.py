"""Controllo di plausibilita' delle proporzioni lobari (nucleo puro).

Perche' serve. La guardia di completezza (label_qc) verifica che i 6 lobi siano
PRESENTI, ma "completo" non vuol dire "corretto": labels.py puo' etichettare tutti
e sei i lobi e nondimeno ripartire male l'albero (caso reale casoDAS: 6/6 lobi
presenti, ma il lobo superiore destro riceveva lo 0.2% del volume e il medio il
22% — una chiara inversione/errore di partizione che la guardia di presenza NON
puo' intercettare). Questo controllo aggiunge una rete di sicurezza sulle
PROPORZIONI: segnala partizioni anatomicamente implausibili.

Filosofia (coerente col resto del progetto: esplorativo, non sovradichiarare).
Le soglie sono LARGHE e ancorate a fatti anatomici quasi universali e robusti alla
malattia, non a intervalli di normalita' stretti:
  - ogni lobo e' una quota non banale del polmone;
  - il lobo MEDIO e' di norma il piu' piccolo del polmone destro;
  - la lingula e' di norma piu' piccola del resto del lobo superiore sinistro;
  - nessun intero emitorace e' quasi assente.
Un flag NON significa "sbagliato": significa "verifica questa ripartizione contro
la segmentazione". Anatomie molto distorte (enfisema severo, resezioni) possono far
scattare un flag legittimamente. Il segnale forte e' un flag su un albero che per il
resto sembra plausibile.

Puro: nessun I/O. Testato in tests/test_plausibility_core.py.
"""
LOBES = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')
RIGHT = ('RUL', 'RML', 'RLL')
LEFT = ('LUL', 'LING', 'LLL')


def lobe_plausibility(vol_by_lobe, tiny_frac=0.03, side_lo=0.30):
    """Valuta la plausibilita' delle proporzioni lobari.

    vol_by_lobe: {aid_lobo -> volume}   (ml o conteggio voxel: contano solo i
      rapporti, quindi qualsiasi unita' coerente va bene). I lobi assenti o con
      volume None/<=0 sono ignorati nelle quote (la completezza la copre label_qc).

    Ritorna un dict:
      {'ok': bool, 'flags': [{'code','severity','msg'}], 'fracs': {aid: frazione}}
    severity in {'alto','medio'}. 'ok' e' True se non ci sono flag."""
    vol = {lb: float(v) for lb, v in vol_by_lobe.items()
           if lb in LOBES and v is not None and float(v) > 0}
    total = sum(vol.values())
    flags = []
    if total <= 0:
        return {'ok': False, 'flags': [{
            'code': 'no_volume', 'severity': 'alto',
            'msg': 'nessun volume lobare disponibile per il controllo di plausibilita\''}],
            'fracs': {}}

    fr = {lb: v / total for lb, v in vol.items()}

    # A. lobo con quota trascurabile: quasi sempre un errore di partizione.
    for lb in LOBES:
        if lb in fr and fr[lb] < tiny_frac:
            flags.append({
                'code': 'tiny_lobe', 'severity': 'alto',
                'msg': f'{lb} riceve solo il {100 * fr[lb]:.1f}% del volume polmonare '
                       f'(soglia {100 * tiny_frac:.0f}%): probabile errore di partizione '
                       f'(ramo lobare mancato o assegnato altrove)'})

    # B. emitorace quasi assente.
    for side, name in ((RIGHT, 'destro'), (LEFT, 'sinistro')):
        present = [lb for lb in side if lb in fr]
        if present:
            sfr = sum(fr[lb] for lb in present)
            if sfr < side_lo:
                flags.append({
                    'code': 'side_imbalance', 'severity': 'alto',
                    'msg': f'il polmone {name} riceve solo il {100 * sfr:.0f}% del volume '
                           f'totale (soglia {100 * side_lo:.0f}%): possibile intero lato '
                           f'mal assegnato'})

    # C. il lobo medio dovrebbe essere il piu' piccolo a destra.
    if all(lb in fr for lb in RIGHT):
        smallest_r = min(RIGHT, key=lambda lb: fr[lb])
        if smallest_r != 'RML':
            sev = 'alto' if fr['RML'] == max(fr[lb] for lb in RIGHT) else 'medio'
            flags.append({
                'code': 'rml_not_smallest', 'severity': sev,
                'msg': f'il lobo medio (RML {100 * fr["RML"]:.1f}%) non e\' il piu\' piccolo '
                       f'del polmone destro (piu\' piccolo: {smallest_r} '
                       f'{100 * fr[smallest_r]:.1f}%): il medio e\' di norma il minore — '
                       f'possibile scambio/errore di etichetta a destra'})

    # D. la lingula dovrebbe essere piu' piccola del resto del lobo superiore sx.
    if 'LING' in fr and 'LUL' in fr and fr['LING'] > fr['LUL']:
        flags.append({
            'code': 'ling_gt_lul', 'severity': 'medio',
            'msg': f'la lingula (LING {100 * fr["LING"]:.1f}%) risulta piu\' grande del '
                   f'lobo superiore sinistro (LUL {100 * fr["LUL"]:.1f}%): atipico — '
                   f'verificare la separazione LUL/lingula'})

    fracs = {lb: round(fr[lb], 3) for lb in fr}
    return {'ok': len(flags) == 0, 'flags': flags, 'fracs': fracs}


def plausibility_banner_html(vol_by_lobe, **kw):
    """Banner ambra HTML se le proporzioni lobari sono implausibili, altrimenti ''.

    Distinto dal banner rosso di INCOMPLETEZZA (label_qc): qui i lobi ci sono tutti
    ma le proporzioni non tornano. E' un avviso di verifica, non un errore fatale."""
    res = lobe_plausibility(vol_by_lobe, **kw)
    if res['ok']:
        return ''
    items = ''.join(f'<li>{_esc(f["msg"])}</li>' for f in res['flags'])
    hard = any(f['severity'] == 'alto' for f in res['flags'])
    head = ('proporzioni lobari IMPLAUSIBILI' if hard
            else 'proporzioni lobari da verificare')
    return (
        '<div style="background:#7a5b12;color:#fff;border:2px solid #d0a020;'
        'border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:13px">'
        f'⚠ <b>CONTROLLO DI PLAUSIBILITA\': {head}</b> — l\'etichettatura e\' '
        'completa (6/6 lobi) ma le proporzioni volumetriche non sono anatomicamente '
        f'attese:<ul style="margin:6px 0 4px 18px;padding:0">{items}</ul>'
        'Un flag significa <b>verificare la ripartizione contro la segmentazione</b>, '
        'non necessariamente che sia errata: anatomie molto distorte possono farlo '
        'scattare. Esplorativo.</div>')


def _esc(s):
    return (str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;'))
