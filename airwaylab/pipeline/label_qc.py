"""Guardia di completezza dell'etichettatura lobare (nucleo puro).

L'etichettatura anatomica (labels.py) e' "best effort": su alcune anatomie non
propaga le etichette lobari a tutto l'albero (es. un intero polmone resta senza
lobo). In quei casi i risultati PER-LOBO (mappa multi-asse, discordanza, vascolare
per lobo, incertezza) rappresentano solo una parte del polmone e NON vanno mostrati
come completi. Questa guardia rileva l'incompletezza e produce il banner d'avviso.

Puro: nessun I/O. Testato in tests/test_label_qc.py.
"""
LOBES = ('RUL', 'RML', 'RLL', 'LUL', 'LING', 'LLL')


def labeling_status(present, expected=LOBES):
    """Da un iterabile di lobi presenti, ritorna lo stato di completezza.
    {complete, n_present, n_expected, missing}."""
    pres = set(present) & set(expected)
    missing = [lb for lb in expected if lb not in pres]
    return {
        'complete': len(missing) == 0,
        'n_present': len(pres),
        'n_expected': len(expected),
        'missing': missing,
    }


def labeling_banner_html(present, expected=LOBES):
    """Banner rosso HTML se l'etichettatura e' incompleta, altrimenti stringa vuota."""
    st = labeling_status(present, expected)
    if st['complete']:
        return ''
    miss = ', '.join(st['missing'])
    return (
        '<div style="background:#7a1f1f;color:#fff;border:2px solid #c0392b;'
        'border-radius:8px;padding:10px 14px;margin-bottom:16px;font-size:13px">'
        '⛔ <b>ETICHETTATURA LOBARE INCOMPLETA</b> — riconosciuti '
        f'{st["n_present"]}/{st["n_expected"]} lobi; <b>mancano: {miss}</b>. '
        'I risultati per-lobo qui sotto rappresentano <b>solo una parte del polmone</b> '
        'e NON sono affidabili come mappa completa (probabile fallimento di labels.py '
        'su questa anatomia — la segmentazione può comunque essere corretta: verificare il QC).'
        '</div>')
