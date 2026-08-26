"""Controllo di qualita' della segmentazione: leak e connettivita' (nucleo puro).

Perche' RIFATTO (review avversariale GPT + Gemini). La vecchia idea "leak = voxel
delle vie aeree FUORI dalla maschera polmonare" e' cieca proprio sulla malattia che
studiamo: in un polmone enfisematoso i leak vanno DENTRO il parenchima — bolle,
cisti, honeycombing — tutte dentro la maschera; e la penalita' sui "rami senza lume"
punisce il tracking corretto di un tappo di muco (asma). Servono metriche che vedano
i leak INTERNI e non premino ne' puniscano la cosa sbagliata.

Due famiglie, entrambe calcolabili SENZA ground-truth (girano su ogni paziente):

  1. RADIUS-EXPLOSION — un ramo distale marcatamente PIU' LARGO del suo genitore.
     Le vie aeree si assottigliano andando in periferia: un figlio piu' largo del
     padre e' spesso un leak in uno spazio non-aereo (cisti/bolla/esofago). Si misura sul
     diametro della MASCHERA (d_mask_eq), quindi si vede ANCHE dove non c'e' lume.
     Scala-indipendente e robusto alla numerazione delle generazioni.

  2. CONNETTIVITA' / ISOLE — un albero aereo e' UN solo componente connesso.
     Frammenti staccati (isole) sono spesso falsi positivi. Si contano i componenti e
     il volume che NON sta nel componente principale.

NON incluso in v1 (onestamente): il leak EXTRAPOLMONARE/esofageo. La maschera
polmonare di lung.py SOTTRAE le vie aeree per costruzione (air & ~dilate(airway)),
quindi "vie aeree fuori dal polmone" e' ~100% per definizione; e dilatare il polmone
non separa in modo pulito l'esofago dalla trachea/principali (extrapolmonari per
anatomia). Un leak extrapolmonare corretto richiede un vero INVILUPPO delle vie aeree
(polmone + corridoio tracheobronchiale centrale) e le ROI negative annotate
(esofago, bolle, vasi): e' il pezzo successivo, non un numero da inventare qui.

Precisione/recall vs riferimento manuale NON stanno qui: richiedono ground-truth e
appartengono all'harness di confronto tra backend, non al QC per-caso.

Puro: nessun I/O. Testato in tests/test_leak_qc_core.py.
"""

# vie aeree centrali legittimamente larghe: la transizione verso di loro non e' un leak
CENTRAL_AIDS = {'TRACHEA', 'RMB', 'LMB', 'BI'}


def radius_explosion(branches, ratio_hi=1.6, floor_mm=4.0):
    """Rami il cui diametro-maschera supera quello del genitore oltre `ratio_hi`.

    branches: [{id, aid, gen, d_mask, parent_d}] — d_mask/parent_d in mm (diametro
      equivalente dalla maschera; None se non disponibile). I rami centrali
      (CENTRAL_AIDS) sono esclusi (la loro transizione e' il punto piu' largo per
      anatomia). `floor_mm`: sotto questo diametro i rapporti sono rumorosi, si salta.
    Ritorna la lista dei rami sospetti, ordinata dal rapporto maggiore."""
    out = []
    for b in branches:
        if b.get('aid') in CENTRAL_AIDS:
            continue
        d = b.get('d_mask')
        pd = b.get('parent_d')
        if d is None or pd is None or pd <= 0 or d < floor_mm:
            continue
        ratio = d / pd
        if ratio >= ratio_hi:
            out.append({'id': b.get('id'), 'gen': b.get('gen'),
                        'd_mask': round(d, 1), 'parent_d': round(pd, 1),
                        'ratio': round(ratio, 2)})
    return sorted(out, key=lambda x: -x['ratio'])


def connectivity_flag(n_components, largest_frac, leaked_ml,
                      frac_lo=0.97, leaked_floor_ml=0.5):
    """Valuta la connettivita' della maschera. Un albero = un componente.
    Segnala se una frazione/volume non banale sta in isole staccate."""
    bad = (leaked_ml >= leaked_floor_ml) and (largest_frac < frac_lo)
    if not bad:
        return None
    sev = 'alto' if leaked_ml >= 5.0 else 'medio'
    return {'code': 'islands', 'severity': sev,
            'msg': f'maschera in {n_components} componenti: {leaked_ml:.1f} ml '
                   f'({100 * (1 - largest_frac):.1f}%) fuori dall\'albero principale '
                   f'(isole = frammenti staccati, spesso falsi positivi)'}


# La severita' del radius-explosion scala con la DIMENSIONE del ramo che si gonfia:
# un leak in cisti/bolla/esofago diventa un pallone (>= ~10 mm); un ramo appena sopra
# soglia a 4-5 mm e' quasi sempre una biforcazione o rumore. Cosi' il QC resta
# silenzioso sui casi benigni e grida solo quando serve. Soglie OPERATIVE/ESPLORATIVE,
# non clinicamente validate.
BALLOON_MM = 10.0     # >= : sospetto leak (alto)
MED_MM = 6.0          # >= : da verificare (medio); sotto = basso (informativo)


def leak_summary(explosion, n_components, largest_frac, leaked_ml, total_ml=None, **kw):
    """Combina le due famiglie in un verdetto strutturato.
    Ritorna {ok, flags:[{code,severity,msg}], metrics:{...}}. `ok` e' True se non ci
    sono flag di severita' alta/media (i flag 'basso' restano informativi)."""
    balloon = kw.get('balloon_mm', BALLOON_MM)
    med = kw.get('med_mm', MED_MM)
    flags = []
    if explosion:
        n = len(explosion)
        worst = max(explosion, key=lambda e: e['d_mask'])   # severita' = DIMENSIONE
        d = worst['d_mask']
        if d >= balloon:
            sev = 'alto'
            tail = ('probabile leak in cisti/bolla/esofago (visibile sulla maschera '
                    'anche senza lume)')
        elif d >= med:
            sev = 'medio'
            tail = 'possibile leak — verifica la maschera'
        else:
            sev = 'basso'
            tail = ('a questo calibro spesso benigno (biforcazione/rumore); '
                    'verifica solo se il quadro lo suggerisce')
        flags.append({
            'code': 'radius_explosion', 'severity': sev,
            'msg': f'{n} ram{"o" if n == 1 else "i"} piu\' larg{"o" if n == 1 else "hi"} '
                   f'del genitore (peggiore: {worst["id"]} gen{worst["gen"]} '
                   f'{worst["d_mask"]}mm vs {worst["parent_d"]}mm, ×{worst["ratio"]}): {tail}'})
    cf = connectivity_flag(n_components, largest_frac, leaked_ml,
                           frac_lo=kw.get('frac_lo', 0.97),
                           leaked_floor_ml=kw.get('leaked_floor_ml', 0.5))
    if cf:
        flags.append(cf)
    metrics = {
        'n_radius_explosion': len(explosion),
        'radius_explosion_worst': max(explosion, key=lambda e: e['d_mask']) if explosion else None,
        'n_components': n_components,
        'largest_component_frac': round(largest_frac, 4),
        'leaked_islands_ml': round(leaked_ml, 2),
        'airway_total_ml': round(total_ml, 2) if total_ml is not None else None,
    }
    ok = not any(f['severity'] in ('alto', 'medio') for f in flags)
    return {'ok': ok, 'flags': flags, 'metrics': metrics}
