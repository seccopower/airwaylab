"""Mucus Plug Score segmentario — nucleo di calcolo (puro).

Lo score standard (Fahy/UCSF, Dunican) conta QUANTI dei 18 segmenti
broncopolmonari contengono almeno un bronco completamente occluso da muco. E'
deliberatamente grossolano: il criterio per segmento e' BINARIO, quindi tre
tappi in un lobo superiore destro pesano quanto uno solo. Il fenotipo si legge
poi su tre fasce, 0 / 1-3 / >=4.

Perche' contare i segmenti e non i tappi. Il numero di tappi dipende da quanto
in profondita' arriva la segmentazione e da come si contano i frammenti di uno
stesso tappo; il numero di segmenti coinvolti no. Contare i segmenti sacrifica
risoluzione in cambio di stabilita' fra protocolli, ed e' la ragione per cui la
letteratura usa questa forma.

Cosa questo modulo fa e cosa NON fa. Qui c'e' solo l'aritmetica dello score dato
l'input. L'ASSEGNAZIONE di un tappo al suo segmento — che e' la parte difficile e
quella che decide davvero il risultato — sta a monte e non e' compresa. Anche il
criterio di "occlusione completa" e' a monte: questo modulo si fida che ogni
codice ricevuto rappresenti gia' un bronco completamente occluso.

Trasparenza sugli scarti. Un tappo che non si riesce ad assegnare a un segmento
ABBASSA lo score, perche' non accende nessuna casella. Silenziarlo darebbe un
numero piu' basso e all'apparenza piu' pulito, quindi `n_unassigned` e
`n_unknown_code` sono riportati accanto al punteggio: uno score 2 con tre tappi
non assegnati non e' uno score 2.

Esplorativo, non endpoint validato: nessuna soglia e' stata verificata su questa
implementazione.

Puro: solo aritmetica, nessun I/O. Testato in tests/test_mps_core.py.
"""

# I 18 segmenti broncopolmonari nell'ordine canonico: destra (10) poi sinistra
# (8). A sinistra LB1+2 e LB7+8 sono fusi, com'e' d'uso nell'anatomia sinistra:
# per questo i segmenti sono 18 e non 20.
SEGMENTS_18 = (
    ('RB1', 'RUL apicale'),
    ('RB2', 'RUL posteriore'),
    ('RB3', 'RUL anteriore'),
    ('RB4', 'RML laterale'),
    ('RB5', 'RML mediale'),
    ('RB6', 'RLL superiore'),
    ('RB7', 'RLL basale mediale'),
    ('RB8', 'RLL basale anteriore'),
    ('RB9', 'RLL basale laterale'),
    ('RB10', 'RLL basale posteriore'),
    ('LB1+2', 'LUL apicoposteriore'),
    ('LB3', 'LUL anteriore'),
    ('LB4', 'lingula superiore'),
    ('LB5', 'lingula inferiore'),
    ('LB6', 'LLL superiore'),
    ('LB7+8', 'LLL basale anteromediale'),
    ('LB9', 'LLL basale laterale'),
    ('LB10', 'LLL basale posteriore'),
)

SEGMENT_CODES = tuple(code for code, _ in SEGMENTS_18)
SEGMENT_NAMES = {code: name for code, name in SEGMENTS_18}

# Fasce di fenotipo. Soglie della letteratura, NON validate su questa
# implementazione: cambiare qui e non altrove.
PHENOTYPE_LOW_MAX = 3        # 1-3 segmenti = carico basso
THRESHOLDS = {'low': '1-3', 'high': '>=4'}


def phenotype_for(mps):
    """Fenotipo corrispondente a un punteggio. 'none' / 'low' / 'high'."""
    n = int(mps)
    if n <= 0:
        return 'none'
    return 'low' if n <= PHENOTYPE_LOW_MAX else 'high'


def mucus_plug_score(plug_segments):
    """Mucus Plug Score da un elenco di segmenti occlusi.

    plug_segments: iterabile di codici-segmento, UNO per ogni bronco
      completamente occluso (ogni tappo gia' ridotto al suo segmento). Puo'
      contenere None — tappo non assegnabile a un segmento — e codici che non
      appartengono ai 18: entrambi vengono contati a parte e NON entrano nello
      score, perche' inventare un segmento sarebbe peggio che dichiarare la
      lacuna.

    Ritorna un dict con per_segment (tutti e 18, 0/1), segments_occluded, mps,
    phenotype, i tre conteggi di ingresso e le soglie usate.

    Piu' tappi nello stesso segmento contano UNA volta: il criterio per segmento
    e' binario per costruzione."""
    per_segment = {code: 0 for code in SEGMENT_CODES}
    n_plugs = 0
    n_unassigned = 0
    n_unknown = 0

    for item in (plug_segments or ()):
        n_plugs += 1
        if item is None:
            n_unassigned += 1
        elif item in per_segment:
            per_segment[item] = 1
        else:
            n_unknown += 1

    # ordine canonico, non quello di arrivo: il confronto fra casi dev'essere
    # possibile senza riordinare a valle
    occluded = [code for code in SEGMENT_CODES if per_segment[code]]
    mps = len(occluded)

    return {
        'status': 'exploratory',
        'method_id': 'mps_airwaylab',
        'per_segment': per_segment,
        'segments_occluded': occluded,
        'mps': mps,
        'phenotype': phenotype_for(mps),
        'n_plugs': n_plugs,
        'n_unassigned': n_unassigned,
        'n_unknown_code': n_unknown,
        'thresholds': dict(THRESHOLDS),
    }
