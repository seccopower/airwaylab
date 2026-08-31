"""Assegnazione dei tappi di muco ai 18 segmenti broncopolmonari (nucleo puro).

Un tappo e' individuato su un RAMO; il Mucus Plug Score si conta per SEGMENTO. Il
ramo occluso e' spesso piu' distale del bronco segmentale, quindi l'assegnazione
risale l'albero dal ramo del tappo fino al primo antenato che porti
un'etichetta segmentale, e traduce il suo `aid` (anatomy.py) in uno dei 18
codici usati da mps_core.

Due regole che decidono il risultato piu' di ogni altra cosa.

1. NESSUN RIPIEGO SUL LOBO. Se risalendo si incontra un'etichetta lobare (RUL,
   LLL, lingulare...) prima di una segmentale, il tappo resta NON assegnato.
   Attribuirlo al lobo darebbe un punteggio piu' alto e piu' pulito inventando
   un'informazione che non c'e': il segmento non e' stato identificato, e un
   MPS costruito su segmenti dedotti non e' l'MPS della letteratura.

2. IL DENOMINATORE VERO NON E' SEMPRE 18. L'etichettatura segmentale e'
   best-effort: su un caso reale ne sono stati identificati 14 su 18 (a destra
   mancavano B2, B7, B8 e B9). Un tappo che cade in un segmento mai etichettato
   e' inassegnabile, e ABBASSA lo score senza che nulla lo dica. Per questo il
   risultato riporta `segments_labeled`: quanti dei 18 erano raggiungibili in
   questo albero, e `assignable_frac`: quanta parte dell'albero potrebbe contribuire
allo score. Uno score 3 su 14 segmenti etichettati non e' uno score 3 su 18, e un
punteggio basso su un albero poco assegnabile non distingue 'pochi tappi' da
'pochi segmenti raggiungibili'.

Il criterio di "occlusione completa" resta a monte: qui si assume che ogni tappo
ricevuto rappresenti gia' un bronco completamente occluso.

Puro: solo aritmetica e visita dell'albero, nessun I/O. Testato in
tests/test_mps_assign_core.py.
"""
from mps_core import SEGMENT_CODES

# aid (anatomy.py) -> codice segmento MPS.
# A destra la corrispondenza e' uno a uno. A sinistra alcuni segmenti sono fusi
# nell'anatomia corrente: labels.py assegna 'B1+2 sx' solo quando trova B1 senza
# B2, altrimenti restano separati — quindi B1_L, B2_L e B1+2_L confluiscono
# tutti in LB1+2. Analogamente B7_L e B8_L confluiscono in LB7+8 (a sinistra
# labels.py non assegna mai B7, ma la mappa lo copre comunque).
AID_TO_SEGMENT = {
    'B1_R': 'RB1', 'B2_R': 'RB2', 'B3_R': 'RB3', 'B4_R': 'RB4', 'B5_R': 'RB5',
    'B6_R': 'RB6', 'B7_R': 'RB7', 'B8_R': 'RB8', 'B9_R': 'RB9', 'B10_R': 'RB10',
    'B1+2_L': 'LB1+2', 'B1_L': 'LB1+2', 'B2_L': 'LB1+2',
    'B3_L': 'LB3', 'B4_L': 'LB4', 'B5_L': 'LB5', 'B6_L': 'LB6',
    'B7+8_L': 'LB7+8', 'B7_L': 'LB7+8', 'B8_L': 'LB7+8',
    'B9_L': 'LB9', 'B10_L': 'LB10',
}

# motivi di mancata assegnazione, riportati per tappo
REASON_OK = 'assegnato'
REASON_NO_BRANCH = 'ramo_assente_dall_albero'
REASON_NO_SEGMENT = 'nessun_antenato_segmentale'

MAX_WALK = 64          # guardia contro alberi malformati (cicli nei genitori)


def aid_to_segment(aid):
    """Codice segmento MPS per un aid anatomico, o None se non e' segmentale."""
    if not aid:
        return None
    return AID_TO_SEGMENT.get(str(aid))


def build_parent_map(branches):
    """{id_figlio: id_genitore} dalla topologia u->v dei rami.

    Un ramo e' figlio di quello che termina dove lui comincia."""
    starts = {}
    for b in branches:
        starts.setdefault(b.get('u'), []).append(b.get('id'))
    parent = {}
    for p in branches:
        for child_id in starts.get(p.get('v'), ()):
            if child_id != p.get('id'):
                parent[child_id] = p.get('id')
    return parent


def labeled_segments(branches):
    """I codici dei 18 segmenti effettivamente etichettati in questo albero,
    in ordine canonico. E' il denominatore vero dello score."""
    found = {aid_to_segment(b.get('aid')) for b in branches}
    return [c for c in SEGMENT_CODES if c in found]


def resolve_segment(branch_id, by_id, parent, max_walk=MAX_WALK):
    """Risale dall'id dato al primo antenato con etichetta segmentale.

    Ritorna (segmento, ramo_che_ha_etichettato, livelli_risaliti, motivo); il
    segmento e' None quando la risalita finisce senza trovarne uno. Unica
    implementazione della risalita: la usano sia l'assegnazione dei tappi sia il
    calcolo della copertura, cosi' non possono divergere."""
    if branch_id not in by_id:
        return (None, None, None, REASON_NO_BRANCH)
    cur, steps, seen = branch_id, 0, set()
    while cur is not None and steps < max_walk and cur not in seen:
        seen.add(cur)
        cand = aid_to_segment((by_id.get(cur) or {}).get('aid'))
        if cand:
            return (cand, cur, steps, REASON_OK)
        cur = parent.get(cur)
        steps += 1
    return (None, None, None, REASON_NO_SEGMENT)


def assignable_coverage(branches, max_walk=MAX_WALK):
    """Quanti rami dell'albero POTREBBERO contribuire allo score.

    Un ramo e' assegnabile se, risalendo, incontra un segmento etichettato. La
    frazione dice quanta parte dell'albero e' fuori portata dell'MPS prima
    ancora di sapere dove stanno i tappi: se e' bassa, uno score basso puo'
    significare 'pochi tappi' oppure 'pochi segmenti raggiungibili', e le due
    cose non si distinguono guardando il solo punteggio."""
    branches = list(branches or ())
    by_id = {b.get('id'): b for b in branches}
    parent = build_parent_map(branches)
    n = len(branches)
    ok = sum(1 for b in branches
             if resolve_segment(b.get('id'), by_id, parent, max_walk)[0])
    return {'n_branches': n, 'n_branches_assignable': ok,
            'assignable_frac': (ok / n) if n else None}


def assign_plugs_to_segments(plugs, branches, max_walk=MAX_WALK):
    """Assegna ogni tappo al suo segmento risalendo l'albero.

    plugs:    iterabile di dict con 'pid' e 'branch' (id del ramo occluso)
    branches: iterabile di dict con 'id', 'u', 'v' e, se etichettato, 'aid'

    Ritorna un dict con:
      assignments  una riga per tappo: pid, branch, segment (o None),
                   via_branch (il ramo che ha fornito l'etichetta), levels_up
                   (0 = il ramo stesso; piu' alto = evidenza piu' debole),
                   reason
      segments     lista parallela di codici (None dove non assegnato), pronta
                   per mucus_plug_score
      segments_labeled / n_segments_labeled  il denominatore vero
      n_plugs, n_assigned, n_unassigned, unassigned_by_reason
      n_branches / n_branches_assignable / assignable_frac  quanta parte
                   dell'albero potrebbe contribuire allo score (vedi
                   assignable_coverage)
    """
    branches = list(branches or ())
    by_id = {b.get('id'): b for b in branches}
    parent = build_parent_map(branches)
    labeled = labeled_segments(branches)

    assignments = []
    segments = []
    reasons = {}

    for p in (plugs or ()):
        pid = p.get('pid')
        start = p.get('branch')
        seg, via, levels, reason = resolve_segment(start, by_id, parent, max_walk)

        assignments.append({'pid': pid, 'branch': start, 'segment': seg,
                            'via_branch': via, 'levels_up': levels,
                            'reason': reason})
        segments.append(seg)
        if seg is None:
            reasons[reason] = reasons.get(reason, 0) + 1

    n_plugs = len(assignments)
    n_assigned = sum(1 for a in assignments if a['segment'])
    cov = assignable_coverage(branches, max_walk)
    return {
        'status': 'exploratory',
        'method_id': 'mps_assign_airwaylab',
        'assignments': assignments,
        'segments': segments,
        'segments_labeled': labeled,
        'n_segments_labeled': len(labeled),
        'n_plugs': n_plugs,
        'n_assigned': n_assigned,
        'n_unassigned': n_plugs - n_assigned,
        'unassigned_by_reason': reasons,
        'n_branches': cov['n_branches'],
        'n_branches_assignable': cov['n_branches_assignable'],
        'assignable_frac': cov['assignable_frac'],
    }
