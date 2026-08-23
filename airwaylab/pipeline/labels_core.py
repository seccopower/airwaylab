"""Nucleo puro per l'individuazione robusta dei bronchi principali (vedi labels.py).

Problema: assumere che i due bronchi principali siano i rami di generazione 1 e'
fragile — lo scheletro puo' creare monconi spuri alla carena o saltare generazioni,
e allora la carena VERA sta piu' in basso (caso reale casoDAS: entrambi i polmoni
pendevano da un solo ramo, l'altro figlio della trachea era un moncone di 1 mm).

Soluzione: i bronchi principali sono i due rami GRANDI, ciascuno "puro" di un lato
(quasi tutto il sottoalbero a sinistra o a destra della linea mediana), il piu'
PROSSIMALI possibile. Questo scarta i monconi (sottoalbero piccolo) e trova la
carena vera indipendentemente dalla numerazione delle generazioni.

`choose_mains` e' pura: opera su statistiche gia' calcolate per ramo. Testata in
tests/test_labels_core.py.
"""


def choose_mains(branch_info, min_sub=8, purity=0.8):
    """Sceglie (id_destro, id_sinistro) dei bronchi principali.

    branch_info: {id -> {'depth':int, 'n_sub':int, 'n_left':int, 'n_right':int}}
      depth   = profondita' dal ramo radice (numero di antenati)
      n_sub   = numero di rami del sottoalbero (incluso se stesso)
      n_left  = endpoint del sottoalbero sul lato SINISTRO paziente (x > mediana)
      n_right = endpoint sul lato DESTRO (x < mediana)   [n_left + n_right = n_sub]

    Un principale deve avere sottoalbero grande (>= min_sub) ed essere "puro" di un
    lato (frazione del lato >= purity). Tra i candidati di un lato si prende il piu'
    prossimale (depth minima); a parita', il sottoalbero maggiore.
    Ritorna (rmb_id, lmb_id); ciascuno None se non trovato."""
    def cands(side):
        out = []
        for bid, s in branch_info.items():
            n = s.get('n_sub', 0)
            if n < min_sub:
                continue
            frac = (s.get('n_right', 0) if side == 'R' else s.get('n_left', 0)) / n
            if frac >= purity:
                out.append(bid)
        return out

    def best(ids):
        if not ids:
            return None
        return sorted(ids, key=lambda b: (branch_info[b]['depth'],
                                          -branch_info[b]['n_sub']))[0]

    rmb = best(cands('R'))
    lmb = best(cands('L'))
    # se coincidono (albero degenere) o uno manca, ritorna quel che c'e'
    if rmb is not None and rmb == lmb:
        lmb = None
    return rmb, lmb
