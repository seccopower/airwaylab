"""Selezione pura del backend di segmentazione delle vie aeree (vedi airway_backend.py).

Il resto della pipeline consuma UNA sola cosa dal segmentatore: una maschera del
lume aereo allineata alla TC. Rendere il segmentatore sostituibile significa
scegliere QUALE modello la produce dietro un'unica chiamata, senza toccare la
pipeline. Questa e' la logica di scelta, tenuta pura e testabile: date le opzioni
disponibili sulla macchina, decide quale usare, in modo SEMPLICE e PREVEDIBILE.

Regola: se il backend e' richiesto esplicitamente (--backend) si usa quello (o si
fallisce con un messaggio chiaro se non e' disponibile); altrimenti si prende il
primo disponibile secondo una lista di preferenza fissa. Nessun auto-detect della
GPU, nessuna euristica opaca.

Puro: nessun I/O. Testato in tests/test_airway_backend_core.py.
"""

# Ordine di preferenza quando NON e' richiesto un backend esplicito: vince il
# PRIMO disponibile. AeroPath-ONNX (CPU) sara' messo davanti a totalsegmentator
# quando il suo adapter arrivera' (step 2); finche' non c'e', totalsegmentator e'
# l'unica voce e resta il default. Ribaltare il default e' una riga qui.
DEFAULT_PREFERENCE = ("aeropath_onnx", "totalsegmentator")


class BackendError(SystemExit):
    """Interrompe con diagnostica quando non si puo' selezionare un backend usabile.
    Sottoclasse di SystemExit: stampa il messaggio ed esce non-zero, come prima."""


def choose_backend(available, requested=None, preference=DEFAULT_PREFERENCE):
    """Sceglie (nome, motivo) del backend vie aeree.

    available : iterabile di nomi di backend usabili SU QUESTA macchina.
    requested : nome esplicito (da --backend) oppure None per la scelta automatica.
    preference: ordine di preferenza per la scelta automatica.

    Ritorna (nome, motivo_leggibile). Solleva BackendError se nulla e' usabile o se
    il backend richiesto non e' disponibile."""
    av = list(dict.fromkeys(available))  # dedup preservando l'ordine

    if requested:
        if requested in av:
            return requested, f"richiesto esplicitamente (--backend {requested})"
        disp = ", ".join(av) if av else "nessuno"
        raise BackendError(
            f"backend '{requested}' non disponibile su questa macchina. "
            f"Disponibili: {disp}.")

    for name in preference:
        if name in av:
            motivo = ("unico disponibile" if len(av) == 1
                      else "primo disponibile secondo la preferenza")
            return name, motivo

    # nessun preferito disponibile: prendi comunque il primo disponibile, se c'e'
    if av:
        return av[0], "unico disponibile (fuori dalla lista di preferenza)"

    raise BackendError(
        "nessun backend di segmentazione delle vie aeree disponibile "
        "(installa TotalSegmentator, oppure fornisci maschere gia' calcolate).")
