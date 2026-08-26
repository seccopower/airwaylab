"""Carta d'identita' condivisa per i JSON dei descrittori esplorativi.

Perche' esiste. Ogni descrittore per-soggetto che aggiungiamo — Pi10, tapering,
morfometria dell'albero, parenchima, pruning vascolare, densitometria — e' un
numero MISURATO con i suoi limiti, NON un endpoint biologico validato. La revisione
avversariale ha segnalato che il report mostrava questi numeri senza la loro
provenienza, come se lo fossero. Questo modulo timbra ogni JSON con la stessa carta
d'identita' — stato, id-metodo, parametri, denominatori, esclusioni, backend e
versione del tool — cosi' il report puo' mostrare il caveat e un revisore puo'
ricostruire esattamente come e' stato calcolato il numero.

`build_provenance` e' puro (nessun I/O): riceve il dict di seg_info gia' letto ed e'
testato in tests/test_provenance.py. `provenance` e' il wrapper che legge
out/seg_info.json e degrada con grazia se manca.
"""
import json
import os

STATUS_EXPLORATORY = 'exploratory'


def build_provenance(seg_info, method_id, params=None, denominators=None,
                     exclusions=None, status=STATUS_EXPLORATORY):
    """Costruisce il blocco provenienza da annidare sotto 'provenance' nel JSON.

    seg_info: dict di out/seg_info.json (o {} se assente) — da' backend, versione,
      iso. method_id: identificatore stabile e onesto del metodo (es.
      'pi10_airwaylab'). params/denominators/exclusions: cosa e' entrato nel calcolo
      e cosa e' stato escluso. Nessun valore inventato: cio' che manca resta None."""
    info = seg_info or {}
    return {
        'status': status,
        'method_id': method_id,
        'airwaylab_version': info.get('airwaylab_version'),
        'backend': info.get('backend'),
        'iso_mm': info.get('iso'),
        'params': params or {},
        'denominators': denominators or {},
        'exclusions': exclusions or {},
    }


def _read_seg_info():
    for p in ('out/seg_info.json', 'seg_info.json'):
        if os.path.exists(p):
            try:
                return json.load(open(p))
            except (ValueError, OSError):
                return {}
    return {}


def provenance(method_id, params=None, denominators=None, exclusions=None,
               status=STATUS_EXPLORATORY):
    """Come build_provenance, ma legge out/seg_info.json dal work dir corrente."""
    return build_provenance(_read_seg_info(), method_id, params=params,
                            denominators=denominators, exclusions=exclusions,
                            status=status)
