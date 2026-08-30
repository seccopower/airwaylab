"""Backend di segmentazione delle vie aeree, sostituibile (STEP 1: solo TotalSegmentator).

Il resto della pipeline consuma UNA cosa dal segmentatore: la maschera del lume
aereo allineata alla TC. Questo modulo nasconde QUALE modello la produce dietro
un'unica chiamata, `segment_airways(ct, segdir)`, cosi' che in futuro si possa
innestare un altro backend (es. AeroPath-ONNX, che gira su CPU) registrando un
adapter, senza toccare la pipeline. La scelta e' semplice e prevedibile e stampa
sempre quale backend ha usato e perche'; la provenienza finisce in
segdir/backend_info.json.

Nota di scopo: e' sostituibile SOLO la maschera delle vie aeree. I vasi
(arterie/vene) continuano a venire dal task lung_vessels di TotalSegmentator.

La logica di scelta e' pura e testata in airway_backend_core.py.
"""
import hashlib
import json
import os
import shutil
import subprocess

from airway_backend_core import BackendError, choose_backend  # noqa: F401
from geometry_qc_core import grids_match


def _grid_mismatch(ct, mask):
    """Motivo per cui `mask` NON e' sulla griglia di `ct`, o None.

    Legge le sole intestazioni (nibabel e' lazy: nessun voxel caricato).

    Segnala SOLO un disallineamento osservato: se una delle due intestazioni non
    e' leggibile la verifica non e' possibile e si restituisce None. "Non ho
    potuto controllare" non e' "ho trovato un problema" — scartare maschere valide
    perche' la CT non si apre peggiorerebbe le cose, e il guasto vero emerge
    comunque a valle dal controllo di qualita' di extmask.py, con un messaggio
    piu' preciso di quanto potremmo dare qui."""
    try:
        import nibabel as nib
        a, b = nib.load(ct), nib.load(mask)
        grids = ((a.shape[:3], a.header.get_zooms()[:3]),
                 (b.shape[:3], b.header.get_zooms()[:3]))
    except Exception:
        return None
    return grids_match(*grids[0], *grids[1])['reason']


def _sha256(path, chunk=1 << 20):
    """Hash del file maschera, per la provenienza (checksum riproducibile)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class TotalSegmentatorBackend:
    """Adapter per TotalSegmentator (task lung_vessels). Incapsula esattamente il
    comportamento storico: se le maschere ci sono gia' le riusa, altrimenti lancia
    TotalSegmentator dal PATH."""
    name = "totalsegmentator"
    task = "lung_vessels"
    airways_name = "lung_airways.nii.gz"

    def is_available(self, ct, segdir):
        # usabile se le maschere esistono gia', oppure se l'eseguibile e' nel PATH
        if os.path.exists(os.path.join(segdir, self.airways_name)):
            return True
        return bool(shutil.which("TotalSegmentator") or shutil.which("totalsegmentator"))

    def version(self):
        try:
            from importlib.metadata import version
            return version("TotalSegmentator")
        except Exception:
            return None

    def run(self, ct, segdir):
        airways = os.path.join(segdir, self.airways_name)
        stale = None
        if os.path.exists(airways):
            # La sola presenza del file non basta: una maschera calcolata su una
            # griglia diversa e' inservibile e piu' avanti extmask.py la trova
            # vuota. Meglio accorgersene qui e rigenerare.
            stale = _grid_mismatch(ct, airways)
            if not stale:
                print(f"maschere gia' presenti in {segdir} — salto TotalSegmentator")
                return airways
            print(f"maschere presenti in {segdir} ma su una griglia diversa dalla "
                  f"CT ({stale}) — le rigenero")
        exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
        if not exe:
            raise BackendError(
                ("maschere presenti ma inutilizzabili (" + stale + ") e "
                 if stale else "") +
                "TotalSegmentator non trovato nel PATH. Installalo, oppure fornisci "
                "gia' le maschere in " + segdir + " (lung_airways/arteries/veins).")
        print(f"\n=== TotalSegmentator (task {self.task}) -> {segdir} ===")
        r = subprocess.run([exe, "-i", ct, "-o", segdir, "--task", self.task])
        if r.returncode != 0:
            raise BackendError(f"ERROR in TotalSegmentator (exit {r.returncode})")
        if not os.path.exists(airways):
            raise BackendError(
                "TotalSegmentator non ha prodotto lung_airways.nii.gz in " + segdir)
        return airways


class AeroPathOnnxBackend:
    """Adapter per il modello AGU-Net di AeroPath (Raidionics, ONNX, gira su CPU).

    Licenza — la ragione del disegno. Il CODICE di AeroPath e' MIT e le librerie
    Raidionics (raidionicsseg/Raidionics-models) sono BSD-2, MA la licenza dei PESI
    del modello vie aeree non e' dichiarata esplicitamente. Percio' AirwayLab NON
    impacchetta i pesi: l'adapter CONSUMA una maschera prodotta da AeroPath sulla
    macchina dell'utente. Usare un modello in locale non e' ridistribuirlo — ed e'
    esattamente il pattern raccomandato dalla review (backend fuori processo,
    fornito dall'utente). La maschera entra poi nel percorso a backend esterno di
    AirwayLab (air witness, gate di risoluzione, centerline raffinata), come una
    qualsiasi maschera --mask.

    v1 = consumatore di maschera (nessun run automatico, che va validato insieme):
      - riusa una maschera AeroPath gia' presente in segdir; oppure
      - adotta il file indicato da AIRWAYLAB_AEROPATH_MASK.
    """
    name = "aeropath_onnx"
    mask_name = "aeropath_airways.nii.gz"

    def _existing(self, segdir):
        p = os.path.join(segdir, self.mask_name)
        return p if os.path.exists(p) else None

    def _env_mask(self):
        p = os.environ.get("AIRWAYLAB_AEROPATH_MASK")
        return p if p and os.path.exists(p) else None

    def is_available(self, ct, segdir):
        # disponibile solo se l'utente ha gia' prodotto la maschera AeroPath:
        # cosi' resta un opt-in esplicito e prevedibile (TS resta il default finche'
        # non c'e' una maschera AeroPath da usare).
        return bool(self._existing(segdir) or self._env_mask())

    def version(self):
        # versione della libreria di inferenza, se installata (best effort)
        try:
            from importlib.metadata import version
            return version("raidionicsseg")
        except Exception:
            return os.environ.get("AIRWAYLAB_AEROPATH_VERSION")

    def run(self, ct, segdir):
        os.makedirs(segdir, exist_ok=True)
        got = self._existing(segdir)
        if got:
            print(f"maschera AeroPath gia' presente in {segdir} — la riuso")
            return got
        env_mask = self._env_mask()
        if env_mask:
            dst = os.path.join(segdir, self.mask_name)
            print(f"adotto la maschera AeroPath da AIRWAYLAB_AEROPATH_MASK: {env_mask}")
            if os.path.abspath(env_mask) != os.path.abspath(dst):
                shutil.copy(env_mask, dst)
            return dst
        raise BackendError(
            "backend aeropath_onnx: nessuna maschera disponibile. Produci la maschera "
            "delle vie aeree con AeroPath (demo HuggingFace o raidionics), poi indicala "
            "con AIRWAYLAB_AEROPATH_MASK=/percorso/mask.nii.gz (oppure copiala in "
            + os.path.join(segdir, self.mask_name) + "). "
            "I pesi non sono impacchettati in AirwayLab: licenza dei pesi non dichiarata.")


# registro dei backend disponibili. L'ordine di preferenza (airway_backend_core)
# mette aeropath_onnx davanti: appena una maschera AeroPath e' disponibile diventa
# il default in automatico; TS resta il default finche' non lo e'.
REGISTRY = {b.name: b for b in (TotalSegmentatorBackend(), AeroPathOnnxBackend())}


def available_backends(ct, segdir):
    """Nomi dei backend usabili su questa macchina per questo caso."""
    return [name for name, b in REGISTRY.items() if b.is_available(ct, segdir)]


def segment_airways(ct, segdir, requested=None):
    """Produce (o riusa) la maschera delle vie aeree per `ct`.

    Sceglie il backend in modo semplice e prevedibile (vedi airway_backend_core),
    stampa quale e perche', scrive segdir/backend_info.json con la provenienza
    (backend, versione, task, checksum della maschera) e ritorna il percorso della
    maschera delle vie aeree."""
    os.makedirs(segdir, exist_ok=True)
    av = available_backends(ct, segdir)
    name, reason = choose_backend(av, requested=requested)
    print(f"[backend vie aeree] {name} — {reason}")

    backend = REGISTRY[name]
    airways = backend.run(ct, segdir)

    info = {
        "backend": name,
        "reason": reason,
        "task": getattr(backend, "task", None),
        "version": backend.version() if hasattr(backend, "version") else None,
        "airways_mask": os.path.basename(airways),
        "airways_sha256": _sha256(airways),
    }
    try:
        with open(os.path.join(segdir, "backend_info.json"), "w") as f:
            json.dump(info, f, indent=1)
    except OSError:
        pass  # la provenienza e' un extra: non far fallire il run se non si scrive
    return airways
