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
        if os.path.exists(airways):
            print(f"maschere gia' presenti in {segdir} — salto TotalSegmentator")
            return airways
        exe = shutil.which("TotalSegmentator") or shutil.which("totalsegmentator")
        if not exe:
            raise BackendError(
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


# registro dei backend disponibili. STEP 2 aggiungera' qui l'adapter AeroPath-ONNX.
REGISTRY = {b.name: b for b in (TotalSegmentatorBackend(),)}


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
