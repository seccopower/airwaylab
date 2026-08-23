#!/usr/bin/env python3
"""Inferenza AeroPath in LOCALE su CPU -> maschera vie aeree per AirwayLab.

Produce la maschera delle vie aeree con il modello AGU-Net di AeroPath (Raidionics,
ONNX, CPU) e la salva dove vuoi. Poi la dai in pasto ad AirwayLab con:
    set AIRWAYLAB_AEROPATH_MASK=...\\aeropath_airways.nii.gz   (PowerShell: $env:...)
    airwaylab report <dicom> --name casoX_aero --outdir ...

NON fa parte del pacchetto airwaylab: e' un ponte verso la libreria esterna
raidionicsrads, che ha dipendenze proprie. INSTALLALA IN UN VENV SEPARATO per non
sporcare l'ambiente di AirwayLab:
    python -m venv .venv-aero
    .venv-aero\\Scripts\\activate      (Windows)   |   source .venv-aero/bin/activate
    pip install raidionicsrads

MODELLI (una volta sola). Servono DUE cartelle-modello: CT_Airways e
MRI_SequenceClassifier (il pipeline le usa entrambe). raidionicsrads prova a
scaricarle da solo; se non le trova, prendile dagli asset della release di AeroPath
(https://github.com/raidionics/AeroPath/releases) e mettile sotto la cartella
--models, una sottocartella per modello:
    <models>/CT_Airways/...            <models>/MRI_SequenceClassifier/...

Riproduce fedelmente l'invocazione del demo ufficiale (run_rads, gpu_id=-1 = CPU,
input in patient/T0/, output *-Airways*). NON ancora validato end-to-end da noi:
al primo run reale rifiniamo insieme partendo dall'output d'errore. Usa --dry-run
per generare config/pipeline/layout SENZA lanciare l'inferenza (per ispezione).
"""
import argparse
import configparser
import glob
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

# archivi-modello dallo zoo Raidionics (v1.3.0-rc). Servono entrambi: il pipeline
# usa il classificatore di sequenza e poi il modello vie aeree.
MODEL_URLS = {
    "MRI_SequenceClassifier":
        "https://github.com/raidionics/Raidionics-models/releases/download/"
        "v1.3.0-rc/Raidionics-MRI_SequenceClassifier-v13.zip",
    "CT_Airways":
        "https://github.com/raidionics/Raidionics-models/releases/download/"
        "v1.3.0-rc/Raidionics-CT_Airways-v13.zip",
}


def _has_model(models_dir, name):
    d = os.path.join(models_dir, name)
    if not os.path.isdir(d):
        return False
    return bool(glob.glob(os.path.join(d, "**", "*.onnx"), recursive=True)
                or glob.glob(os.path.join(d, "**", "pre_processing.ini"), recursive=True))


def _place_extracted(extract_dir, models_dir, name):
    """Trova nella cartella estratta la radice del modello (quella che contiene
    pre_processing.ini o un .onnx) e la colloca in models_dir/name."""
    marker = None
    for root, _dirs, files in os.walk(extract_dir):
        if "pre_processing.ini" in files or any(f.endswith(".onnx") for f in files):
            marker = root
            break
    if marker is None:
        raise RuntimeError(f"archivio {name}: nessun modello (pre_processing.ini/.onnx) trovato")
    dest = os.path.join(models_dir, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(marker, dest)
    return dest


def ensure_models(models_dir, names=("MRI_SequenceClassifier", "CT_Airways"), download=True):
    """Garantisce che i modelli richiesti siano in models_dir; li scarica se mancano."""
    os.makedirs(models_dir, exist_ok=True)
    for name in names:
        if _has_model(models_dir, name):
            print(f"modello presente: {name}")
            continue
        if not download:
            sys.exit(f"modello mancante: {name}. Scaricalo da {MODEL_URLS[name]} "
                     f"ed estrailo in {os.path.join(models_dir, name)}, oppure togli --no-download.")
        url = MODEL_URLS[name]
        print(f"scarico {name} ...\n  {url}")
        with tempfile.TemporaryDirectory() as td:
            zp = os.path.join(td, "m.zip")
            urllib.request.urlretrieve(url, zp)
            ex = os.path.join(td, "ex")
            os.makedirs(ex)
            with zipfile.ZipFile(zp) as z:
                z.extractall(ex)
            _place_extracted(ex, models_dir, name)
        print(f"  installato in {os.path.join(models_dir, name)}")


def _stem_ext(path):
    base = os.path.basename(path)
    if base.endswith(".nii.gz"):
        return base[:-7], "nii.gz"
    stem, ext = os.path.splitext(base)
    return stem, ext.lstrip(".")


def build_layout(ct, workdir):
    """Prepara patient/T0/<stem>-t1gd.<ext> (naming atteso dal pipeline del demo)
    e la cartella di output. Ritorna (input_folder, output_folder, stem)."""
    stem, ext = _stem_ext(ct)
    patient = os.path.join(workdir, "patient")
    t0 = os.path.join(patient, "T0")
    out = os.path.join(workdir, "output")
    os.makedirs(t0, exist_ok=True)
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(t0, f"{stem}-t1gd.{ext}")   # suffisso del demo (retaggio MRI)
    if os.path.abspath(ct) != os.path.abspath(dst):
        shutil.copy(ct, dst)
    return patient, out, stem


def write_pipeline(output_folder):
    """Scrive il pipeline a due passi del demo: classificatore di sequenza -> CT_Airways."""
    pip = {
        "1": {"task": "Classification", "inputs": {}, "target": ["MRSequence"],
              "model": "MRI_SequenceClassifier",
              "description": "Classification of the MRI sequence type for all input scans."},
        "2": {"task": "Model selection", "model": "CT_Airways", "timestamp": 0,
              "format": "thresholding",
              "description": "Identifying the best CT_Airways segmentation model for existing inputs"},
    }
    path = os.path.join(output_folder, "test_pipeline.json")
    with open(path, "w") as f:
        json.dump(pip, f, indent=1)
    return path


def write_config(workdir, input_folder, output_folder, model_folder, pipeline_path):
    cfg = configparser.ConfigParser()
    cfg["Default"] = {"task": "mediastinum_diagnosis", "caller": ""}
    cfg["System"] = {"gpu_id": "-1", "input_folder": input_folder,
                     "output_folder": output_folder, "model_folder": model_folder,
                     "pipeline_filename": pipeline_path}
    cfg["Runtime"] = {"reconstruction_method": "thresholding",
                      "reconstruction_order": "resample_first",
                      "use_preprocessed_data": "False"}
    path = os.path.join(workdir, "rads_config.ini")
    with open(path, "w") as f:
        cfg.write(f)
    return path


def collect_airways(output_folder, dest):
    """Trova la maschera *-Airways* prodotta e la copia in `dest`. Ritorna dest o None."""
    hits = glob.glob(os.path.join(output_folder, "**", "*Airways*.nii*"), recursive=True)
    if not hits:
        return None
    hits.sort(key=lambda p: len(p))
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    shutil.copy(hits[0], dest)
    return dest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Inferenza AeroPath locale (CPU) per AirwayLab")
    ap.add_argument("input", help="CT anonimizzata .nii.gz (la STESSA che processera' AirwayLab)")
    ap.add_argument("output", help="dove salvare la maschera vie aeree (.nii.gz)")
    ap.add_argument("--models", default="./aeropath_models",
                    help="cartella con le sottocartelle CT_Airways e MRI_SequenceClassifier")
    ap.add_argument("--workdir", default="./aeropath_work", help="cartella di lavoro temporanea")
    ap.add_argument("--dry-run", action="store_true",
                    help="genera config/pipeline/layout senza lanciare l'inferenza")
    ap.add_argument("--no-download", action="store_true",
                    help="non scaricare i modelli mancanti (falliscono se assenti)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.input):
        sys.exit(f"input non trovato: {args.input}")
    os.makedirs(args.workdir, exist_ok=True)
    model_folder = os.path.abspath(args.models)

    input_folder, output_folder, stem = build_layout(args.input, args.workdir)
    pipeline_path = os.path.abspath(write_pipeline(output_folder))
    config_path = write_config(args.workdir, os.path.abspath(input_folder),
                               os.path.abspath(output_folder), model_folder, pipeline_path)
    print(f"config:   {config_path}")
    print(f"pipeline: {pipeline_path}")
    print(f"input:    {input_folder}/T0/{stem}-t1gd.*")
    print(f"models:   {model_folder}  (servono CT_Airways/ e MRI_SequenceClassifier/)")

    if args.dry_run:
        print("dry-run: config/pipeline/layout pronti, inferenza NON lanciata.")
        return 0

    # modelli: scaricali/verificali PRIMA di run_rads (che non li scarica da solo
    # quando model_folder e' impostato)
    ensure_models(model_folder, download=not args.no_download)

    try:
        from raidionicsrads.compute import run_rads
    except Exception as e:
        sys.exit("raidionicsrads non importabile: installalo in un venv dedicato "
                 f"(pip install raidionicsrads). Dettaglio: {e}")

    print("\n=== run_rads (CPU) — puo' richiedere minuti ===")
    run_rads(config_filename=config_path)

    dest = collect_airways(output_folder, args.output)
    if not dest:
        sys.exit("inferenza finita ma nessun file *Airways* trovato in "
                 f"{output_folder}. Ispeziona l'albero di output e dimmi cosa c'e'.")
    print(f"\nOK -> maschera vie aeree: {dest}")
    print("Ora in AirwayLab:")
    print(f"  set AIRWAYLAB_AEROPATH_MASK={os.path.abspath(dest)}   (PowerShell: $env:AIRWAYLAB_AEROPATH_MASK=...)")
    print("  airwaylab report <dicom> --name <caso>_aero --outdir <...>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
