"""AirwayLab command-line interface.

Usage:
    airwaylab anonymize <dicom_dir> <output.nii.gz> [series_index]
    airwaylab run <ct.nii.gz> [--name CASE] [--outdir DIR]
    airwaylab run <ct.nii.gz> --mask <airway_mask.nii.gz>   (external/DL backend)
    airwaylab version
"""
import argparse
import html as _html
import json
import os
import shutil
import subprocess
import sys

from . import __version__

PKG = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.join(PKG, "pipeline")
ASSETS = os.path.join(PKG, "assets")

TAIL_STEPS = [
    ("snapshots.py", "per-branch verification sections"),
    ("profile.py", "longitudinal caliber/wall profiles"),
    ("pi10.py", "Pi10 wall-remodeling metric (√WA at Pi=10)"),
    ("tapering.py", "airway tapering (child/parent ratio + gradient)"),
    ("treestats.py", "tree morphometry: counts + fractal dimension"),
    ("lung.py", "lungs, densitometry, dysanapsis"),
    ("parenchyma.py", "parenchyma beyond mean density (heterogeneity, LAA clusters)"),
    ("territory.py", "parenchymal territories + Murray fit"),
    ("vessels.py", "vascular tree: TBV, BV5"),
    ("vascular_gradient.py", "vascular pruning gradient (small-vessel density vs pleura)"),
    ("vtree.py", "vascular graph"),
    ("pair.py", "bronchus-artery pairing (BA ratio)"),
    ("dual.py", "airway-vascular mismatch map"),
    ("vasculature.py", "artery/vein analysis (pruning, A/V) — optional"),
    ("plugs.py", "mucus plug candidates"),
    ("cpr.py", "straightened CPR panels"),
    ("viz.py", "report data"),
]


def build_steps(mask=None, refine=False):
    """Step list for a run. --mask switches to the external-segmentation
    backend and enables the air witness and the refined centerline."""
    steps = [("extmask.py", "external segmentation backend") if mask
             else ("segment.py", "airway segmentation")]
    steps += [("tree.py", "skeleton, graph, generations"),
              ("labels.py", "anatomical labels (best effort)")]
    if mask or refine:
        steps.append(("refine.py", "sub-voxel centerline + B-spline"))
    steps.append(("measure.py", "lumen caliber + wall thickness"))
    if mask:
        steps.append(("witness.py", "air witness + resolution gates"))
    return steps + TAIL_STEPS


def _step(script, args=(), env=None):
    r = subprocess.run([sys.executable, os.path.join(PIPE, script), *args], env=env)
    if r.returncode != 0:
        sys.exit(f"ERROR in {script} (exit {r.returncode})")


def _build_html(prefix, name, biopsies_json="[]"):
    info = json.load(open("out/seg_info.json"))
    tpl = open(os.path.join(ASSETS, "template.html"), encoding="utf-8").read()
    pl = open(os.path.join(ASSETS, "plotly.min.js"), encoding="utf-8").read()
    data = open("out/map_data.json", encoding="utf-8").read()
    sp = info.get("native_spacing") or ["?", "?", "?"]
    safe_name = _html.escape(name)
    if info.get("threshold") is not None:
        seg_desc = f"adaptive region growing ({int(info['threshold'])}&#8202;HU)"
    else:
        seg_desc = _html.escape(info.get("backend", "external segmentation")) + \
            " · air witness + resolution gates"
    sub = (
        f"{safe_name} — native slices {sp[2]}&#8202;mm, {info['iso']}&#8202;mm isotropic "
        f"reconstruction · {seg_desc} · "
        f"AirwayLab v{__version__}"
    )
    html = (
        tpl.replace("__SUBTITLE__", sub)
        .replace("__PLOTLY__", pl)
        .replace("__DATA__", data)
        .replace("__PRE_EXCLUDED__", "[]")
        .replace("__PRE_BIOPSIES__", biopsies_json)
    )
    with open(prefix + "_report.html", "w", encoding="utf-8") as f:
        f.write(html)


def _qc_images(prefix):
    import numpy as np
    import SimpleITK as sitk
    from PIL import Image

    ct = sitk.GetArrayFromImage(sitk.ReadImage("out/ct_iso.nii.gz")).astype(np.float32)
    mask = sitk.GetArrayFromImage(sitk.ReadImage("out/airway_mask.nii.gz")).astype(bool)
    win = np.clip((ct + 1000) / 1400, 0, 1)
    rgb = np.stack([win.mean(axis=1)] * 3, -1)
    rgb[mask.max(axis=1)] = [1, 0.2, 0.2]
    Image.fromarray((np.flipud(rgb) * 255).astype(np.uint8)).save(prefix + "_qc_coronal.png")
    zs = np.nonzero(mask.any(axis=(1, 2)))[0]
    if len(zs) == 0:
        print("WARNING: empty airway mask; skipping axial QC image")
        return
    picks = np.linspace(zs.max(), zs.min(), 6).astype(int)
    tiles = []
    for z in picks:
        w2 = np.clip((ct[z] + 1000) / 1200, 0, 1)
        t = np.stack([w2] * 3, -1)
        t[mask[z]] = t[mask[z]] * 0.4 + np.array([0.6, 0.1, 0.1])
        tiles.append(t)
    img = np.concatenate(
        [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0
    )
    Image.fromarray((img * 255).astype(np.uint8)).save(prefix + "_qc_axial.png")


def cmd_run(args):
    src = os.path.abspath(args.ct)
    mask = os.path.abspath(args.mask) if args.mask else None
    refined = bool(mask or args.refine)
    tight = bool(mask)
    # explicit per-run environment: no leakage into (or from) other runs in
    # the same process, flags removed when not applicable
    run_env = {k: v for k, v in os.environ.items()
               if k not in ("AIRWAYLAB_SPACING", "AIRWAYLAB_REFINED",
                            "AIRWAYLAB_TIGHT_SMALL", "AIRWAYLAB_ARTERIES",
                            "AIRWAYLAB_VEINS")}
    if args.spacing:
        run_env["AIRWAYLAB_SPACING"] = str(args.spacing)
    if refined:
        run_env["AIRWAYLAB_REFINED"] = "1"
    if tight:
        run_env["AIRWAYLAB_TIGHT_SMALL"] = "1"
    # maschere arteria/vena: accanto al --mask (stesso run TotalSegmentator) o
    # da flag espliciti. Opzionali: lo step vascolare si salta se assenti.
    if mask:
        _mdir = os.path.dirname(mask)
        _art = args.arteries or os.path.join(_mdir, "lung_arteries.nii.gz")
        _vein = args.veins or os.path.join(_mdir, "lung_veins.nii.gz")
        if os.path.exists(_art):
            run_env["AIRWAYLAB_ARTERIES"] = os.path.abspath(_art)
        if os.path.exists(_vein):
            run_env["AIRWAYLAB_VEINS"] = os.path.abspath(_vein)
    name = args.name or os.path.basename(src).replace(".nii.gz", "").replace(".nii", "")
    outroot = os.path.abspath(args.outdir or os.getcwd())
    work = os.path.join(outroot, name + "_work")
    os.makedirs(work, exist_ok=True)
    os.chdir(work)
    def _record_provenance():
        # run-mode audit trail: written right after segmentation so every
        # later step (viz.py included) sees the complete provenance
        info = json.load(open("out/seg_info.json"))
        info.setdefault("backend", "adaptive region growing")
        info["refined_centerline"] = refined
        info["tight_small_window"] = tight
        info["airwaylab_version"] = __version__
        json.dump(info, open("out/seg_info.json", "w"), indent=1)

    for k, (script, desc) in enumerate(build_steps(mask=mask, refine=args.refine)):
        print(f"\n=== {script} — {desc} ===")
        if script == "segment.py":
            _step(script, [src], env=run_env)
        elif script == "extmask.py":
            _step(script, [src, mask], env=run_env)
        else:
            _step(script, [], env=run_env)
        if k == 0:
            _record_provenance()
    biopsies_json = "[]"
    if getattr(args, "biopsies", None):
        with open(args.biopsies, encoding="utf-8") as f:
            loaded = json.load(f)
        # accetta sia la lista nuda sia l'export del report {caso, biopsie}
        biopsies_json = json.dumps(
            loaded.get("biopsie", loaded) if isinstance(loaded, dict) else loaded)
    prefix = os.path.join(outroot, name)
    _build_html(prefix, name, biopsies_json)
    _qc_images(prefix)
    shutil.copy("out/branches.csv", prefix + "_branches.csv")
    if os.path.exists("out/routes.csv"):
        shutil.copy("out/routes.csv", prefix + "_routes.csv")
    print(
        f"""
=== DONE ===
  {prefix}_report.html      interactive report (open in any browser)
  {prefix}_branches.csv     per-branch measurements
  {prefix}_qc_coronal.png   ALWAYS review before trusting the numbers
  {prefix}_qc_axial.png     the red mask must be airway lumen only
  NIfTI masks (for 3D Slicer): {os.path.join(work, 'out')}
"""
    )


EXPLORATORY_STEPS = [
    ("leak_qc.py", "airway-mask leak / connectivity QC (exploratory)"),
    ("flow.py", "1D airflow model (exploratory)"),
    ("morphomap.py", "multi-axis structural map (exploratory)"),
    ("uncertainty.py", "robustness ensemble (exploratory)"),
    ("dual_viz.py", "airway-vascular discordance page"),
    ("av_viz.py", "artery/vein page"),
    ("flow_viz.py", "flow model page"),
]


def _exploratory_and_report(prefix, name, run_env):
    """Nel work dir (cwd): esegue gli step esplorativi standalone e assembla il
    report unico a schede. Riusa gli output della pipeline gia' presenti in out/.
    Ritorna il percorso del report unico o None se non prodotto."""
    env = dict(run_env)
    env["AIRWAYLAB_CASE"] = name
    for script, desc in EXPLORATORY_STEPS:
        print(f"\n=== {script} — {desc} ===")
        _step(script, [], env=env)
    # il report principale (out/report_main.html) e' l'input della prima scheda
    main_report = prefix + "_report.html"
    if os.path.exists(main_report):
        shutil.copy(main_report, os.path.join("out", "report_main.html"))
    print("\n=== report_unico.py — unified tabbed report ===")
    _step("report_unico.py", [], env=env)
    # guardia: etichettatura lobare completa? (altrimenti i per-lobo sono parziali)
    ti_path = os.path.join("out", "territory_index.json")
    if os.path.exists(ti_path):
        import sys as _sys
        _sys.path.insert(0, PIPE)
        from label_qc import labeling_status
        lobi = {v.get("lobe") for v in json.load(open(ti_path)).values()}
        st = labeling_status(lobi)
        if not st["complete"]:
            print("\n" + "*" * 70)
            print("*** ATTENZIONE: ETICHETTATURA LOBARE INCOMPLETA ***")
            print(f"    riconosciuti {st['n_present']}/{st['n_expected']} lobi; "
                  f"mancano: {', '.join(st['missing'])}")
            print("    I risultati PER-LOBO (mappa multi-asse, discordanza, vascolare)")
            print("    rappresentano solo una parte del polmone e NON sono affidabili.")
            print("    La segmentazione può essere corretta lo stesso: verifica il QC.")
            print("*" * 70)

    # guardia di plausibilita': 'completo' != 'corretto'. Anche con 6/6 lobi la
    # partizione puo' essere anatomicamente implausibile (es. lobo superiore dx allo
    # 0.2% del volume). Legge il controllo gia' calcolato in out/morphomap.json.
    mm_path = os.path.join("out", "morphomap.json")
    if os.path.exists(mm_path):
        try:
            plaus = json.load(open(mm_path)).get("plausibilita_lobare", {})
        except (ValueError, OSError):
            plaus = {}
        if plaus and not plaus.get("ok", True):
            print("\n" + "*" * 70)
            print("*** ATTENZIONE: PROPORZIONI LOBARI IMPLAUSIBILI ***")
            print("    l'etichettatura è completa (6/6) ma i volumi lobari non tornano:")
            for f in plaus.get("flags", []):
                print(f"    [{f.get('severity')}] {f.get('msg')}")
            print("    'completo' non vuol dire 'corretto': verifica la ripartizione")
            print("    dell'albero contro la segmentazione prima di usare i per-lobo.")
            print("*" * 70)

    # guardia di leak/connettivita' della segmentazione: vede i leak INTERNI
    # (cisti/bolle via radius-explosion) e le isole, che la vecchia guardia
    # "fuori dal polmone" non vedeva. Legge out/leak_qc.json gia' calcolato.
    lk_path = os.path.join("out", "leak_qc.json")
    if os.path.exists(lk_path):
        try:
            lk = json.load(open(lk_path))
        except (ValueError, OSError):
            lk = {}
        if lk and not lk.get("ok", True):
            print("\n" + "*" * 70)
            print("*** ATTENZIONE: POSSIBILI LEAK NELLA SEGMENTAZIONE DELLE VIE AEREE ***")
            for f in lk.get("flags", []):
                print(f"    [{f.get('severity')}] {f.get('msg')}")
            print("    Verifica la maschera (QC axial/coronal) prima di fidarti dei")
            print("    per-ramo: un leak in cisti/bolle o nell'esofago inquina l'albero.")
            print("*" * 70)

    src = os.path.join("out", "report_unico.html")
    if os.path.exists(src):
        dst = prefix + "_report_unico.html"
        shutil.copy(src, dst)
        return dst
    return None


def cmd_report(args):
    """DICOM -> report unico, in un solo comando. Idempotente: salta gli step
    i cui output esistono gia'. Anonimizza (scelta serie automatica), segmenta,
    esegue pipeline + analisi esplorative, assembla il report a schede."""
    outroot = os.path.abspath(args.outdir or os.getcwd())
    name = args.name
    os.makedirs(outroot, exist_ok=True)
    ct = os.path.join(outroot, name + ".nii.gz")
    segdir = os.path.join(outroot, name + "_seg")

    # 1) anonimizzazione con scelta serie automatica (o --series per forzare)
    if os.path.exists(ct):
        print(f"CT anonimizzata gia' presente: {ct} — salto anonimizzazione")
    else:
        extra = [str(args.series)] if args.series is not None else []
        _step("anonymize.py", [args.dicom_dir, ct, *extra])

    # 2) segmentazione DL delle vie aeree, dietro il backend sostituibile
    #    (scelta semplice e prevedibile; provenienza in segdir/backend_info.json).
    #    I vasi (arterie/vene) restano dal task lung_vessels di TotalSegmentator.
    sys.path.insert(0, PIPE)
    from airway_backend import segment_airways
    mask = segment_airways(ct, segdir, requested=getattr(args, "backend", None))

    # 3) pipeline completa (riusa cmd_run): backend DL, arterie/vene auto accanto al mask
    run_args = argparse.Namespace(
        ct=ct, mask=mask, name=name, outdir=outroot, spacing=args.spacing,
        biopsies=None, refine=False, arteries=None, veins=None)
    cmd_run(run_args)   # lascia cwd nel work dir e ricostruisce run_env internamente

    # 4) coda esplorativa + report unico (nel work dir dove cmd_run ci ha lasciati)
    work = os.path.join(outroot, name + "_work")
    os.chdir(work)
    prefix = os.path.join(outroot, name)
    # ricostruisci l'env come in cmd_run (arterie/vene per le pagine)
    run_env = {k: v for k, v in os.environ.items()
               if k not in ("AIRWAYLAB_ARTERIES", "AIRWAYLAB_VEINS")}
    _art = os.path.join(segdir, "lung_arteries.nii.gz")
    _vein = os.path.join(segdir, "lung_veins.nii.gz")
    if os.path.exists(_art):
        run_env["AIRWAYLAB_ARTERIES"] = _art
    if os.path.exists(_vein):
        run_env["AIRWAYLAB_VEINS"] = _vein
    report = _exploratory_and_report(prefix, name, run_env)
    print("\n=== REPORT PRONTO ===")
    print(f"  {report or '(report unico non prodotto)'}")


def cmd_anonymize(args):
    extra = [args.series] if args.series is not None else []
    _step("anonymize.py", [args.dicom_dir, args.output, *extra])


def main():
    p = argparse.ArgumentParser(
        prog="airwaylab",
        description="Transparent quantitative CT analysis of the airways. "
        "Research software — not a medical device.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("anonymize", help="DICOM folder/patient CD -> anonymous NIfTI")
    pa.add_argument("dicom_dir")
    pa.add_argument("output")
    pa.add_argument("series", nargs="?", default=None, help="series index (see printed list)")
    pa.set_defaults(func=cmd_anonymize)

    pr = sub.add_parser("run", help="full pipeline on an anonymized CT")
    pr.add_argument("ct", help="anonymized chest CT (.nii.gz)")
    pr.add_argument("--name", default=None, help="case identifier (default: file name)")
    pr.add_argument("--outdir", default=None, help="output directory (default: cwd)")
    pr.add_argument("--spacing", type=float, default=None,
                    help="isotropic resampling spacing in mm (default: finest native, floor 0.5)")
    pr.add_argument("--mask", default=None, metavar="AIRWAY_MASK",
                    help="externally produced airway mask (e.g. TotalSegmentator "
                    "lung_airways.nii.gz): skips region growing, enables the air "
                    "witness, resolution gates, tight small-airway window and the "
                    "refined centerline. Caliber is reported only where the mask "
                    "still carries size information (two-regime rule).")
    pr.add_argument("--biopsies", default=None, metavar="BIOPSIES_JSON",
                    help="biopsy sites to pre-load in the report (JSON exported "
                    "with the report's 'copia biopsie' button, or a bare list of "
                    "{rotta, s_mm} objects) — for baseline/follow-up site pairing")
    pr.add_argument("--refine", action="store_true",
                    help="sub-voxel centerline refinement + B-spline also with the "
                    "built-in segmentation (default off; always on with --mask)")
    pr.add_argument("--arteries", default=None, metavar="ARTERY_MASK",
                    help="pulmonary artery mask (TotalSegmentator lung_arteries.nii.gz); "
                    "default: looked up next to --mask. Enables the exploratory "
                    "artery/vein analysis (pruning, A/V).")
    pr.add_argument("--veins", default=None, metavar="VEIN_MASK",
                    help="pulmonary vein mask (lung_veins.nii.gz); default next to --mask")
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser(
        "report",
        help="un solo comando: cartella DICOM -> report unico (anonimizza con "
        "scelta serie automatica, segmenta con TotalSegmentator, esegue pipeline "
        "e analisi esplorative, assembla il report a schede). Idempotente.")
    prep.add_argument("dicom_dir", help="cartella DICOM del paziente (anche CD/PACS)")
    prep.add_argument("--name", required=True, help="ID neutro del caso (es. caso06)")
    prep.add_argument("--outdir", default=None,
                      help="cartella di lavoro/output (default: cwd). CT anonimizzata, "
                      "maschere e report finiscono qui.")
    prep.add_argument("--series", type=int, default=None,
                      help="forza l'indice di serie (default: scelta automatica della "
                      "ricostruzione sottile piu' adatta)")
    prep.add_argument("--backend", default=None,
                      help="backend di segmentazione delle vie aeree da usare "
                      "(default: scelta automatica semplice e prevedibile; attualmente "
                      "'totalsegmentator'). La provenienza finisce in "
                      "<caso>_seg/backend_info.json.")
    prep.add_argument("--spacing", type=float, default=None,
                      help="spacing isotropico di ricampionamento in mm")
    prep.set_defaults(func=cmd_report)

    pv = sub.add_parser("version", help="print version")
    pv.set_defaults(func=lambda a: print(f"AirwayLab v{__version__}"))

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
