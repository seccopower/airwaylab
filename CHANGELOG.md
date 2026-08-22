# Changelog

## v1.0.0 — clean baseline

First consolidated release of AirwayLab as a single, reproducible research
pipeline. It supersedes the internal prototype iterations (developed and
refined against anonymized chest CTs, with every measurement change reviewed
against images by a radiologist) and starts the versioned history from a clean
baseline.

### Airway tree
- Two segmentation backends: adaptive explosion-controlled region growing
  (built-in), or an external / deep-learning airway mask via `--mask`
  (e.g. TotalSegmentator `lung_vessels`).
- Skeletonization, branch graph, generations, and best-effort anatomical
  labels (lobar bronchi, B1–B10), with stable `aid` codes decoupled from the
  Italian display labels.
- Sub-voxel centerline refinement + B-spline (`--refine`, automatic with
  `--mask`): removes the voxel staircase and de-inflates branch lengths; used
  for the measurement sections themselves.

### Measurement, with an honesty regime
- Lumen caliber by half-max radial boundary on all core sections (~1 mm
  spacing), area-equivalent diameter, per-branch QC with attempted/accepted
  section counts.
- Sector-wise FWHM wall thickness with a positive-parenchyma requirement;
  WA% defined only where wall is measurable; the physiological cap flags,
  it does not censor.
- **Two-regime rule** (enforced end-to-end): with a deep mask, branches below
  the orientation-dependent resolution floor are kept for topology and
  territories but report **no caliber** (`sotto-risoluzione`); an air witness
  rejects mask branches with no visible lumen in the CT; an escape gate
  (`fuga-contorno`) discards half-max values that overshoot the mask diameter.
  Demoted branches are nulled in the clinical channel of every product (CSV,
  profiles, CPR) and preserved only under explicit `*_raw_nonreportable`
  columns; the HTML report renders them dotted grey and labelled NON
  REPORTABILE. The witness thresholds and the resolution floor are documented
  as provisional processing bounds, not validated physical limits.

### Beyond caliber (exploratory)
- Lung volume, mean lung density, LAA-950, Perc15; dysanapsis; a multi-airway
  airway-to-lung index and ALR4.
- Parenchymal territories per terminal branch and a caliber–territory scaling
  fit.
- Vascular segmentation (TBV, BV5/BV10), bronchus–artery pairing (BA ratio),
  airway–vascular mismatch map.
- Mucus-plug candidate detector (propose-and-confirm, territory-weighted).
- Straightened CPR panels with synchronized image + longitudinal profiles.
- A standalone, exploratory 1D air-flow model on the measured tree
  (`flow.py` / `flow_viz.py`): a series/parallel resistance network solved to
  convergence, kept off the standard report and clearly marked as a
  simulation, not a measurement.

### Reporting and reproducibility
- One interactive self-contained HTML report per case; per-branch CSV; coronal
  and axial QC images (always review before trusting the numbers).
- One-command entry point `auto_report.py`: DICOM folder → automatic series
  selection → anonymization → segmentation → full pipeline → report.
- Run provenance (backend, refinement, version) recorded in `seg_info.json`;
  per-run isolated subprocess environments.
- Apache-2.0; CI runs the synthetic-phantom test suite on every push.

> **Research software.** AirwayLab is not a medical device and must not be used
> for clinical decision-making. Exploratory indices are research hypotheses and
> must not be interpreted as validated measurements. See
> [`docs/VALIDATION_BACKLOG.md`](docs/VALIDATION_BACKLOG.md) for the open
> validation program.
