# Changelog

## Non rilasciato

- **Guardia di geometria in `anonymize.py`** (`geometry_qc_core.py`, nucleo puro).
  Un CD paziente conteneva ogni fetta duplicata — 834 file su 417 posizioni z,
  pixel identici, solo il SOPInstanceUID diverso: il reader ordinava le fette,
  meta' delle differenze consecutive valeva 0 e lo spacing crollava a 0.08 mm
  contro il millimetro dichiarato. Il volume veniva scritto senza un errore e la
  pipeline avrebbe misurato calibri in mm sbagliati di un fattore ~12. Ora le
  posizioni ripetute sono rilevate e deduplicate (riordinando per z), e la
  geometria del volume prodotto e' verificata: spacing, copertura e coerenza fra
  spessore DICHIARATO e MISURATO. Soglie larghe, ancorate all'acquisizione.
- **Cache delle maschere verificata per griglia** (`airway_backend.py`). Il salto
  di TotalSegmentator si basava sulla sola presenza dei file: maschere di una
  conversione precedente venivano riusate su una CT diversa e risultavano vuote,
  con l'errore che emergeva solo piu' a valle in `extmask.py`. Ora dimensioni e
  spacing devono combaciare con la CT corrente, altrimenti si rigenera. Il
  controllo segnala solo disallineamenti OSSERVATI: se un'intestazione non e'
  leggibile non blocca.
- **Air-witness dipendente dal calibro** (backlog #29). La soglia fissa
  `HU_AIR = -750` era cieca al calibro: il volume parziale alza l'attenuazione
  misurata nei lumi piccoli, e i rami scartati come `no-lume` erano i piu'
  piccoli dell'albero (d_maschera mediana 1.40 mm, hu_lume -680, contro 2.80 mm
  e -990 degli accettati). Il criterio sulla mediana ora segue la curva del
  volume parziale, limitato in modo da non essere mai piu' severo del gate
  storico ne' piu' permissivo di un tetto. Nuova colonna `soglia_aria_hu` con la
  soglia effettiva per ramo. Effetto misurato: 9 rami su 17 recuperati.
- **Leggibilita' della parete al limite di risoluzione** (backlog #28). Sotto un
  certo calibro il tetto fisiologico sta SOTTO il minimo che la FWHM puo'
  restituire, quindi `parete_oltre_cap_pct` al 95% segnalava il limite del metodo
  leggendosi come ispessimento diffuso. Nuove colonne `floor_parete_mm`,
  `parete_al_floor` e `cap_sotto_floor` rendono la distinzione esplicita.
  **Nessuna censura**: la demozione resta subordinata allo sweep del #27, perche'
  la costante non e' calibrata e l'esito e' ipersensibile (0%, 62% o 88% di
  pareti demolite a seconda del valore scelto).
- **Floor di risoluzione misurato invece che assunto** (backlog #27).
  `phantom_core.py` costruisce un tubo digitale seguendo la catena reale di
  formazione dell'immagine (oggetto continuo -> PSF -> integrazione del voxel ->
  rumore) e `tools/floor_sweep.py` fa scendere il diametro verso il voxel
  misurando con lo stimatore vero della pipeline. Il floor e' definito sulla
  RISPOSTA (pendenza, precisione, misurabilita'), non sul bias, che e' un offset
  sistematico calibrabile. Risultato: floor mediano 3.5 voxel, intervallo
  2.5-5.0, dominato dal KERNEL piu' che dallo spacing. **Il guadagno atteso non
  esiste**: in 0 configurazioni su 8 un limite a 2 voxel sarebbe sufficiente, e
  `VOXELS_FLOOR = 3.0` risulta semmai leggermente permissivo. Nessun parametro
  e' stato cambiato: serve prima decidere cosa fare dei valori gia' pubblicati.
- **210 test** (da 162).
- **`VALIDATION_BACKLOG` sezione F: dove la resa si perde davvero.** Quattro
  perdite quantificate su un caso reale sottile (218 rami) invece che elencate
  per plausibilita': il floor di risoluzione (#27, 29% -> 66% potenziale), la
  parete non informativa sotto ~3.7 mm (#28), l'air-witness cieco al calibro
  (#29), meta' dei settori di parete scartati (#30), piu' la sostituzione della
  FWHM (#31). Vincolo esplicito: nessun fantoccio fisico disponibile, quindi
  ogni item e' progettato per essere risolvibile in silico o per auto-riferimento
  sui dati gia' in mano.
- **README: sezione sul kernel di ricostruzione.** Densitometria e morfometria
  delle vie aeree vogliono kernel opposti (il rumore di un kernel sharp spinge
  parenchima normale sotto la soglia dei -950 HU e gonfia il LAA-950; un kernel
  morbido sfuma il bordo di parete e fa sovrastimare la FWHM). Documentata la
  soluzione pulita — ricostruire due volte la stessa acquisizione — e reso
  esplicito il peso dello spessore di strato sul floor di risoluzione. Voce
  corrispondente in `docs/DIZIONARIO_PARAMETRI.md`, sezione 0, con l'elenco
  dei parametri spostati in ciascuna direzione.

## 1.1.0 — exploratory descriptors + audit-driven honesty hardening

Additive on the 1.0.0 baseline (no breaking changes to the caliber/wall
endpoints). Two threads: more descriptors extracted from a single CT, and an
adversarial code audit that made sure every one of them says only what the
method measures.

- **New per-subject descriptors** (all exploratory), each with a pure tested
  core: **Pi10_AirwayLab** (√WA at a fixed 10 mm perimeter), **tapering**
  (child/parent ratio + log-diameter gradient), **tree morphometry** (counts +
  `afd_3d_fixedgrid`), **parenchyma beyond mean density** (histogram shape,
  regional heterogeneity, LAA-cluster D), **vascular pruning gradient**
  (small-structure density vs distance to pleura), and opt-in **opportunistic
  body composition** (vertebral HU, muscle, fat). Surfaced in a new
  **"Sintesi quantitativa"** report tab.
- **Provenance on every descriptor** — each JSON stamps `status='exploratory'`,
  `method_id`, parameters, denominators, exclusions, backend and tool version
  (`provenance.py`); the summary tab opens with an exploratory banner and a
  provenance footer, so the report never presents a number without its caveats.
- **Pi10 diagnostics** — observed perimeter range, an explicit extrapolation
  flag when 10 mm falls outside it, coverage, a deterministic bootstrap 95% CI
  and leave-one-out sensitivity; the tile turns amber when extrapolated.
- **Honest relabels** — labels reworded to match the method: `Pi10_AirwayLab`
  (not "normalizes WA%"), `afd_3d_fixedgrid` (not FracLac), opening-residual
  proxy (not scale-space BV5), dense-structure mask (not pure vessel), aerated
  component (not anatomical lung); body-composition bone flag retired,
  `torso_fat` reported as `internal_fat` (not VAT), no "sarcopenia" claim.
- **Known-limitations section** — `docs/VALIDATION_BACKLOG.md` section E gathers
  the deep methodological limits of the new descriptors and the minimum
  conditions before a methods paper.
- **Flow-report panels** — the exploratory flow tab replaces the
  per-generation Plotly bar with four pure-CSS/SVG panels driven by
  `flow.json`: pressure drop per lobe, measured-vs-imputed dissipation split,
  specific-ventilation histogram, and mucus-plug occlusion before/after (with
  a graceful empty state when no plugs). 3D tree kept; theme-aware.
- **162 unit tests** (from 122).

External dependency unchanged: TotalSegmentator (tested with v2.18.0).

## 1.0.0 — baseline release (frozen reference)

First stable, citable baseline: the reference every future change builds on.
Everything from the prototype phase below, plus, in this cycle:

- **One-command workflow** — `airwaylab report <dicom> --name <case>`: anonymise
  (automatic series selection) → segment → full pipeline → exploratory analyses
  → single tabbed HTML report. Idempotent.
- **Three QC safety nets**, shown in the report and printed to the terminal:
  lobe-labeling completeness; lobe-proportion plausibility ("complete ≠
  correct" — catches anatomically implausible partitions the presence guard
  cannot see); and airway-mask **leak / connectivity** QC (radius-explosion +
  disconnected islands — sees leaks *inside* the lung, in cysts/bullae, that an
  "outside-the-lung-mask" check misses).
- **Swappable segmentation backend** behind a single seam, with recorded
  provenance (`backend_info.json`: backend, version, mask checksum).
  TotalSegmentator is the default; an `aeropath_onnx` adapter consumes a
  user-supplied AeroPath mask (weights not bundled — licence unstated upstream).
- **Robust anatomical main-bronchus finding** — structural (subtree side-purity
  + descend-to-split), resistant to spurious skeleton stubs and to generation
  inflation that broke the previous generation-1 assumption.
- **122 unit tests.**

External dependency: TotalSegmentator (tested with v2.18.0), installed
separately; its version is recorded per run in `backend_info.json`.

## Prototype phase (v0.1–v0.25)

AirwayLab grew through 20 internal iterations on two real (anonymized) chest
CTs and one public test volume, with every measurement change reviewed against
images by a radiologist. Highlights of the prototype phase (v0.1–v0.20,
August 2026), condensed:

- **v0.1–v0.4** — airway segmentation (explosion-controlled region growing with
  anatomical seed scoring), skeleton/graph/generations, first measurements,
  single-command runner, DICOM/patient-CD anonymizer.
- **v0.5–v0.7** — half-max lumen boundary on recentered perpendicular sections
  (replacing mask-edge diameters after radiologist review); PCA tangents;
  per-branch verification snapshots; automatic QC classes + interactive
  reader exclusions with exportable audit list.
- **v0.8–v0.10** — sector-wise wall thickness on the same sections; double
  contour (lumen + outer wall in valid sectors); physiological cap and
  circular-coherence filter; positive parenchyma requirement; wall chart.
- **v0.11–v0.13** — longitudinal caliber/wall profiles; lung segmentation and
  densitometry; dysanapsis; parenchymal territories and Murray-law fit;
  written measurement protocol.
- **v0.14–v0.16** — vascular segmentation (TBV, BV5), vascular graph,
  bronchus-artery pairing (BA ratio), airway-vascular mismatch map; mucus plug
  candidate detector (propose-and-confirm, territory-weighted); ALR4 per
  Shimada 2025; straightened CPR with synchronized image + profile panel.
- **v0.21** — first packaged release: `airwaylab` CLI, Apache-2.0, tests on a
  synthetic tube phantom, CI, English documentation.
- **v0.22** — hardening pass following an independent external adversarial review:
  LPS orientation canonicalization on input (with RAS regression test); stable
  anatomical ids (`anatomy.py`) decoupled from display labels, with ALR4
  producibility test; full geometry (origin/direction) on downsampled masks;
  diagnostic `QualityError` guards on empty/implausible masks; soft-fail
  labeling; measurement code deduplicated into `lumen.py`; configurable
  resampling spacing with a memory guard (`--spacing` / `AIRWAYLAB_SPACING`);
  dependency version bounds; HTML-escaped case names; `AGENTS.md` with
  four-member team rules (maintainer, architect, adversarial reviewer, local
  operator).
- **v0.25.2 (same PR, third review round)** — the remaining floor blocker:
  the per-axis projection formula max(s_i·sqrt(1−t_i²)) underestimated the
  worst in-plane resolution on oblique planes (~13% at 45° z-y on
  0.72/0.72/1.25 mm); replaced with the largest eigenvalue of the projected
  anisotropic sampling metric P·M·P (P = I − t·tᵀ, M = diag(s²)) — the true
  worst direction, which mixes axes. Branch-level floors now aggregate LOCAL
  tangents sampled along the path (max of per-section floors) instead of the
  global chord, so curved branches are floored by their worst-oriented
  segment. CPR branch boundaries use half-open intervals (bifurcation sample
  belongs to the starting branch). New tests: oblique worst-plane value
  (explicitly rejecting the per-axis formula), curved branch vs chord,
  split_reportable invariant (unit-level), 23-column cardinality, provenance
  keys guard (source-level; synthetic end-to-end coverage tracked in the
  test-coverage issue). OBSERVED EFFECT on the single development case
  (caso02, DL mask, 0.72x0.72x1.25 mm): with the corrected orientation-
  dependent floor, 91 of 451 branches (~20%) remain caliber-reportable;
  350 are `sotto-risoluzione`, 9 `fuga-contorno`, 1 `no-lume`. This is the
  observed behaviour of a still-provisional gate on one exam — not a
  validation result; per-branch floors on this acquisition span
  2.17-3.75 mm (median 3.35).
- **v0.25.1 (same PR, second review round)** — the reviewer's five blockers:
  demotion now propagates to EVERY product (profiles.json and cpr.json null
  the clinical d/w channel for demoted branches/route segments and carry the
  values only under `*_raw_nonreportable`, with a per-sample `reportable`
  mask in CPR); the HTML report renders non-reportable curves dotted grey
  with explicit NON REPORTABILE labels and hovers (no raw value under a
  normal label anywhere); the caliber floor is orientation-dependent — the
  coarsest native spacing projected into each branch's section plane
  (in-plane branches on 1.25 mm slices are floored at 3.75 mm, axial ones at
  2.17 mm; conservative worst-axis fallback without orientation; per-branch
  `floor_calibro_mm` exported); the plug-detector mask-diameter fallback is
  restricted to `sotto-risoluzione` (never `no-lume`/`fuga-contorno`); audit
  completeness — `d_min_raw`/`wa_raw` join the CSV as explicit
  `*_raw_nonreportable` columns and `d_min_hm`/`d_max_hm` are preserved as
  raw instead of dropped; provenance is written right after segmentation so
  `map_data.json`/HTML expose backend, refinement and version.
- **v0.25** — two-regime rule ENFORCED (response to third adversarial review,
  13 points, pre-release this time). The v0.24 release *declared* that demoted
  branches report no caliber but still exported the values: now `d_mean`,
  `d_min`, `wall`, `wa_pct` are nulled for `no-lume`/`sotto-risoluzione`/
  `fuga-contorno` branches, with originals preserved in explicit
  `*_raw_nonreportable` audit columns. One unified per-branch CSV schema with
  and without `--mask` (witness columns empty on built-in runs). Witness now
  samples HU/EDT tri-linearly at the sub-voxel centerline (no grid-phase
  dependence) and its thresholds are documented as PROVISIONAL, single-case
  development values (`qc_params.py`; ROC validation in the backlog). The
  caliber floor is anchored to the native in-plane spacing, so upsampling
  cannot loosen it (it remains a processing bound, not a validated physical
  resolution limit; slice thickness not yet accounted). CPR and the mucus-plug
  detector now use the refined centerline too (v0.24 left them on the raw
  skeleton — undeclared inconsistency, fixed); plug screening falls back to
  the mask diameter for sub-resolution leaves. Snapshots and profiles of
  demoted branches remain visible BY DESIGN (every branch ships its image,
  including excluded ones) but carry a NON REPORTABILE badge/flag. Run
  provenance (`backend`, `refined_centerline`, `tight_small_window`, version)
  recorded in `seg_info.json`; per-run subprocess environments (no flag
  leakage between programmatic runs). Effects of the refined centerline on
  branch length (−12% staircase de-inflation) propagate to: stump QC (3 mm),
  CSV lengths, report geometry, section positions/tangents/counts, profile
  axes, witness sampling — and indirectly to dysanapsis/ALR4, caliber-
  territory fit and BA ratio via diameters and QC; topology, generations and
  labels are built before refinement and unchanged. Tests: witness gates
  exercised via the production module with borderline cases (median ±1 HU,
  air fraction 59/61%, bimodal half-mucus lumen); curved-tube phantom with
  known arc length (chord explicitly rejected as truth); floor anchoring.
- **v0.24** — deep-learning segmentation backend (radiologist-validated on
  caso02, 451 branches vs 97): `airwaylab run --mask` accepts an externally
  produced airway mask (e.g. TotalSegmentator `lung_airways`) and switches on
  three new defenses. (1) *Air witness*: every branch is verified against the
  CT itself — median centerline HU and air fraction — so a generous DL mask
  cannot introduce airways with no visible lumen (caso02: 1/451 rejected).
  (2) *Two-regime rule*: branches whose mask diameter is below the resolution
  floor (~3 voxels) are kept for topology/territories but report NO caliber
  (`sotto-risoluzione`); an escape gate (`fuga-contorno`) discards half-max
  values that overshoot the mask diameter. (3) *Tight small-airway search
  window*: prevents the half-max rays from skipping thin walls onto nearby
  structures (median caliber at generation 8-9 dropped from ~5.5 to ~4 mm on
  the DL mask). Plus: sub-voxel centerline refinement + B-spline (`--refine`;
  automatic with `--mask`) — recentering on the lumen centroid removes the
  voxel staircase and skeleton wander, de-inflating branch lengths by ~12%
  (caso02 trachea: raw path 137 mm vs 115 mm chord; spline 118 mm) and used
  for the measurement sections themselves; 3D-map polylines smoothed
  (cosmetic fallback when refinement is off); per-branch witness columns
  (`hu_lume`, `aria_pct`, `d_maschera_mm`, `qc_misura`) in the CSV; tests for
  recentering, spline de-inflation and witness thresholds.
- **v0.23** — response to second adversarial review (25 points, triaged in
  `docs/VALIDATION_BACKLOG.md`): caliber now measured on ALL core sections at
  ~1 mm spacing with attempted/accepted counts (removes the representative-
  section bias that discarded stenoses/dilatations); the physiological wall
  cap no longer censors measurements — it is a QC flag (% sectors above cap
  reported); d_min/d_max from measured sections; README endpoint table gains
  core/exploratory status; "every number ships with its image" claim made
  precise; WA% formula documented.
