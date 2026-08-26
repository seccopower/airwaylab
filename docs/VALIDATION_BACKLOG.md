# Validation backlog

The open methodological and validation program for AirwayLab, grouped by
priority. Items marked **addressed** are implemented in the current baseline;
**open** items are the work still required before any endpoint can be described
as validated. Convert each item to an issue as it is picked up.

## A. Method integrity

1. **Full-profile caliber** *(addressed)* — caliber is measured on all core
   sections at ~1 mm spacing (no representative-section preselection, which
   would discard stenoses and dilatations); per-branch output reports sections
   attempted/accepted; d_min/d_max come from the measured sections.

2. **Wall cap does not censor** *(addressed)* — the physiological wall cap
   (0.18·d + 0.6 mm) is a QC flag only (fraction of sectors above cap reported
   per branch), not a censoring rule. Vessel/mediastinum exclusion relies on the
   parenchyma requirement and the circular-coherence filter.

3. **Sector validity and non-random missingness** *(open)* — declare the
   estimand explicitly as "wall in parenchyma-facing sectors"; report coverage
   (already partly done) and add the angular-distribution requirement plus
   stratified analysis.

4. **Metrological validation of half-max lumen and oblique QC** *(open)* —
   physical phantom with known diameters/walls/angles/kernels/doses; ROC for the
   axial-ratio threshold. Needs scanner time (clinical lead).

5. **Method stratification for fallback diameters** *(open)* — never aggregate
   mask-fallback diameters and half-max values in one endpoint; the per-branch
   method column exists, the analysis-side rule must be documented.

## B. Definitions and documentation

6. **Protocol/version alignment** *(in progress)* — rewrite the measurement
   protocol to the current version with an endpoint → module → version →
   validation-status matrix; vessels/ALR4/plugs/mismatch must be defined there.

7. **WA% definition** *(in progress)* — WA% = 100·((d/2 + w)² − (d/2)²)/(d/2 + w)²,
   with d = area-equivalent lumen diameter (half-max), w = median wall of valid
   sectors; defined only when wall is reported; no extrapolation to full
   circumference. To protocol + README.

8. **Dysanapsis / ALR4 definitions** *(in progress)* — fix units and airway sets
   in the protocol to match the code (aid-based ALR4); rename the multi-airway
   index to avoid claiming identity with published definitions; remove silent
   fallbacks or report them as flags.

9. **Exploratory framing of Murray / territories / plugs / BA / mismatch**
   *(wording addressed; validation open)* — these are exploratory indices, not
   verified physical laws; the methods paper prespecifies primary endpoints
   (caliber; lung volume) and labels the rest exploratory.

10. **"Every number ships with its image" precision** *(wording addressed;
    UI open)* — the claim is restricted to per-branch caliber/wall; aggregate
    indices link to their inputs (accepted/attempted sections reported). A full
    per-section drill-down UI is still open.

## C. Validation program (needs clinical lead + scanner)

11. **Independent validation study** *(open)* — multi-scanner cases, normal and
    pathological, reference standard, multiple readers, reader-agreement
    statistics, reading manual, coded exclusion reasons; designed as a separate
    protocol.

12. **Threshold provenance + sensitivity analysis** *(open)* — for every
    heuristic parameter: source, rationale, sensitivity sweep. Start with the
    ones that move endpoints most (growing thresholds, half-max jump,
    axial-ratio, parenchyma −650 HU).

13. **Resampling / PSF honesty** *(open)* — express filters in mm, document that
    resampling adds no resolution, add invariance tests across spacing and
    orientation.

14. **Input requirements & acquisition QC** *(open)* — DICOM requirement table,
    automatic acquisition checks, predefined exclusion criteria.

## D. Engineering / reproducibility

15. **Reproducible environment** *(open)* — lockfile, container, environment
    manifest, input checksums, two-system reproducibility test.

16. **Test coverage per endpoint** *(open, top of the engineering queue)* —
    end-to-end synthetic phantom (branching tree, known geometry) exercising
    segmentation → graph → measures → territories; numeric regression tests per
    endpoint; failure-case tests.

17. **Audit-trail operationalization** *(open)* — exclusions persisted with
    timestamp / version / input-hash and coded reasons.

18. **Per-endpoint limits section** *(in progress)* — reorganize the limits by
    endpoint, with expected bias direction and failure conditions.

## E. New quantitative descriptors (added 2026-08 — all exploratory)

Following an adversarial code audit, a group of per-subject descriptors was
added (Pi10, tapering, tree morphometry/AFD, parenchyma, vascular pruning
gradient, opportunistic body composition). They are **exploratory descriptors,
not validated endpoints**. The audit's verdict — "recoverable as an exploratory
platform" — sets the bar for this section: none of these may be described as a
biological endpoint or a validated measure until the corresponding item is
closed. Provenance and framing are already shipped; the metrological work is
open.

19. **Exploratory status + provenance on every descriptor** *(addressed)* —
    each descriptor JSON stamps `status='exploratory'`, `method_id`, parameters,
    denominators and exclusions, plus backend and tool version (`provenance.py`);
    the summary page opens with an exploratory banner and a provenance footer.
    The report no longer presents these numbers without their caveats.

20. **Pi10 metrological validation** *(open; diagnostics addressed)* —
    `Pi10_AirwayLab` is our implementation of the Nakano regression, renamed to
    stop implying identity with the published, validated Pi10. Shipped
    diagnostics: observed perimeter range, an explicit **extrapolation flag**
    when 10 mm falls outside that range, target coverage, a deterministic
    bootstrap 95% CI and leave-one-out sensitivity. Still open: circular-
    perimeter bias; half-max wall overestimation on small airways; agreement
    against a reference Pi10 pipeline. Ties to phantom item #4.

21. **Tapering estimand** *(open)* — the child/parent ratio is computed on
    **topological parent-child adjacency** (mono-child chains included, not only
    true bifurcations); the gradient uses cumulative length **from the root** to
    the branch midpoint. The 0.79 (`2^(-1/3)`) and 0.9 values are a theoretical
    reference and an operating threshold, **not** clinical normality cut-offs;
    cohort references and sensitivity to spurious branches are open.

22. **Counts / AFD** *(open; denominators addressed)* — counts include **all
    graph branches** (no-lumen / sub-resolution included); `n_branches_grafo` and
    `n_branches_qc_ok` are now exported separately so the count is not read as
    "visible airways". `afd_3d_fixedgrid` uses a single grid origin and few
    scales — it is **not** the multi-position FracLac standard; grid-origin /
    scale sensitivity and agreement against the standard are required before any
    AFD claim.

23. **Parenchyma estimand** *(open)* — MLD / LAA-950 / Perc15 / heterogeneity /
    cluster-D are computed on the **aerated component** (air threshold of
    `lung.py`), **not** the anatomical lung: dense consolidation / effusion /
    atelectasis are excluded by construction. Declare this estimand in the
    protocol. Heterogeneity is regional density SD, **not** validated mosaic /
    air-trapping (needs the expiratory scan). Cluster-D uses OLS rank-size, not
    MLE/x_min; the ×3 subsample alters cluster topology; not enfisema-specific
    (Gupta found no asthma difference).

24. **Vascular pruning proxy** *(open)* — "small structures" = morphological
    **opening residual at r ≈ 1.26 mm** on a dense-structure candidate-vessel
    mask; this is a proxy, **not** the scale-space BV5 (Estépar) and **not** the
    orthogonal vessel calibre. Non-contrast CT does not isolate vessels. The ×3
    distance field aliases; `pruning_ratio` is exploratory. Validate against a
    real BV5 pipeline and a contrast reference before any pruning claim.

25. **Body-composition ROIs** *(open)* — bone: whole-vertebra ROI eroded by
    2 vox at **mixed thoracic levels**, not the Pickhardt trabecular ROI; the
    ~110 HU threshold is not validated for this ROI, so the automatic low-bone
    flag is **retired** (value kept, alarm removed). Muscle: not the L3
    single-slice / EWGSOP2 index → not "sarcopenia". Fat: TotalSegmentator
    `torso_fat` is not segmented VAT → reported as `internal_fat_ml`, not
    `vat_ml`. Each needs the standard ROI / definition before any clinical read.

26. **Minimum conditions before a methods paper** *(open — consolidates the
    audit's checklist)* — (a) prespecify primary endpoints (caliber; lung
    volume) and label every descriptor above exploratory; (b) for each
    descriptor, state estimand, denominators and exclusions (provenance shipped,
    protocol text open); (c) validate Pi10 / AFD / BV5 against their published
    definitions **or** keep the `_AirwayLab` / proxy naming (naming done,
    validation open); (d) protocol-invariance tests across spacing / orientation
    / kernel (ties #13); (e) source + sensitivity sweep for every heuristic
    threshold (ties #12); (f) reference-standard study for anything claimed
    diagnostic (ties #11); (g) reproducible environment + per-endpoint
    regression tests (ties #15, #16).
