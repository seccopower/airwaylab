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
