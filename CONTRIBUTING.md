# Contributing to AirwayLab

AirwayLab is research software for quantitative CT analysis of the airways.
These are the working rules for anyone contributing code, documentation, or
methods. The clinical lead has final say on every merge and owns all
clinically validated thresholds.

## Hard rules

1. **Never commit patient data.** No DICOM, no NIfTI, no screenshots containing
   patient-identifiable content. `.gitignore` already excludes `*.nii(.gz)` and
   `out/` — do not weaken it. If you find patient data in the tree, stop and
   flag it before doing anything else.
2. **Do not change clinically validated thresholds** (HU thresholds, QC limits
   such as `MAX_AX_RATIO`, wall caps, ALR/dysanapsis definitions) without an
   explicit request from the clinical lead. These were validated against images
   by a radiologist; they are data, not style.
3. **Match on stable ids, not display strings.** Anatomical identity lives in
   `airwaylab/pipeline/anatomy.py` (`aid` codes: `TRACHEA`, `RMB`, `B6_R`, …).
   Display labels are Italian and may change; `aid` is the contract.
4. **Run `pytest` before proposing any change.** If you add behavior, add a
   test; the synthetic tube phantom in `tests/` is the pattern (no patient data
   needed).
5. **Minimal diffs.** Do not reformat, rename, or restyle files you are not
   otherwise changing. One task, one concern, one branch.
6. **Method-touching changes are reviewed before merge.** Any change that
   affects a reported number is reviewed against images before it lands.
7. **Transparency principle.** Every measurement this tool reports must remain
   verifiable against an image by the reader. Do not add numbers to the report
   without their visual evidence or their QC status.

## Working agreement

- Work on a feature branch (`feat/...`, `fix/...`), never directly on `main`.
- The pipeline scripts under `airwaylab/pipeline/` communicate via files in the
  case work directory (`out/`); they are executed by `airwaylab/cli.py` in the
  order defined there. If you add a stage, register it in `cli.py` and document
  it in the README table.
- Known architectural debt (script-at-import style, `out/` file coupling) is
  tracked for refactoring; do not "fix" it piecemeal inside unrelated tasks.
- Language: code and docs in English; report UI strings in Italian (clinical
  users). The measurement protocol in `docs/` is normative for methods.

## Safety scope

Work **only inside this repository folder**. Patient folders, DICOM archives,
and anonymization staging areas are out of scope even when technically
reachable. Anonymization always runs locally: only voxel + geometry NIfTI, with
neutral case ids, ever leaves the source machine.
