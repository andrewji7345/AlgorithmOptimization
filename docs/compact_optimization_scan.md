# Compact expanded optimization scan

This ntuplizer implements the expanded reconstruction family while keeping
event gates factorized from reconstruction. The [compact evaluator](compact_scan_evaluation.md) uses these files, which contain the event-level quantities needed to
measure gate efficiency, reconstruction efficiency conditional on the gate,
end-to-end retention, physicality, and later signal sensitivity without
re-running MiniAOD.

## Reconstruction and reuse

One job covers one signal sample and one anti-kT radius. For every event it:

1. multiplies all four components of each finite, positive-weight
   `PackedCandidate` by `puppiWeight()` exactly once;
2. clusters those weighted particles into uncorrected anti-kT jets once;
3. stores their pT values in descending order so every `(n, T_gate)` decision
   can be made offline;
4. selects all AK jets above each `T_keep` and caches thresholds that select the
   same AK-jet prefix;
5. boosts the selected AK constituents into their combined COM frame and finds
   the thrust axis once per distinct retained system;
6. clusters those boosted constituents with longitudinally invariant
   Cambridge/Aachen for each CA radius; and
7. evaluates the hybrid CA-to-SJ assignments, caching cuts that produce the
   same fixed/ambiguous labels.

CA clustering is therefore in the selected system's COM frame, not the lab
frame. It is the y-phi Cambridge/Aachen algorithm used by the existing study;
the spherical `ee_genkt` alternative remains outside this scan. No JEC is
applied because arbitrary AK radii do not yet have a consistent correction
set.

The hybrid cut `c` has explicit endpoint behavior:

- `c=0`: pure thrust; nonnegative cosine goes to SJ 1 and negative cosine to
  SJ 2;
- `0<c<1`: cosine at least `c` or at most `-c` fixes a CA jet to a thrust side,
  while `abs(cosine)<c` is assigned by exhaustive mass balancing;
- `c=1`: pure mass balancing, with every CA jet ambiguous.

The exhaustive stage minimizes the bounded symmetric objective
`abs(m1-m2)/(m1+m2)`. Both sides must be nonempty and numerical. A cut with
more than 12 ambiguous CA jets receives `complexity_guard` for only that
event/configuration; the event and all other configurations remain in the
file. Pure mass is supported but excluded from the production cut grid.

## Compact ROOT schema

With the cfg module label `compactScan`, each file has:

- `compactScan/Metadata`: one entry containing the schema version, sample,
  algorithm choices, grids, base/config index mapping, status dictionary,
  event count, and generator-weight sums;
- `compactScan/Events`: one entry per input event with `run`, `lumi`, `event`,
  `genWeight`, sorted `akJetPt`, base-level `nCAJets`, and flat per-config
  `recoStatus`, `nAmbiguous`, `sj1Mass`, and `sj2Mass` vectors.

Invalid masses are quiet NaNs. The status codes are:

| Code | Name |
|---:|---|
| 0 | `valid` |
| 1 | `no_selected_jets` |
| 2 | `no_selected_constituents` |
| 3 | `invalid_com` |
| 4 | `invalid_thrust` |
| 5 | `no_ca_jets` |
| 6 | `complexity_guard` |
| 7 | `no_valid_partition` |
| 8 | `numerical_failure` |

Gate results are not stored or allowed to suppress reconstruction. For a gate
requiring `n` jets above `T_gate`, count entries of `akJetPt` strictly greater
than `T_gate`. Combine that gate with only reconstructions satisfying
`T_keep <= T_gate`; `n=0` is the canonical ungated case. This factorization is
what permits unbiased gate, reconstruction-given-gate, and end-to-end
efficiencies without duplicating masses for every gate.

## Defaults and local use

The production grids are:

- AK radius: one of `0.4, 0.6, ..., 1.6` per job;
- `T_keep` and metadata `T_gate`: `100, 120, ..., 400` GeV;
- CA radius: `0.4, 0.6, ..., 1.6`, filtered by the legacy
  `CA >= max(0.4, AK-0.2)` constraint;
- hybrid cut: `0.0, 0.1, ..., 0.9, 0.95`;
- gate multiplicity metadata: `0, 1, ..., 6`;
- ambiguous-jet limit: 12;
- ROOT compression: ZSTD level 6;
- event limit: 1,000 by default, overridable to 10,000 or another positive
  count.

Example:

```bash
cmsRun SuuAnalysis/ExistingOptimization/test/runCompactOptimizationScan_cfg.py \
  inputRootFiles=SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt \
  outputRootFile=WbWb_4000_1000_akR0p8.root \
  akRadius=0.8 \
  maxEvents=1000

python3 SuuAnalysis/ExistingOptimization/test/validate_compact_scan.py \
  --strict-branches WbWb_4000_1000_akR0p8.root
```

CSV overrides are available for `collectionPtCuts`, `caRadii`,
`cosThrustCuts`, `defaultGateJetCounts`, and `defaultGatePtCuts`. Use
`enforceLegacyRadiusConstraint=False` for a radius ablation. The grouped
Condor workflow and its dry-run commands are documented in
`run_scripts/condor_scans/README_compact_grouped.md`.

## Validation performed

- isolated CMSSW 15.0.19 compilation;
- deterministic end-to-end cmsRun with PUPPI weights and all endpoint/status
  paths;
- explicit proof that 12 ambiguous jets are evaluated and values above the
  cap are guarded;
- a 1,000-event, 1,056-configuration stress run;
- strict ROOT schema/content validation and six validator regression cases;
- 114-sample by 7-AK-radius Condor manifest dry run (798 unique jobs), plus
  local JDL materialization without submission.

Use `evaluate_compact_scan.py` and `evaluate_compact_scan_diagnostics.py` for
this format; see the [evaluation guide](compact_scan_evaluation.md). The
per-point evaluators archived under `legacy/` read a different event format.
