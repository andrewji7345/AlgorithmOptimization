# Compact reconstruction and ROOT format

This ntuplizer implements the expanded reconstruction family while keeping
event gates factorized from reconstruction. The active [sensitivity evaluator](sensitivity_objective.md) requires calibrated schema 3. Historical raw scans use schema 1/2 and remain readable for mass diagnostics.

## Reconstruction and reuse

One job covers one signal or background sample and one anti-kT radius. For every event it:

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
the spherical `ee_genkt` alternative remains outside this scan. In raw mode no JEC is applied. Sensitivity mode supports laboratory AK radii 0.4 and 0.8. It applies radius-specific PUPPI JEC/JER to each reclustered jet and scales its constituents consistently. AK4 reconstruction jets require pT>50 GeV and abs(eta)<2.5; AK8 retains ET>300 GeV and the first-two/subsequent centrality rule. Both require tight ID and the trial's additional pT threshold. Standard PAT AK4/AK8 objects independently define the fixed AN baseline; the retained reconstruction threshold is the optimization parameter. See [the reference prescription](analysis_reference.md).

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

Sensitivity outputs use **schema version 3**. Raw grouped scans still write version 2, which added `suuMass`; the reader and validator also support version 1. Version 3 requires `useJEC=true`, `analysisSelection="AN-23-067-UL2017-cutbased-v1"`, `correctionPrescription="UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3"`, and `sampleKind` equal to `signal` or `background`.

With the cfg module label `compactScan`, each file has:

- `compactScan/Metadata`: one entry containing the schema version, sample,
  algorithm choices, grids, base/config index mapping, status dictionary,
  event count, and generator-weight sums;
- `compactScan/Events`: one entry per input event with `run`, `lumi`, `event`,
  `genWeight`, sorted `akJetPt`, base-level `nCAJets`, and flat per-config
  `recoStatus`, `nAmbiguous`, `sj1Mass`, `sj2Mass`, and `suuMass` vectors.

`suuMass` is a `vector<float>` in GeV aligned with `configId`, just like the
SJ mass vectors. It stores the invariant mass `M(p4_SJ1 + p4_SJ2)`, evaluated
before the lab-frame boost; it is not the scalar sum of the two SJ masses.
It is filled only for valid reconstructions, independently of the offline
event gate. Invalid masses are quiet NaNs. The status codes are:

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

Schema 3 adds these event branches:

| Branches | Type / meaning |
| --- | --- |
| `analysisWeight` | Float; nominal PU, b-tag, prefiring and applicable top-pT factors, excluding `genWeight`. |
| `passesTrigger`, `passesFilters`, `passesLeptonVeto`, `passesJetVeto`, `passesBaseline` | Boolean cutflow decisions. |
| `analysisHT` | Float; nominal corrected AK4 HT in GeV. |
| `analysisNAK4`, `analysisNAK8`, `analysisNHeavyAK8`, `analysisNBTags` | Unsigned-short object counts. |
| `sj1NCA4E300`, `sj2NCA4E300` | Per-configuration unsigned-short counts of R=0.4 CA jets with energy >300 GeV in each SJ rest frame. |
| `passesRecoJetVeto` | Per-configuration unsigned-byte boolean; every retained reconstruction jet passes the analysis veto map. |
| `passesSignalRegion` | Per-configuration unsigned-byte boolean: baseline, reconstruction-jet veto, ≥1 medium b tag, valid reconstruction, and ≥2 such CA4 jets per SJ. |

Physicality is evaluated on baseline plus the trial gate, before the per-configuration SR cut. Generator-weight metadata cover **every processed input event**. Nothing is filtered out of the output before those denominators are recorded. The strict validator checks flag consistency, vector widths, finite weights and valid/invalid mass conventions.

## Historical raw grouped defaults

The raw grouped grids are:

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

## Historical raw-mode validation

- isolated CMSSW 15.0.19 compilation;
- deterministic end-to-end cmsRun with PUPPI weights and all endpoint/status
  paths;
- explicit proof that 12 ambiguous jets are evaluated and values above the
  cap are guarded;
- a 1,000-event, 1,056-configuration stress run;
- strict ROOT schema/content validation and six validator regression cases;
- 114-sample by 7-AK-radius Condor manifest dry run (798 unique jobs), plus
  local JDL materialization without submission.

Use `evaluate_sensitivity.py` for calibrated outputs and `evaluate_compact_scan_diagnostics.py` for mass diagnostics. The former physicality/retention evaluator is preserved as `legacy/evaluate_compact_scan_physicality.py`; see the [historical evaluation guide](compact_scan_evaluation.md). The
per-point evaluators archived under `legacy/` read a different event format.

The v3 correction profile adds scalar `analysisBTagWeight` (float) and
`analysisBTagWeightFallback` (bool). They record the reference analysis's
whole-event b-tag fallback; a set fallback flag requires weight exactly 1.
The strict validator checks the weight range [0,100] and that invariant.
Older v1/v2 correction profiles remain readable with their matching campaigns.
