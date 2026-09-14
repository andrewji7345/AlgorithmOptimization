# Reference likelihood validation

This workflow produces reusable MC ntuples, exports a representative likelihood,
and tests the existing fast sensitivity score against blind Higgs Combine
expected limits and discovery significance. It is a bounded validation study,
not the full optimization or a reproduction of the final CMS result.

The [reference reconstruction](reference_observables.md) and
[likelihood audit](likelihood_reference.md) describe the original source choices,
the implemented uncertainties, and the remaining differences. In particular,
the source-ported PAT-AK8 reconstruction is a separate anchor: a configurable
AK8 scan point is not automatically the original analysis. Both use the same
new `referenceWeight` convention in this comparison. The existing Optuna
objective remains available; its previous nominal `analysisWeight` convention
has not silently changed.

## Produce fixed grids on independent files

Use `produce_reference_ntuples.py --config CONFIG.json` to prepare a campaign.
The configuration specifies `campaign`, a DAS `input_dir`, `run_dir`, a built
`backend.cmssw_bundle`, radii `[0.4,0.8]`, and the five `systematics`:
`nominal,JECUp,JECDown,JERUp,JERDown`. It also requires positive integer
`files_per_fold`, `max_parallel_groups`, `poll_seconds`, and
`group_timeout_seconds`. The required `max_events_per_job` is a positive integer
or `-1` to process all events in each chosen source file. The campaign must explicitly declare pilot subset
normalization. The ordinary Condor backend resource options still apply.

The complete grid is declared as, for example:

```json
{
  "collection_pt_cuts": [120, 320],
  "ca_radii": [0.6, 1.2],
  "cos_thrust_cuts": [0.5, 0.85],
  "gate_counts": [0, 3],
  "gate_pt_cuts": [320, 500]
}
```

Set it under `scan_grid`. Each job processes one source file and stores eight
reconstruction configurations plus the reference. The default comparison uses
the ungated configurations, giving sixteen hybrid choices across both radii.
The stored gate observables can support a separately specified later study.

The producer chooses stable file-disjoint development and validation folds.
Optional `development_anchors` and `validation_anchors` map sample names to
ordered lists of source URLs already assigned to each fold. Every anchor is
retained in its named fold. To expand an existing study, copy **all** of its
selected development and validation input URLs into these respective mappings;
copying only development anchors would allow an old validation file to enter
the enlarged development fold. Both mappings must contain known sample names,
unique exact catalog URLs, and no shared files. A sample's requested
`files_per_fold` must accommodate every anchor in either fold. The producer
rejects insufficient catalog capacity instead of silently reducing the request.
CMS source-file aliases through different redirectors are also rejected, as is
reuse of a source across sample catalogs.

Use optional `sample_overrides` to allocate additional MC to specific processes.
Each entry accepts only `files_per_fold` and `max_events_per_job`; omitted fields
inherit the global values. For example:

```json
{
  "files_per_fold": 1,
  "max_events_per_job": 20000,
  "sample_overrides": {
    "QCDMC_Pt_470to600": {"files_per_fold": 26, "max_events_per_job": -1},
    "WJetsMC_QQ_HT800toInf": {"files_per_fold": 11, "max_events_per_job": 150000}
  },
  "development_anchors": {"QCDMC_Pt_470to600": ["root://HOST//store/PREVIOUS_DEV.root"]},
  "validation_anchors": {"QCDMC_Pt_470to600": ["root://HOST//store/PREVIOUS_VALIDATION.root"]}
}
```

Replace the illustrative URLs with actual catalog inputs. These example counts
are allocations, not guarantees of sufficient selected-event statistics. The
same per-sample allocation and source files apply to both radii and every
kinematic state. After removing both sets of anchors, the seeded hash ordering
fills development first and validation second; the selection does not inspect
event acceptance. Unknown configuration keys, sample names, override keys,
noninteger counts, and invalid event limits are rejected before jobs are prepared.
The frozen definition records the overrides, both anchor mappings, resolved
`sample_settings`, and selected input files. Each output campaign records the
production fingerprint, fold, anchors, and resolved settings under
`reference_production`; worker tasks record their actual event limit.

All processed generator weights, including selection failures, enter each
sample's normalization. This remains a representative subset estimate, not an
independently audited full-dataset denominator. Expanding either fold requires
a new run directory and definition. The producer reruns its selected files in
that directory; it does not merge old and new output files automatically. When
an event cap increases on an anchored file, its new ntuple contains the earlier
events too, so adding both old and new copies would double count them.

After packaging the compiled release, submit nominal jobs first:

```bash
python3 produce_reference_ntuples.py --config CONFIG.json --execute \
  --only-systematic nominal
```

Repeating the command resumes the persisted scheduler state. `--once` performs
one controller iteration. Once nominal outputs and MC support are checked, omit
`--only-systematic` to produce the remaining shifts. Input selection, grid,
event limits, executable source and archive hashes are frozen. A lost submission
response is reconciled through queue/history; the producer never blindly submits
the same group again. Failed groups remain failed for inspection. Use a new run
directory and explicit configuration to change the production definition.

## Compare the proxy with actual inference

Install Combine in an isolated supported CMSSW release as described in
[the runtime guide](combine_runtime.md). Keep the ntuple Python/ROOT runtime
separate from Combine's runtime. The wrapper receives literal command arguments.

```bash
python3 compare_reference_likelihoods.py \
  --production-dir /path/to/production \
  --output-dir /path/to/comparison \
  --runtime-wrapper /path/to/combine-env
```

The comparison first verifies development/validation event disjointness, then
checks identical source events and generator weights between nominal and shifted
samples. It uses the same four fixed Suu bins and QCD groups for every model and
fold. It compares each signal hypothesis separately, fitting the SR by default;
`--region CR`, `AT1b`, or `AT0b` selects another individual region. It does not
invent an ABCD constraint or a simultaneous-region likelihood absent from the
reference analysis.

Each configuration, benchmark and fold receives two cards with the same grouped
finite-MC nuisance directions: `statistics_only` omits analysis priors;
`reference` broadens each QCD group coefficient by the 15% blanket term in
quadrature and includes the implemented normalization and shape nuisances.
Bins in a QCD group share one fluctuation in both stages; this is an AN-inspired
correlation approximation. Stage contract version 2 records this controlled
comparison. Earlier version-1 diagnostics changed the QCD covariance between
stages and cannot isolate the effect of analysis uncertainties. Every card has
background-only Asimov observations. Combine evaluates
blind median expected 95% CLs limits, expected discovery significance at the
catalog signal cross section, and a separate signal-plus-background Asimov
diagnostic fit. Successful fit status and covariance quality are required.
Observed data are never fitted. Expected cross-section limits refer to the
catalog's fully hadronic WbWb final state, including its branching fractions.

The discovery and diagnostic fits use separate numerical settings. Discovery
uses minimizer strategy 2 and tolerance `1e-5`, with
`rMax = max(20, 5*rInjected)`. Controlled weak-signal tests found that default
precision could report zero significance for an Asimov signal with Z about 0.01;
the tighter setting agreed with tolerance `1e-7` in those tests.
FitDiagnostics instead uses
`rMax = max(20, 5*rInjected, 2*expectedLimitQuantile97p5)`. Its actual ROOT MINOS
upper endpoint must remain below 99% of that bound. A positive symmetric error
and covariance quality 3 alone can coexist with an interval clipped at the
parameter boundary. The physical lower bound remains zero, and covariance
quality 3, successful fit status and injection recovery remain required.
These ranges follow the need to resolve parameter uncertainties described in
the [Combine fitting documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/v11.0.0/part3/nonstandard/).
Both policies are recorded in receipts and verified against saved command
arguments and ROOT results on resume. Older inference receipts must be audited
with their frozen source; a new comparison reruns inference with the current
policies instead of relabeling old results. Histogram evaluation caches may be
copied unchanged when their evaluator, inputs and binning are identical.

`--nominal-only` explicitly omits JEC/JER templates for an intermediate check.
`--allow-unsupported-mc` permits diagnostic fits below the AN's 17.5% background
MC uncertainty target; these points remain ineligible for optimization. Neither
option fills zero backgrounds, repairs negative templates, or makes an incomplete
model production-ready. Nominal-zero bins with nonzero systematic migrations
are rejected: additional MC or a common, justified rebinning is needed.

`rows.json` records every successful, unsupported or failed fit. `summary.json`
compares fast sensitivity with Combine significance and inverse limit ordering,
including Spearman correlations, rank reversals and pairwise agreement. It shows
diagnostic and eligible populations separately. Development selects a candidate;
validation measures its performance without selecting it again. Aggregate
rankings require all three signal benchmarks to pass physicality and MC support.
Sensitivity regret still uses sensitivity and globally eligible candidates.
These finite-set correlations test the proxy in the chosen space; they cannot
certify it for arbitrary new selections or omitted nuisance models.

The fixed source-ported AN anchor may serve as a diagnostic denominator for
expected-limit ratios even if it fails the physicality gate. Its fits must be
complete and its observed MC support sufficient for every benchmark used in
the ratio. The report records its physicality and selection eligibility
separately. This does not admit a failing anchor into candidate selection or
sensitivity regret. Development choices use development reference limits;
held-out ratios use the corresponding validation reference limits.

Evaluation caches, cards, templates, workspaces, logs and inference receipts
remain in the comparison directory. Input ROOT files, campaign definitions,
likelihood configuration and executable code are hashed. Resume verifies those
inputs and completed artifacts. An interrupted or failed inference requires
inspection and a new comparison directory rather than overwriting its evidence.

## What is still required for an analysis result

The implementation provides total JEC and JER shifts, pileup, prefiring, medium
b-tagging variations, the source's one-sided top-pT convention, normalization
priors and a documented approximation to QCD modeling/MC statistics. It does not
provide the seven JEC source correlations, eta-split JER, PDF/scale treatment for
the QCD pT samples, the final two-dimensional superbin map/private statistical
implementation, full signal decay mixtures, a Run-2 combination, or data closure.
Those omissions are recorded in every model manifest. Expected performance from
this workflow must be described as a representative MC study.

The [normalization audit](reference_normalization_audit.md) also identifies
differences between the updated AN cross-section table and historical source
scaling constants. For example, the WJets LNu HT2500+ constants differ by a
factor of 3.307. Numerical agreement with an older cutflow table is therefore
not itself a normalization validation. This workflow retains the updated
cross sections and signed all-processed-event normalization for both models.

## Resumed comparison safeguards

Comparison rows record the authoritative campaign fast-score feasibility and
failure reasons separately from any relaxed diagnostic score. Selection,
sensitivity regret and filled plot markers require authoritative feasibility,
physicality, likelihood MC support and a successful fit. A fixed diagnostic
reference denominator needs authoritative and likelihood MC support and a
successful fit; its physicality failure remains recorded but does not prevent
that diagnostic normalization. Legacy rows without authority evidence remain
diagnostic; they must be regenerated from the preserved evaluation inputs.

Summaries receive the full campaign signal list and total planned fit count.
They retain missing benchmarks and mark intermediate snapshots provisional.
Development selections are withheld until every task has reported. A completed
report can still be inconclusive when fits or eligibility gates fail.

## A separately declared coarse-bin study

Pass `--binning-config config/reference_coarse_2017.json` to the comparison in
a new output directory. The configuration declares common edges, a partition
of the new bins into QCD groups, the choice rationale and validation scope.
Only removing existing mass boundaries while retaining both endpoints is
allowed. Every model, benchmark, fold and supplied systematic uses that same
map. The configuration is frozen in the comparison definition and recorded
in evaluation, likelihood and report provenance. Cache reuse with different
edges, groups or study metadata is rejected.

The supplied exploratory example uses `[0,3500,12000]` GeV and singleton QCD
groups `[[0],[1]]`. Yields, sumw2 and each named Up/Down template are recomputed
from the original events. The two stages share the resulting coarse-bin MC
basis. Rebuilding these groups changes the covariance relative to the original
four-bin model, so the difference is not solely a loss of mass resolution.
Independent 15% blanket terms remain a modeling assumption, not a measured
closure covariance. No old result or physicality gate is replaced.

The supplied configuration explicitly declares `exploratory_reuse`: both
existing folds have already been inspected. Completed choices and reports
therefore describe an exploratory stability test, even if numerical gates
pass. A fresh independent confirmation and missing-background coverage remain
necessary before using a choice for physics. A configuration marked
`independent_validation` is a caller's study-design declaration, not automatic
certification by the software that the sample has never been inspected.
