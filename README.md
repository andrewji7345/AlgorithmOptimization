# ExistingOptimization

Optimize boosted Suu→ChiChi reconstruction with **signal sensitivity**, using
Optuna to propose configurations and HTCondor to produce calibrated signal and
background ntuples. The default campaign follows the **2017 cut-based selection
in AN-23-067** at 41.48 fb⁻¹.

The current search uses a fixed **15 × 17 mass grid** and an Asimov discovery
score with finite background-MC uncertainty. The horizontal Suu-mass bins cover
2500–10000 GeV in 500 GeV steps; the vertical average-superjet-mass bins cover
750–5000 GeV in 250 GeV steps. The new campaign covers **114 mass/decay hypotheses**
and 23 background samples. Optuna varies reconstruction and the additional
jet-count/pT gate while the AN baseline and region rules remain fixed.

Physicality is off in this workflow. The aggregate minimizes worst relative
Asimov regret plus one quarter of its mean, using references frozen before the
main search. Start with the [2D workflow](docs/sensitivity_2d_workflow.md); the
[score definition](docs/sensitivity_objective.md) explains the weights, plots,
normalization and handling of bins with insufficient background information.

## 1. Build and validate one sample

From `CMSSW_15_0_19/src`:

```bash
cmsenv
scram b -j 8
edmPluginRefresh "$CMSSW_BASE/lib/$SCRAM_ARCH"
cmsRun SuuAnalysis/ExistingOptimization/test/runSensitivityScan_cfg.py \
  inputRootFiles=SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt \
  sampleName=WbWb_4000_1000 sampleKind=signal \
  outputRootFile=WbWb_4000_1000_sensitivity.root maxEvents=1000 \
  akRadius=0.8 collectionPtCuts=100 caRadii=0.8 cosThrustCuts=0.5 \
  defaultGateJetCounts=0,3 defaultGatePtCuts=100 \
  enforceLegacyRadiusConstraint=False
python3 SuuAnalysis/ExistingOptimization/test/validate_compact_scan.py \
  --strict-branches WbWb_4000_1000_sensitivity.root
```

Use an initialized CMS proxy for remote MiniAOD input. The controller needs
Python packages in [requirements-sensitivity.txt](requirements-sensitivity.txt),
including pinned Optuna. Keep access to CMSSW's Python/ROOT runtime; a separate
Python interpreter with incompatible shared libraries cannot run PyROOT tests.

The ntuplizer writes every processed event and its generator weight, including
selection failures. Schema 3 adds nominal correction weights, baseline and
signal-region decisions, cutflow quantities and SJ rest-frame CA4 multiplicities.
The Suu mass remains `M(SJ1 + SJ2)` for each reconstruction configuration.

## 2. Review samples and prepare references

For the remaining commands, enter this repository:

```bash
cd "$CMSSW_BASE/src/SuuAnalysis/ExistingOptimization"
```

[The background catalog](config/backgrounds_2017.json) contains the 23 analysis
QCD, ttbar, single-top and W+jets datasets. [The full signal catalog](config/signal_grid_2017.json)
records all 114 available mass/decay samples and their analysis-source branching
conventions. The HtZt nominal 6/2 TeV alias contains an explicitly recorded
6.2/1.95 TeV substitute; its approved production-cross-section interpolation and
actual generated masses are preserved. Check the catalog offline with:

```bash
python3 build_signal_grid_catalog.py --check
```

Both new campaigns are **pilots** whose normalization uses all processed
preselection events. They need representative sampling; full production needs
complete coverage and audited signed generator totals. The fine grid may have
undefined cells even when an older coarse-bin score worked. Plots remain
available, but an unsupported hypothesis cannot supply a valid optimizer score.

Use [the calibration configuration](config/optuna_2d_calibration_2017.json) to
establish positive Asimov references for every hypothesis. Freeze the declared
calibration results with `freeze_sensitivity_references.py`, then start the
separate main study. Commands and operational prerequisites are in the
[2D workflow](docs/sensitivity_2d_workflow.md). The main reference file is
intentionally absent until actual calibration supports it.

## 3. Prepare and run Optuna

After calibration and reference freezing:

```bash
python3 optimize_sensitivity.py --config config/optuna_2d_2017.json
```

This prepares reviewable trial manifests and JDLs. Add `--execute` to submit
and manage the configured study once the compiled bundle, proxy and scheduler
are ready. Repeating the same command resumes it. Each trial runs through the
terminal without per-trial prompts. The controller validates completed shards,
records immutable inputs and code/calibration provenance, and refuses changed
reference content. See the [scheduler guide](docs/optuna_workflow.md) for details.

Optuna chooses AK4 or AK8 with the corresponding PUPPI jet corrections, collection
threshold, rest-frame CA radius, thrust cut and extra gate multiplicity/threshold.
Every evaluated 2D trial exports signal maps, each background dataset and class,
summed background, per-cell Asimov maps, and ROOT/JSON histograms with variances.
The reported best utility and retrospective worst-plus-mean regret rankings are
separate quantities with explicitly recorded references.

For standalone evaluation of an existing produced campaign:

```bash
python3 evaluate_sensitivity.py --campaign /path/to/trial/campaign.json \
  --trial-json /path/to/trial/trial.json --output-dir results/sensitivity/check
```

`evaluate_compact_scan.py` aliases this evaluator. The earlier
`config/sensitivity_2017.json` and `config/optuna_2017.json` retain the three-signal
1D pilot definition, including its explicit physicality gate, for reproducing
old work. Uncorrected legacy ntuples cannot supply the current calibrated score.

## 4. Validate the objective against a reference likelihood

The [reference validation workflow](docs/reference_workflow.md) produces a fixed
reconstruction grid on independent development and validation MC files. It adds
a separately identified source port of the original PAT-AK8 reconstruction,
SR/CR/AT0b/AT1b observables, and real jet and event-weight uncertainty variations.
`compare_reference_likelihoods.py` exports representative cards and runs blind
Combine expected limits, significance and diagnostic fits to test the fast score.

This separate likelihood-validation comparison enforces a 17.5% background MC
uncertainty target on its fit bins and retains its older physicality gate. The
current raw 2D search does not impose that superbin criterion on each raw cell. Its [likelihood model](docs/likelihood_reference.md) records
missing systematics, coarse binning and other approximations explicitly. It is
an intermediate validation tool; it does not claim the final analysis's sensitivity.

A separate coarse-bin comparison can add
`--binning-config config/reference_coarse_2017.json` in a new output directory.
Inference checks use distinct discovery and uncertainty-fit ranges, tighter
discovery minimization, and an explicit check for clipped MINOS upper intervals;
saved fits must satisfy the recorded current policy before reuse.
This declares `[0,3500,12000]` GeV bins, explicit QCD grouping and exploratory
reuse of the existing folds. It preserves the original four-bin results and
does not resolve missing-background coverage; see the workflow for the statistical
assumptions and confirmation requirements.

## Repository layout

| Path | Purpose |
| --- | --- |
| `plugins/`, `interface/`, `python/` | Reconstruction, AN selection and calibration integration. |
| `data/analysis_2017/` | Versioned correction payloads and provenance. |
| `config/` | Backgrounds, full signal catalog, calibration/main campaigns and Optuna definitions. |
| `test/runSensitivityScan_cfg.py` | Calibrated signal/background ntuplizing entry point. |
| `test/signalMCFiles/`, `test/backgroundMCFiles/` | Source MiniAOD file lists. |
| `sensitivity_metrics.py`, `evaluate_sensitivity.py`, `plot_sensitivity_2d.py` | Weighted yields, Asimov evaluation and 2D plot/export. |
| `sensitivity_objective.py`, `freeze_sensitivity_references.py` | Fixed-reference utility and retrospective regret ranking. |
| `optimize_sensitivity.py`, `sensitivity_jobs.py`, `run_scripts/sensitivity/` | Persistent optimizer, terminal scheduler and workers. |
| `produce_reference_ntuples.py`, `evaluate_reference.py` | Independent MC folds, reusable grids, reference reconstruction and systematic templates. |
| `likelihood_model.py`, `combine_runner.py`, `compare_reference_likelihoods.py` | Representative likelihood, blind inference and proxy comparison. |
| `evaluate_compact_scan_diagnostics.py` | Detailed selected-configuration mass diagnostics. |
| `test/` | ROOT/schema, objective and orchestration tests. |
| `docs/` | Analysis reference, schema, objective and operational details. |
| `legacy/` | Historical physicality/retention evaluators, old scripts and results. |

The grouped raw ntuplizer and its old Condor scripts remain available for
reconstruction studies with arbitrary AK radii. See the
[format and reconstruction guide](docs/compact_optimization_scan.md) and
[legacy index](legacy/README.md). Existing results remain historical artifacts.
Generated ntuples, plots, logs, study databases and bundles are ignored by Git;
calibration ROOT files under `data/analysis_2017/` are source inputs.

## Checks

The [2D validation record](docs/sensitivity_2d_validation.md) documents the real
ungated/four-jet checks, numerical export audits and remaining empty-background
cells. The maps work with existing ntuples; all 114-hypothesis optimization still
needs adequate MC coverage and positive frozen references.

```bash
python3 -m unittest discover -s test -p 'test_*.py' -v
```

Run in the configured CMSSW/Optuna environment to include ROOT and controller
integration tests. See [validation results](docs/sensitivity_validation.md) for
what was exercised and which production checks remain.

The real AK4/AK8 batch pilot completed 52 jobs and both Optuna evaluations.
It exposed and fixed LPC scheduler edge cases and the reference b-tag
event-weight fallback. See the [pilot results](docs/sensitivity_validation.md#submitted-batch-pilot)
for validation evidence and the physics constraints that rejected both trials.
