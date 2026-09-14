# Run the fixed 2D search

This workflow scans reconstruction and the additional jet-count/pT gate while
retaining the existing nominal AN selection implementation. It does not run
Combine. Run its commands from `$CMSSW_BASE/src/SuuAnalysis/ExistingOptimization`
after initializing the CMSSW runtime. The objective, weights, statistical limits and plot formats are defined
in [sensitivity_objective.md](sensitivity_objective.md).

## Inputs and signal normalization

`config/signal_grid_2017.json` expands the 114 nonempty signal lists already in
`test/signalMCFiles`: six decay modes at 19 nominal mass pairs. It includes
WbWb, WbHt, WbZt, HtHt, HtZt and ZtZt. The grid uses Suu masses 4, 5, 6, 7 and
8 TeV, with available Chi masses between 1 and 3 TeV. All 23 original analysis
backgrounds remain in the new campaigns. Dataset names were checked against
the file lists; this inventory alone does not verify current DAS availability
or readability of every remote LFN.

The signal normalization is reproduced from the public analysis source pinned
in `data/analysis_2017/signal_normalization_provenance.json`. Cross sections
include the source's Suu-to-ChiChi branching convention and specified hadronic
decay factors; no extra branching or NLO factor should be applied. In particular,
the source's Higgs factor 0.58 is retained as its chosen decay convention, not
relabeled as an inclusive hadronic Higgs branching fraction. Reproduce/check
all values without network access:

```bash
python3 build_signal_grid_catalog.py --check
```

`HtZt_6000_2000.txt` is an explicit substitute: its generated masses are
**6200/1950 GeV**, while its alias retains the nominal grid point. With user
approval, its LO production cross section is interpolated logarithmically
between the source's 6 and 7 TeV values:

```text
sigma_prod(6.2 TeV) = exp(0.8 * log(137 fb) + 0.2 * log(23.1 fb))
                    = 95.96190254 fb.
```

The branching function is then evaluated at the actual 6.2/1.95 TeV masses,
followed by the HtZt decay factors and factor two for the mixed mode. The final
sample cross section is **0.0008488757072 pb**. This is a documented interpolation
assumption, not a newly calculated generator cross section. Actual and nominal
masses, interpolation inputs, file-list hashes and source provenance are retained.

## Prepare a calibrated pilot

Initialize the built CMSSW runtime and controller dependencies as in the README.
The current changes affect Python evaluation/configuration only; workers still
need the existing compiled AN/JEC/JER ntuplizer and payloads. Create a fresh
bundle at the filename specified in both new Optuna configurations:

```bash
python3 run_scripts/sensitivity/pack_cmssw.py \
  --cmssw-base "$CMSSW_BASE" \
  --output "$CMSSW_BASE/../sensitivity_bundles/CMSSW_15_0_19_sensitivity_2d_v1.tar.gz"
```

Review `config/optuna_2d_calibration_2017.json` before submitting. Its initial
budget is four trials, one active at a time, at most one file and 10000 events
per sample. Each trial includes **137 samples**, so even this limited calibration
is larger than the earlier three-signal pilot. These limits are conservative
operational defaults, not an assertion that every raw mass cell will have enough
MC support. The supplied campaigns explicitly remain representative-subset pilots.

Prepare manifests without submitting:

```bash
python3 optimize_sensitivity.py --config config/optuna_2d_calibration_2017.json
```

After reviewing the manifests and preparing a valid CMS proxy and scheduler,
add `--execute` to the same command to submit and manage the calibration.
The controller uses the explicit `mean_asimov` mode here to produce each mass/channel's
score. Calibration establishes reference scales; the main search uses the robust
aggregate below. More calibration configurations may improve those scales.
Existing ntuples can also supply calibration when they cover the exact declared
signals, backgrounds, physical events and required observables.

The main and calibration configurations must keep identical input files and
per-job event limits. Changing files, splitting, truncation or sample coverage
may change physical events and generator denominators, invalidating references.
A list of simulated inputs does not yet provide audited full-dataset signed
weight totals; those are a separate prerequisite for a production campaign.

## Freeze references once

Inspect calibration maps and `objective.json` first. Every required hypothesis
must have a positive reference in a complete, feasible calibration cohort. A
positive signal in a raw cell with no positive background MC is undefined under
this score. Its plot is still useful, but it cannot be used to manufacture a
reference value. If coverage fails, increase MC coverage, then recalibrate under
that new fixed exposure. The requested raw grid is not silently changed.

Use actual completed result paths; repeat `--calibration-objective` for each
chosen, distinct calibration configuration:

```bash
python3 freeze_sensitivity_references.py \
  --campaign config/sensitivity_2d_2017.json \
  --calibration-objective /path/to/calibration/trial_000000/evaluation/objective.json \
  --calibration-objective /path/to/calibration/trial_000001/evaluation/objective.json \
  --output config/sensitivity_2d_references_2017.json
```

The reference file is intentionally absent from the repository until this step
can be supported by actual calibration results. The main search refuses to
start without it. The builder never overwrites different existing references.
It records the calibration cohort and exact physics/MC definitions. References
must not change during the main study; start a new study after any such change.

## Main search and outputs

The main default budget is twelve trials, with the same fixed sample exposure
as calibration. Prepare and inspect its manifests:

```bash
python3 optimize_sensitivity.py --config config/optuna_2d_2017.json
```

Add `--execute` to submit. Repeat the same command to resume. Optuna maximizes
`min(Z/Z_ref) + 0.25*mean(Z/Z_ref)` across all 114 hypotheses. It varies AK4/AK8
radius, collection threshold, rest-frame CA radius, thrust cut, extra gate count
0–6 and its pT threshold. The original AN baseline and region cuts remain fixed.
An extra gate can only remove baseline-passing events.

Each completed trial produces per-signal and per-background 2D PNG/PDF plots,
background-class and total sums, per-cell Asimov plots, full ROOT/JSON numerical
histograms and a manifest. Undefined cells and low-MC diagnostics are visible.
A stored objective of -1 with `feasible: false` is a correctly evaluated but
unusable statistical configuration. Missing/invalid input is a failed task,
not an observed score. `study_summary.json` identifies the best stationary
utility; `study_rankings.json` separately reports completed-cohort regrets.

This workflow retains nominal AN weights and the existing checked selection
port. It does not establish full historical-analysis closure or a publication
significance/limit. In particular, all 114 hypotheses still need processed,
compatible MC; the available expanded validation ntuples cover only three WbWb
signals. Independent validation after optimization remains necessary to check
that a preferred configuration was not selected for an MC fluctuation.
