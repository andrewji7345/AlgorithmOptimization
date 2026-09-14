# Fixed 2D Asimov sensitivity optimization

The current entry points are `config/sensitivity_2d_calibration_2017.json` and
`config/sensitivity_2d_2017.json`. They use the nominal AN reference weights,
114 signal mass/decay hypotheses and all 23 declared analysis backgrounds.
They are limited-MC pilots, not complete analysis measurements. The older
`sensitivity_2017.json` remains an explicitly physicality-gated, three-signal
1D pilot for reproducing earlier work.

## Event selection and mass grid

`evaluate_sensitivity.py` reads calibrated schema-v3 ntuples and validates the
additive AN observable contract. It applies the stored AN baseline, successful
reconstruction and signal-region flags. Optuna may additionally require at
least `n_gate_jets` calibrated reconstruction jets with `pT > gate_pt_cut`.
Zero required jets disables this extra gate. The gate uses the chosen AK radius;
its threshold is at least the collection threshold. It does not replace the
AN's baseline object, trigger, jet-multiplicity, HT, veto, b-tag or SR rules.
Changing reconstruction can change which events pass those fixed rules.

The fixed axes are:

| Axis | Reconstructed quantity | Range | Width | Bins |
| --- | --- | --- | --- | --- |
| Horizontal | `M(SJ1 + SJ2)` | 2500–10000 GeV | 500 GeV | 15 |
| Vertical | `(M(SJ1) + M(SJ2))/2` | 750–5000 GeV | 250 GeV | 17 |

All 255 rectangular cells are retained. Arrays use `[x bin][y bin]`; flattening
preserves this order and does not change the score. Exact upper endpoints enter
the last bin. With `histogram_flow: "exclude"`, masses outside either axis range
are excluded from these cells and counted separately as `flow_entries`,
`flow_sumw` and `flow_sumw2`. Their generated events still enter normalization.
Legacy campaigns may explicitly use `fold`, which adds these events to edge bins.

`physicality_mode: "off"` removes the old bias, width, tail and reconstruction
retention veto. A model can lose events or reconstruct masses poorly and still
be evaluated on the resulting signal sensitivity. Missing/corrupt input,
nonfinite selected masses and statistically undefined yields remain errors or
infeasible scores. Optional `diagnostic` and `gate` modes preserve the older
physicality implementation; when enabled, its definition is part of the fixed
score context. The new 2D campaign does not use it.

## From simulated events to expected yields

For each sample separately, every selected event has weight

```text
w = luminosity_pb * cross_section_pb * filter_efficiency * k_factor
    / sum_gen_weights
    * genWeight * referenceWeight
```

The luminosity is 41480 inverse picobarns (41.48 inverse femtobarns). Cross
sections in picobarns describe the sample's generated process, including the
specified decay/filter convention. Each background is normalized separately;
its binned event weights are then added to the other backgrounds. In cell i,
`B_i = sum_background_samples sum_selected_events_in_i w`. Each signal hypothesis
gets its own `S_i`. Different mass points are alternative production hypotheses.
At a fixed mass, the six decay modes all contribute to the source model with
its stated branching fractions. This optimization deliberately evaluates those
contributions separately for robustness across decay channels. It does not yet
sum their signal templates to quote an inclusive discovery significance for
that full production model.

`genWeight` is the signed generator contribution. At some perturbative orders,
positive and negative generated contributions must cancel to reproduce the
calculation. A negative weight is not a negative-probability physical event.
Using absolute weights, or counting all events as +1, would change that
calculation. The denominator sums the same signed generator weights over
**all processed events before selection**, including events failing the trigger
or jet gate. It does not sum the final corrected/selected event weights.

`referenceWeight` contains the stored nominal pileup, L1-prefiring and b-tag
corrections. It excludes `genWeight`, luminosity and cross section. These factors
correct simulation-to-data differences and can change shapes because their
values depend on event kinematics or pileup. They are applied once. The nominal
reference convention leaves the ttbar top-pT correction out of the central
weight, matching the existing AN reference implementation; it is available in
the separate uncertainty workflow. The older `weight_convention: "analysis"`
uses `analysisWeight` and retains its different ttbar convention. These two
conventions cannot be mixed in one comparison or frozen-reference study.

Alongside the expected yield, the evaluator accumulates
`V_i = sum_background_events_in_i w^2`, the statistical variance estimate of
that weighted MC prediction. Negative generator weights reduce the signed
yield but add positively to this variance. A large luminosity normalization
therefore does not create additional simulated information. Background
components with no selected MC are explicitly reported in `background_coverage`;
an empty component is not proof of a zero physical rate.

Full-dataset normalization requires numeric signed generator totals and event
counts, and complete matching ntuple coverage. The supplied pilots use
`normalization_scope: "representative_subset"`, with both totals taken from
unfiltered ntuple metadata. This assumes the limited files/events represent
the process. They remain labeled `production_ready: false`. Duplicate events,
missing shards, inconsistent generator sums and missing hypotheses are rejected.
All trials and calibration references must use the same physical MC population.

## Per-cell and per-hypothesis Asimov scores

For a known background, a cell contributes

```text
q_i = 2 * ((S_i + B_i) * log(1 + S_i/B_i) - S_i).
Z_h = sqrt(sum_255_cells q_i).
```

`Z_h` approximates the median discovery significance of that signal contribution
at its declared cross section, considered alone against the background. The displayed cell color is `sqrt(q_i)`. Cell colors
are neither added nor averaged to form the hypothesis score: their **squares**
add. For small signal relative to a known background, one cell approaches
`S/sqrt(B)`. See [Cowan et al.](https://arxiv.org/abs/1007.1727).

The implemented score also accounts for finite background MC using each
cell's variance `V` and an independent effective Poisson auxiliary constraint:

```text
q_i = 2 * [
  (S+B) * log((S+B)*(B+V)/(B*B+(S+B)*V))
  - B*B/V * log(1 + V*S/(B*(B+V)))
].
```

The auxiliary effective count is `B*B/V`; `V=0` uses the known-background limit.
This is equation 20 of [Cowan's background-uncertainty note](https://www.pp.rhul.ac.uk/~cowan/stat/medsig/medsigNote.pdf).
It is a fast planning score. It omits the full analysis's correlated systematic
uncertainties, signal MC uncertainties, data control-region constraints and
expected cross-section limit calculation.

The new campaigns set `minimum_background_effective_events: 0`: no extra hard
per-cell MC-count cut is imposed, while `V` remains in the score. Plots flag
`n_eff < 10` as a diagnostic only. Neither value is the AN's 17.5% requirement,
which concerned its **merged superbins**, not each of these 255 raw cells.

A cell with `S=B=V=0` contributes zero. A positive signal with `B<=0`, negative
expected yield, or `B=0` with nonzero variance does not support this approximation.
It is marked undefined, and the complete hypothesis/trial is infeasible. Other
valid cell colors are still exported. No epsilon background, negative-yield
clipping, trial-dependent bin omission or automatic merging is used. A genuine
zero-selected signal has score zero and remains in the aggregate; missing signal
input is an error. Limited MC can therefore produce useful maps without a
usable full-grid optimizer score.

## Robust ranking across mass points and decay channels

The old retention study minimized `worst_regret + 0.25 * mean_regret`, where
regret was the loss of retention relative to that hypothesis's best configuration.
Raw Asimov values span very different scales across masses. The new adaptation
normalizes each score to a **fixed calibration reference** `Z_ref,h`:

```text
ratio_h = Z_h / Z_ref,h
regret_h = 1 - ratio_h
regret_objective = max_h(regret_h) + 0.25 * mean_h(regret_h)
Optuna utility = min_h(ratio_h) + 0.25 * mean_h(ratio_h)
```

Optuna maximizes the utility. It equals `1.25 - regret_objective`, so it selects
the same optimum as minimizing that regret. It is a dimensionless ranking
utility, not a combined physical discovery significance. Scores above their
reference are allowed, giving negative deficits; improvements are not clipped.

References must be positive and cover all 114 hypotheses. First evaluate a
small declared calibration cohort using `objective.method: "mean_asimov"`, then
freeze the per-hypothesis maximum of its valid scores. Calibration maxima set
scales; they are not claimed to be global optima. The reference builder rejects
incomplete/infeasible calibration results and hypotheses with only zero scores.
It records input hashes, score definitions, normalization and sorted event-ID
and generator-weight digests. A changed reference, mass grid, weight convention
or physical MC population requires a new study and recalibration as appropriate.
Using the best result seen so far inside a running Optuna objective would
silently change the meaning of already stored trials; the frozen references
avoid that problem.

For reporting, `sensitivity_ranking.json` and `study_rankings.json` also calculate
relative regret against the best feasible score in the supplied finished cohort.
They sort by worst regret plus one quarter of mean regret. This retrospective
ranking is explicitly separate from the stationary Optuna utility; the two can
differ because their reference values differ. Stored Optuna values are preserved.

## Run and inspect

See the [2D workflow](sensitivity_2d_workflow.md) for calibration, freezing and
Optuna commands. Standalone evaluation accepts an actual produced campaign:

```bash
python3 evaluate_sensitivity.py --campaign /path/to/trial/campaign.json \
  --configuration '3:300:100:0.8:0.8:0.5' --output-dir results/sensitivity/2d
```

The configuration is `n:Tgate:Tkeep:RAK:RCA:c`. Pass repeated
`--compare-objective` paths to compare compatible completed evaluations.
Outputs include `objective.json`, `per_signal.csv`, a Suu projection for small
campaigns, PNG/PDF
maps for every signal and input background, background-class sums, total
background and every hypothesis's Asimov cells. `histograms_2d.json` and
`histograms_2d.root` retain numerical yields, actual sumw2 variances, entries,
validity/support masks and effective MC counts. `plots_2d_manifest.json` maps
sample identities to collision-resistant artifact names and hashes.

An evaluated but infeasible trial exits successfully with objective `-1` and
explicit failure reasons. Invalid/corrupt/missing input exits with code 2 and
is treated as failed work by the controller. Feasible utilities are nonnegative.
