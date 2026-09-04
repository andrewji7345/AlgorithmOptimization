# Evaluating the compact expanded scan

The compact-format evaluators are an additive workflow. They do not replace
or modify `evaluate_ntuplizer.py`, `evaluate_ntuplizer_pt_ak_ca.py`, or
`evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py`, which continue to read the legacy
event format.

Two new entry points consume `compactScan/Metadata` and
`compactScan/Events`:

- `evaluate_compact_scan.py` compares complete configurations across signal
  mass points and decay channels and constructs physicality/retention Pareto
  summaries;
- `evaluate_compact_scan_diagnostics.py` drills into one sample and one or
  more selected configurations.

Both use `compact_scan_metrics.py` for ROOT discovery, configuration identity,
gate construction, cut-flow metrics, and the physicality definition.

## Configuration identity and event populations

A complete configuration is

```text
(n_gate, T_gate, T_keep, R_AK, R_CA, c)
```

where `c` is the hybrid cosine-thrust cut. `configId` is local to one ROOT
file and is never treated as a global identifier. For `n_gate=0`, there is one
canonical ungated configuration and `T_gate` is absent. For a nonzero gate,
only configurations with `T_keep <= T_gate` are formed.

The gate is reconstructed exactly from the stored, sorted AK-jet pT vector:
an event passes when at least `n_gate` jets have **strictly greater** pT than
`T_gate`. The evaluator reports three separate unweighted efficiencies:

```text
gate efficiency            = N(gate) / N(all)
reco given gate efficiency = N(gate and valid reco) / N(gate)
signal retention           = N(gate and valid reco) / N(all)
```

Thus signal retention is exactly the product of the first two efficiencies.
It is a signal-only retention proxy, not a discovery sensitivity: background
rates and a final analysis selection are intentionally deferred to the later
signal-sensitivity study.

## Physicality constraint

Mass responses use both lab-pT-ordered reconstructed SJs from every gated,
valid event. The physicality score is the maximum of independently normalized
terms:

```text
max(
  absolute response bias / bias limit,
  relative FWHM resolution / resolution limit,
  invalid fraction given gate / invalid-fraction limit,
  response-tail fraction / tail-fraction limit,
  minimum required valid events / observed valid events
)
```

A score no greater than one passes the working constraint. Empty gates or no
valid reconstruction produce an infinite score and fail. Component values and
limits are written to the output so changing a limit cannot silently change
the interpretation of a previous result.

The initial defaults are bias `0.20`, relative resolution `0.50`, invalid
fraction `0.20`, tail fraction `0.50`, and at least 50 gated valid events. The
response-tail window is `[0.70, 1.30]`; the common estimator uses 150 bins over
response `[0, 3]`. These are configurable working limits, not a claim that the
first chosen boundary is the final physics optimum.

The scan-wide mass estimator uses one fixed response histogram for all
configurations. This makes the exhaustive gate-expanded scan tractable and
ensures every regime uses identical binning. The diagnostic evaluator reports
the same primary score and may additionally show exact unbinned summaries for
the selected configurations.

## Nominal and generated masses

The sample-list filename defines the nominal regime shown on grid plots. The
generated masses are inferred from the MiniAOD dataset names inside the sample
list and are used for response normalization. This distinction matters for
`HtZt_6000_2000`: the available dataset is generated at
`MSuu=6200, MChi=1950`, so its reconstructed masses are divided by 1950, not
2000.

## Global outputs

The scan-wide evaluator writes a compressed aggregate table rather than the
roughly 35 million sample/configuration rows in the production grid. It also
writes:

- an input/run manifest and the resolved nominal/generated regime table;
- unconstrained and physicality-qualified per-regime optima;
- a physicality-qualified global regret/rank table;
- the seven-dimensional Pareto front;
- per-regime details for the leading global candidates; and
- compact Pareto, regret, coverage, and regime-summary plots.

Regret is the additive loss in end-to-end signal retention relative to the
best configuration in that regime. The global retention objective follows the
legacy robust convention:

```text
worst-regime regret + mean-weight * mean regret
```

Every configuration remains in the aggregate table even when it fails the
constraint. The fraction of regimes that fail physicality, and the fraction
that are missing, are explicit columns. This preserves the locally
hyper-tuned configurations needed to compare an absolute per-regime optimum
with a globally robust choice.

Pareto dominance does not use the maximum-combined physicality score or the
weighted global regret objective. It minimizes seven quantities independently:
the selected cross-regime statistic (worst by default, optionally q90) of
relative mass bias, relative FWHM, invalid fraction, response-tail fraction,
and `minimum valid events / observed valid events`, together with worst
retention regret and mean retention regret. A configuration is on the front
when no eligible configuration is at least as good in all seven quantities
and strictly better in at least one. The scalar physicality score remains only
for the separate all-regime-qualified ranking and per-regime pass/fail reports.

For one tagged production campaign, run from the package directory:

```bash
python3 evaluate_compact_scan.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization_compact \
  --input-glob 'compactOptimization_global-v1-10k_*.root' \
  --output-dir results/compact_global_v1
```

Quoting the glob is important: discovery reads each file's metadata, but a
tag-specific pattern prevents files from two campaigns from being mixed. A
duplicate `(sample, R_AK)` is rejected rather than silently selected. The
default coverage denominator is all 114 packaged signal regimes. Use
`--discovered-only` for an intentionally incomplete pilot.

To inspect rank 1 from the global physicality-qualified ranking:

```bash
python3 evaluate_compact_scan_diagnostics.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization_compact \
  --pattern 'compactOptimization_global-v1-10k_*.root' \
  --sample WbWb_4000_1000 \
  --ranking-csv results/compact_global_v1/physicality_qualified_ranking.csv \
  --rank 1 \
  --output-dir results/compact_diagnostics
```

Or select a configuration explicitly as
`n:Tgate:Tkeep:RAK:RCA:c`:

```bash
python3 evaluate_compact_scan_diagnostics.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization_compact \
  --pattern 'compactOptimization_global-v1-10k_*.root' \
  --sample HtZt_6000_2000 \
  --configuration 4:300:100:0.8:0.8:0.5
```

For the canonical ungated configuration, use `0:none:Tkeep:RAK:RCA:c`.

## Storage behavior

The evaluator reads one sample/AK file at a time. Large cross-regime arrays
are temporary float32 files under a temporary directory and are removed on
normal exit or failure. Persistent exhaustive output is gzip-compressed CSV;
the diagnostic per-event table is disabled by default and, when explicitly
requested, is also compressed.
