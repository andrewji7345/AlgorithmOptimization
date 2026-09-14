# Fixed 2D evaluation validation — 2026-09-14

The fixed 15×17 Asimov workflow was tested in the CMSSW Python/ROOT environment.
The full suite exercised 245 tests. It exposed one data-inventory mismatch when
the new normalization assets were added; those assets were added to the hashed
provenance inventory and all four asset tests then passed. The other 244 tests
passed in the full run. The validation directory retains the original full-suite
log and the focused correction check; post-integration checks are recorded there
as well.

Coverage includes signed-weight normalization, analytic likelihood limits, an
independent Poisson-control likelihood calculation, bin additivity, exact axis
orientation and boundaries, excluded-flow conservation, physicality-off behavior,
AN reference-weight and region checks, missing/zero signal distinctions, frozen
reference creation followed by evaluation of a changed gate, reference/MC drift
rejection, all 114 signal normalizations, real ROOT export and Optuna scheduler
protocol/resume tests. Scheduler integration tests use local fixtures and do
not submit batch jobs.

## Existing ntuples

Two real evaluations used the same 105 nominal AK4 development ntuples:
**5,646,629 processed events**, three WbWb signals and all 23 backgrounds. They
used collection pT 120 GeV, rest-frame CA radius 0.6 and thrust cut 0.85. Both used
the requested mass rectangle, nominal reference weights, physicality off and
finite background-MC uncertainty with no hard effective-count cut.

| Extra jet gate | Expected background in rectangle | WbWb 4/1 TeV Z | WbWb 6/2 TeV Z | WbWb 8/3 TeV Z |
| --- | ---: | ---: | ---: | --- |
| None | 5341.899 | 13.44794 | 6.26741 | Undefined |
| At least 4 jets with pT>400 GeV | 619.03 | 27.81187 | 9.37345 | Undefined |

These are finite-MC planning scores for the separate WbWb contribution under the
source normalization convention; they are not inclusive model significances or
claims about data. The two gates are validation examples, not an optimization
result. Large changes after a gate motivate independent validation of its MC
support rather than a discovery claim.

For 8/3 TeV, eight cells contain signal but zero background MC. The evaluator
keeps the other 247 cell values, marks those eight as undefined, and records
both full three-benchmark trials as infeasible with objective -1. No full-grid
score or reference is inferred from the partial valid map. The fine grid therefore
still needs improved background coverage before an all-hypothesis main scan.
The available ntuples also cover only three of the 114 signal samples.

The ungated background has 14144 in-range MC entries and 316 excluded flow
entries. Its in-range weighted variance is 39150.0565; the excluded yield is
736.1040 with variance 7350.2794. The four-jet gate retains 3639 in-range entries
and 7 flow entries. Each pass produced 34 PNG/PDF maps: the three signals, 23
background datasets, four background classes, total background and three
Asimov maps, plus full numerical ROOT/JSON export.

## Independent checks and artifacts

- 182 comparisons to the frozen previous AN-reference evaluator passed: selected
counts, normalization, generator totals, yields and variances match for every
sample after adding the newly excluded flows back to the rectangle.
- 156 exact comparisons between the two gates passed for physical event and
generator identities, processed counts, denominators, normalization and input
files. The frozen-reference context is identical; selected bin populations and
sumw2 never increase under the additional gate.
- Each set of exports passed 142 ROOT-array checks and 70 artifact checksum checks.
Actual background and Asimov maps were visually inspected, including undefined
cells and low-statistics markings.

The machine-local validation directory is:

```text
/uscms/homes/a/aji/optimization_work/asimov_2d_20260914/validation/
```

It contains `real_ak4_ng0/`, `real_ak4_ng4_pt400/`,
`real_gate_comparison_audit.json`, test logs and `smoke_audit_tools/` with
reproducible scripts. Each real run has `objective.json`,
`plots_2d_manifest.json`, `histograms_2d.root`, `histograms_2d.json`,
`smoke_summary.json` and its export audit. The ungated normalization audit is
`real_ak4_ng0/nominal_conservation_audit.json`. Renderer hashes identify the
final display code; original ungated render artifacts were preserved separately.

No new batch jobs were submitted, and no C++ reconstruction/calibration code
changed for this step. Full 114-signal references and an actual all-hypothesis
Optuna scan have not been produced. Follow the [2D workflow](sensitivity_2d_workflow.md)
after establishing adequate MC coverage.
