# Evaluating the reference and reconstruction grid

`evaluate_reference.py` reads the additive analysis observables in the new
compact ntuples. It returns the source-ported AN reconstruction anchor plus
every stored **ungated** hybrid configuration. Optional explicit
`ConfigurationKey` objects can include trial gates. Equivalent zero-gate
threshold aliases are not enumerated.

```python
from evaluate_reference import evaluate_reference_campaign, validate_disjoint_folds

audit = validate_disjoint_folds(
    "production/ak8_nominal/development_campaign.json",
    "production/ak8_nominal/validation_campaign.json",
)
result = evaluate_reference_campaign(
    "production/ak8_nominal/development_campaign.json",
    variation_campaigns={
        "JECUp": "production/ak8_JECUp/development_campaign.json",
        "JECDown": "production/ak8_JECDown/development_campaign.json",
        "JERUp": "production/ak8_JERUp/development_campaign.json",
        "JERDown": "production/ak8_JERDown/development_campaign.json",
    },
    region="SR",
    bin_edges=[0, 3500, 5500, 7500, 12000],
    qcd_groups=[[0, 1], [2, 3]],
)
```

Every result is JSON serializable. `models[model_name].per_signal[signal_name]`
contains an `input` dictionary accepted directly by
`likelihood_model.export_model()`, the `fast_score`, separate `physicality`
diagnostics, and combined `feasible`/`failure_reasons` fields. The anchor name
is `reference_AN2017`; hybrid names are the existing canonical configuration
slugs. `physicality` uses the existing `physicality_pass` flag.

The result also preserves per-sample nominal histograms, all fourteen weight
variations, cumulative cutflows, counts for all four regions, normalization
factors, and generated denominators. Signal hypotheses are alternatives: a
builder input includes only its chosen signal, with QCD, ttbar, single top,
and W+jets aggregated according to campaign categories.

## Validation and normalization

The evaluator requires schema-v3 ntuples with `analysisObservableVersion=1`,
the specified source-port reconstruction identifier, and all fourteen named
weight variations. It rejects missing branches, inconsistent vector widths,
nonfinite/negative correction weights, inconsistent CA4 energy counts, invalid
mass/status pairs, duplicate sample event IDs, and mismatched metadata.

Regions are independently recomputed from the stored tagging observables and
compared with the ntuplizer's flags. The cuts retain the strict anti-tag mass
boundary below 150 GeV, zero CA4 jets above 50 GeV for the anti-tagged superjet,
and at least two CA4 jets above 300 GeV for each tagged superjet. Jet veto,
baseline, and reconstruction validity prerequisites are checked. This confirms
the stored region logic, not a full event-by-event comparison with the original
analysis executable. The original-selection provenance audit stays false
unless explicitly provided by the campaign after that independent validation.

Both the anchor and hybrid reconstruction use **`referenceWeight`**, which
excludes the original processor's nominally disabled top-pT factor. Generator
weights are multiplied exactly once. The fourteen stored variation weights are
absolute correction weights, not ratios; their metadata names determine their
columns, so reordering the columns cannot change their interpretation.
`topPtUp`/`topPtDown` affect only the ttbar process and preserve the original
one-sided variation.

Normalization uses all processed preselection events, including failed
reconstructions and events that enter no region. Event-tree generator sums and
squared sums must agree with each file's metadata. Sample shards are combined
before determining their single normalization factor. A subset campaign remains
explicitly a pilot; a selected-event denominator is never substituted for the
generated denominator.

## Independent samples and systematic pairs

`validate_disjoint_folds()` checks complete sample coverage, consistent physics
definitions, distinct source MiniAOD files, distinct output files, and no
overlapping `(run,lumi,event)` identities within each sample. It reads the IDs
directly from ROOT. It does not infer independence just because the output
filenames differ, and it does not compare IDs from unrelated MC datasets.

Kinematic variations must be supplied in Up/Down pairs. Their campaigns must
have identical sample physics definitions and source MiniAOD files. Sorted
event-ID and event-ID/generator-weight hashes must match nominal independently
for every sample. Reordering events is allowed; changing the event set or
per-event generator weight is rejected even if the total denominator happens
to remain the same. Selection, reconstruction, and correction weights are
allowed to change, as the variation is meant to measure those effects.

Only paired templates that are actually supplied become JEC/JER nuisances.
Absent variations are listed in `missing_kinematic_variations`. Total JEC is
named `CMS_jec_Total_2017` and remains distinct from the seven source-specific
JEC nuisances required for full AN fidelity.

## Statistical support and physicality

All models use the same fixed one-dimensional Suu bin edges and QCD groups.
The four-bin default uses `[0,3500,5500,7500,12000]` when those edges are in the
campaign, with groups `[[0,1],[2,3]]`. Histogram flow folds into boundary bins.
There is no automatic per-model bin merging, background flooring, or signed
yield repair.

The authoritative fast score requires at least `1 / 0.175**2` effective
background events in each signal-populated bin, or a stricter campaign
threshold. Physicality uses the baseline-plus-trial-gate population before
region selection, including failed reconstructions in its invalid fraction.
Physicality failure does not erase the fast score: it independently makes the
candidate ineligible. A signal hypothesis with zero selected yield is also
ineligible.

Each model also reports `background_coverage`, including each sample's generated
exposure, selected count, weighted yield, variance, empty bins and signed
cancellations. This report is copied into likelihood provenance. A sample with
no selected events contributes neither yield nor observed variance, so the
total-background effective-count gate cannot detect its missing contribution.
An empty component requires a justified coverage study; it does not establish
a zero physical rate. Passing the numerical gates remains conditional on the
observed templates and is not complete analysis validation. No invented
zero-count upper limit or background floor is applied.

If **only** the effective-background-MC threshold prevents the fast score,
`diagnostic_fast_score` evaluates the same formula with a negligible minimum
count threshold and marks it `diagnostic_only`. This allows a numerical
software comparison against a diagnostic Combine card at limited statistics.
It changes neither the authoritative score nor candidate eligibility. The
diagnostic field is `null` for empty backgrounds, invalid yields, or any case
where that specific relaxation is inappropriate.

Control and anti-tag regions are evaluated independently by changing `region`.
Their MC-only outputs diagnose coverage and signal contamination; they do not
establish data/MC closure or constrain the SR through a transfer factor.

The command-line entry point writes one result JSON without overwriting an
existing file:

```bash
python3 evaluate_reference.py \
  --campaign production/ak8_nominal/development_campaign.json \
  --variation JERUp=production/ak8_JERUp/development_campaign.json \
  --variation JERDown=production/ak8_JERDown/development_campaign.json \
  --region SR --output validation/reference_development_SR.json
```

All local ROOT reads use the project's memory-mapped reader to avoid the
environment's asynchronous local-file transport issue.
