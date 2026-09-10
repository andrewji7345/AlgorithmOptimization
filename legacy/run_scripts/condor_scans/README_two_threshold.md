# Two-threshold ExistingOptimization Condor scan

This is a sideways scan alongside the legacy `pt/AK/CA/thrust` submission.
It uses separate submit, worker, JDL, manifest, log, and CMSSW tarball names;
the existing scan files are not changed.
Its EOS archive is
`CMSSW_15_0_19_existingOptimization_two_threshold.tgz`; it is built in `/tmp`,
outside the CMSSW tree, and is independent of the ParticleTransformer
training-target archive.
The packager excludes the entire ParticleTransformer package, saved analysis
results, and submission artifacts. The ExistingOptimization ntuplizer has no
build or runtime dependency on ParticleTransformer; its source, configuration,
input lists, and compiled CMSSW products remain in the archive.

Each event must have at least four reclustered AK jets with `pT >= 300 GeV`.
For accepted events, all AK jets above the scanned lower threshold enter the
remaining reconstruction. The scan covers:

- decay channels: `WbWb`, `WbZt`, `WbHt`, `ZtZt`, `HtHt`, `HtZt`;
- mass points: `4000_1000`, `6000_2000`, `8000_3000`;
- lower thresholds: `100, 120, ..., 300 GeV`;
- AK and CA radii: `0.4, 0.6, ..., 1.6`;
- radius constraint: `CA >= AK - 0.2`;
- cosine thrust: `0.85`.

There are 34 allowed AK/CA pairs and 6,732 full-scan jobs
(`18 samples x 11 lower thresholds x 34 radius pairs`).

## Build and initialize the proxy

```bash
cd ~/nobackup/research/CMSSW_15_0_19/src
cmsenv
scram b -j 8
voms-proxy-init --valid 192:00 -voms cms
```

## Submit one test point

From this directory, run:

```bash
./submit_ntuplizer_scan_two_threshold.sh --test
```

This submits exactly one point:
`WbWb_4000_1000`, lower threshold `100 GeV`, AK `0.4`, CA `0.4`, and
cosine thrust `0.85`. Check it before the full scan:

```bash
condor_q
tail -f logs_two_threshold/job_*.out
xrdfs root://cmseos.fnal.gov ls /store/user/aji/rootfiles_existingOptimization
```

## Submit the full scan

```bash
./submit_ntuplizer_scan_two_threshold.sh --full
```

The generated manifest is `scan_parameters_two_threshold.tsv`. The JDL limits
materialization and idle jobs so the 6,732 points enter the scheduler gradually.

## Output naming

Outputs share the legacy EOS directory but cannot collide with legacy files:

```text
<sample>_pt<lower>_ak<10R_AK>_ca<10R_CA>_th85_two_threshold_300_<lower>.root
```

For example:

```text
WbWb_4000_1000_pt100_ak4_ca4_th85_two_threshold_300_100.root
```

The legacy stem is preserved exactly. The suffix records both the fixed event
threshold and the scanned lower reconstruction threshold.

## Compare with the legacy scan

From `SuuAnalysis/ExistingOptimization`, run:

```bash
python3 legacy/evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py \
    --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization \
    --output-dir legacy/results/evaluate_ntuplizer_dc_mp_pt_ak_ca_th \
    --th-code 85
```

The evaluator discovers both canonical filename forms automatically. Existing
legacy heatmaps remain legacy-only. The global ranking, regret, and coverage
outputs compare both families, while the two-threshold 3-by-6 score/component
and winning-hyperparameter summaries are written under
`across_regimes/two_threshold_300/`.
