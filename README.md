# ExistingOptimization

Optimize the manual boosted-event reconstruction algorithm in CMSSW 15.0.19.
The current workflow is the **compact grouped scan**: one job per signal sample
and AK radius, with reconstruction thresholds, CA radii, and thrust cuts stored
together and event gates evaluated offline.

## Setup and one-point validation

From `CMSSW_15_0_19/src`:

```bash
cmsenv
scram b -j 8
cmsRun SuuAnalysis/ExistingOptimization/test/runCompactOptimizationScan_cfg.py \
  inputRootFiles=SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt \
  outputRootFile=WbWb_4000_1000_akR0p8.root akRadius=0.8 maxEvents=1000
python3 SuuAnalysis/ExistingOptimization/test/validate_compact_scan.py \
  --strict-branches WbWb_4000_1000_akR0p8.root
```

Evaluation needs Python 3, NumPy, Matplotlib, Awkward, and uproot. ROOT-backed
validation tests run in a CMSSW environment with PyROOT.

## Scan and evaluate

From the package directory, validate the full manifest without submitting:

```bash
cd run_scripts/condor_scans
bash submit_ntuplizer_scan_compact_grouped.sh \
  --full --max-events 10000 --tag new-campaign --dry-run
```

This writes the active manifest. For production, initialize a CMS proxy and
repeat with a unique campaign tag and without `--dry-run`. See the
[Condor guide](run_scripts/condor_scans/README_compact_grouped.md) for submission,
rescue jobs, EOS paths, and account-specific settings.

From the package directory, evaluate a single campaign:

```bash
python3 evaluate_compact_scan.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization_compact \
  --input-glob 'compactOptimization_global-v1-10k_*.root' \
  --output-dir results/compact_global_v1_10k
python3 evaluate_compact_scan_diagnostics.py --help
```

Use a tag-specific glob to avoid mixing campaigns. See the
[scan design and schema](docs/compact_optimization_scan.md) and
[evaluation guide](docs/compact_scan_evaluation.md) for configuration identity,
physicality, retention, rankings, and diagnostic examples.

## Repository layout

| Path | Purpose |
| --- | --- |
| `plugins/`, `python/`, `BuildFile.xml` | CMSSW analyzers, defaults, and build metadata. |
| `test/runCompactOptimizationScan_cfg.py` | Current production configuration. |
| `test/signalMCFiles/` | Shared input lists for 114 signal regimes. |
| `test/validate_compact_scan.py`, `test/test_*.py` | Schema validator and regression tests. |
| `compact_scan_metrics.py` | Shared compact-format metrics. |
| `evaluate_compact_scan.py` | Global scan evaluation and Pareto summaries. |
| `evaluate_compact_scan_diagnostics.py` | Selected-configuration diagnostics. |
| `run_scripts/condor_scans/` | Compact submission/worker scripts, JDLs, and manifests. |
| `docs/` | Current workflow documentation. |
| `results/` | Current generated results; ignored by Git. |
| `legacy/` | Archived per-point evaluators, scans, campaign artifacts, and results. |

The old `ExistingOptimizationNtuplizer.cc`, its cfi, and
`test/runExistingOptimizationNtuplizer_cfg.py` remain in the standard CMSSW
locations for historical reruns. Use the [legacy index](legacy/README.md) to
find the corresponding tools and results.

## Checks

From the package directory in the configured Python/CMSSW environment:

```bash
python3 -m unittest discover -s test -p 'test_*.py'
```

Generated plots, ROOT files, logs, Python caches, and tarballs should stay out
of Git. Saved campaign manifests and rescue JDLs are retained for provenance.
