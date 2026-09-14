# Legacy optimization archive

Archived on 2026-09-10 when the compact grouped scan became the main repository
entry point. All moved files were preserved; no scan results or ROOT ntuples
were deleted. The complete old-to-new mapping is in
[archive_manifest.json](archive_manifest.json).

| Path | Contents |
| --- | --- |
| `evaluate_compact_scan_physicality.py` | Historical compact signal-only physicality/retention rankings; replaced by the active sensitivity evaluator. |
| `evaluate_ntuplizer.py` | Detailed single-ntuple old/new reconstruction diagnostics. |
| `evaluate_ntuplizer_pt_ak_ca.py` | Single-sample per-point grid evaluator. |
| `evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py` | Multi-sample single/two-threshold evaluator. |
| `resolution_metrics.py` | FWHM helper used only by the archived evaluators. |
| `run_scripts/interactive_scans/` | Historical pT, radius, and thrust scan/evaluation wrappers. |
| `run_scripts/condor_scans/` | Single/two-threshold submission scripts, workers, JDLs, manifests, logs, and saved tarball. |
| `results/` | Prior per-point evaluations, including original/top-configuration comparisons. |
| `debug/eventDebugReadout.txt` | Saved event debugging transcript. |
| `workflow.md` | Historical workflow guide with relocated command paths. |

Run evaluators from the package root, for example:

```bash
python3 legacy/evaluate_ntuplizer.py --input /path/to/point.root \
  --output legacy/results/single_point --both
python3 legacy/evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py --help
```

Run Condor scripts/JDLs from `legacy/run_scripts/condor_scans/`. Historical
campaign notes describe the old grids and account-specific EOS locations;
review those settings before a rerun. Interactive `run_*.sh` drivers still
expect the CMSSW release's `src` working directory; evaluation wrappers resolve
the archived evaluator and output directory relative to their own location.

The old CMSSW analyzer, cfi, and runnable cfg remain under `plugins/`, `python/`,
and `test/` at the package root. Both workflows share `test/signalMCFiles/`.
Raw ntuples outside this package, including EOS outputs, were not moved.

Archived results, Condor logs, and tarballs are local, Git-ignored artifacts.
Previously tracked legacy log files disappear from the tracked source tree;
their original bytes remain in this local archive and in Git history. A fresh
clone does not contain these generated artifacts. To restore a moved item,
move its `destination` back to its `source` using the manifest (first check that
the old path is free). Update script/document paths if restoring source files.
