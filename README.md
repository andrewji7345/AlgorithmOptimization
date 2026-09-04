# ExistingOptimization

This CMSSW package searches the configuration space of the manual boosted-event reconstruction algorithm used by SuuAnalysis. It is independent of the ParticleTransformer repository: this package produces reconstruction ntuples for many jet thresholds, AK/CA radii, and cosine-thrust matching cuts, then compares the configurations across signal decays and mass points.

Two algorithm families are supported:

- **Single threshold (legacy):** `jetPtCut` selects every AK jet passed to reconstruction.
- **Two threshold:** an event must first contain `minEventJets` AK jets above `eventJetPtCut`; accepted events use every AK jet above the lower, scanned `jetPtCut`.

Generated evaluation plots and tables belong under `results/` and are ignored by Git.

## Setup

The package targets `CMSSW_15_0_19`. Place it at `CMSSW_15_0_19/src/SuuAnalysis/ExistingOptimization`, then build from the release's `src` directory:

```bash
cd ~/nobackup/research/CMSSW_15_0_19/src
cmsenv
scram b -j 8
```

The evaluators require Python 3 with `awkward`, `matplotlib`, `numpy`, and `uproot`. Condor submission also requires the EOS/XRootD and HTCondor clients and a CMS proxy:

```bash
voms-proxy-init --valid 192:00 -voms cms
```

## Run one reconstruction point

Run from `CMSSW_15_0_19/src`. The input text file contains one ROOT URL per line; blank and `#` comment lines are ignored.

```bash
cmsRun SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py \
  inputRootFiles=SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt \
  outputRootFile=rootfiles_existingOptimization/WbWb_4000_1000_pt300_ak8_ca8_th85.root \
  jetPtCut=300 akRadius=0.8 caRadius=0.8 cosThrust=0.85
```

The configuration currently processes 1,000 events. Set `process.maxEvents.input` in `test/runExistingOptimizationNtuplizer_cfg.py` to `-1` to process all events.

For a two-threshold point, enable the event gate:

```bash
cmsRun SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py \
  inputRootFiles=SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt \
  outputRootFile=rootfiles_existingOptimization/WbWb_4000_1000_pt100_ak8_ca8_th85_two_threshold_300_100.root \
  jetPtCut=100 eventJetPtCut=300 minEventJets=4 \
  akRadius=0.8 caRadius=0.8 cosThrust=0.85
```

The output tree contains every analyzed event and records the event-gate result
in `passesEventSelection`. Evaluation efficiency is therefore the fraction of
all analyzed events that pass the gate. Finite, valid two-chi reconstruction is
tracked separately for the mass-quality metrics. The evaluator reports this as
`selection_efficiency`; `reconstruction_efficiency` is retained as a deprecated
CSV alias. Two-threshold ntuples produced before this branch was added must be
paired with their corresponding all-event single-threshold ntuples. The
evaluator reconstructs their event gate offline from the single-threshold
`ak_pt` branch. New two-threshold ntuples are evaluated directly using their
stored `passesEventSelection` branch.

## Run scans

Scripts in `run_scripts/interactive_scans/` run small grids directly with `cmsRun`. Invoke them from `CMSSW_15_0_19/src`, and edit their sample, loop ranges, and output paths as needed.

The Condor submitters generate a TSV manifest, package and upload CMSSW, and submit one job per point. Start with one test job:

```bash
cd SuuAnalysis/ExistingOptimization/run_scripts/condor_scans
./submit_ntuplizer_scan_pt_ak_ca_th.sh --test
# or
./submit_ntuplizer_scan_two_threshold.sh --test
```

Check `condor_q`, `logs*/job_*.out`, and `/store/user/aji/rootfiles_existingOptimization` before replacing `--test` with `--full`. The scripts contain user-specific CMSLPC/EOS paths for `aji`; update `cmssw_src`, `eos_tarball_dir`, and `eos_output_dir` for another account. The two READMEs in `run_scripts/condor_scans/` document job counts, naming, and operational details.

## Evaluate results

Compare old slimmed-AK8 and directly reclustered reconstruction for one ntuple:

```bash
python3 evaluate_ntuplizer.py --input /path/to/point.root \
  --output results/single_point --both
```

Rank one sample's pT/AK/CA grid:

```bash
python3 evaluate_ntuplizer_pt_ak_ca.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization \
  --sample WbWb_4000_1000 --output-dir results/WbWb_4000_1000
```

Evaluate all discoverable decays and mass points, including both algorithm families:

```bash
python3 evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py \
  --input-dir /eos/uscms/store/user/aji/rootfiles_existingOptimization \
  --output-dir results/evaluate_ntuplizer_dc_mp_pt_ak_ca_th --th-code 85
```

Use `--help` for event limits, mass windows, nominal-point, strictness, and global-ranking controls.

Both scan evaluators define relative mass resolution as
`FWHM / histogram peak`. The FWHM is measured around the principal mass peak
using linearly interpolated half-maximum crossings and Freedman--Diaconis
histogram binning. The multi-sample evaluator applies this definition to
`m_reco / M_chi` and includes it directly in the balanced score.

## File and directory roles

| Path | Role |
| --- | --- |
| `plugins/ExistingOptimizationNtuplizer.cc` | EDAnalyzer implementing truth association, old and reclustered manual reconstruction, event selection, and the output tree. |
| `plugins/BuildFile.xml`, `BuildFile.xml` | CMSSW build metadata and dependencies on PAT, FastJet, and ROOT. |
| `python/ExistingOptimizationNtuplizer_cfi.py` | Default analyzer instance, input collections, and reconstruction parameters. |
| `test/runExistingOptimizationNtuplizer_cfg.py` | Runnable `cmsRun` configuration and command-line options. |
| `test/signalMCFiles/*.txt` | MiniAOD inputs grouped by decay channel and `(M_Suu, M_chi)`; `WbWb_all.txt` is a combined WbWb list. |
| `test/eventDebugReadout.txt` | Saved event-level debugging output. |
| `evaluate_ntuplizer.py` | Detailed single-ntuple old/new reconstruction diagnostics. |
| `evaluate_ntuplizer_pt_ak_ca.py` | Single-sample grid metrics, ranking CSV, heatmaps, and selected mass histograms. |
| `evaluate_ntuplizer_dc_mp_pt_ak_ca_th.py` | Main multi-decay/multi-mass evaluator with family comparisons, global ranking, regret, and coverage summaries. |
| `run_scripts/interactive_scans/run_ntuplizer_scan_*.sh` | Direct pT, radius, and thrust scan drivers. |
| `run_scripts/interactive_scans/evaluate_ntuplizer_scan_*.sh` | Wrappers for evaluating interactive scan outputs. |
| `run_scripts/condor_scans/submit_*.sh` | Validate, generate manifests, package/upload CMSSW, and submit scans. |
| `run_scripts/condor_scans/run_*.sh` | Worker scripts that run one point and copy its ROOT file to EOS. |
| `run_scripts/condor_scans/*.jdl` | HTCondor arguments, resources, logs, manifests, and rescue submissions. |
| `run_scripts/condor_scans/scan_parameters*.tsv` | Generated or saved job manifests. |
| `run_scripts/condor_scans/README*.md` | Legacy and two-threshold campaign notes. |
| `results/` | Generated CSV files and plots; ignored by Git. |

## Output naming

The multi-sample evaluator recognizes:

```text
<decay>_<M_Suu>_<M_chi>_pt<pt>_ak<10R_AK>_ca<10R_CA>_th<th>.root
<decay>_<M_Suu>_<M_chi>_pt<low>_ak<10R_AK>_ca<10R_CA>_th<th>_two_threshold_<high>_<low>.root
```

Thus `ak8`, `ca12`, and `th85` mean radii 0.8 and 1.2 and a cosine-thrust cut of 0.85.
