# Sensitivity workflow validation

Validated on 2026-09-10 with CMSSW 15.0.19 (`el9_amd64_gcc12`), Optuna 4.5.0
and the installed LPC HTCondor 25.0.12 commands. The initial local checks
below preceded batch submission; the submitted pilot results are recorded
at the end of this document.

## Build and automated tests

The package compiled in an isolated CMSSW release. The final Python suite
passed **84 tests**, with no skips, using CMSSW Python/ROOT and the Optuna
packages on its Python path:

```bash
python3 -m unittest discover -s test -p 'test_*.py' -v
```

Coverage includes schema 1/2 compatibility, schema 3 cutflow and SR invariants,
Suu masses, CA4 rest-energy bounds, negative generator weights, signed-yield
normalization, missing/duplicate input rejection, fixed 1D/2D significance,
finite-MC penalties, sparse-background rejection, physicality failure exclusion,
sensitivity regret, DAS failures and dataset membership, and calibration hashes.

Controller tests use actual Optuna/SQLite and terminal scheduler fixtures.
They cover ask/tell completion, a real ROOT-to-evaluator subprocess integration,
resume, lost submission responses, held and failed jobs, timeouts, polling
outages, file locks, immutable source/archive checks, completion receipts and
stale-objective rejection. A code edit during one earlier test run was correctly
detected by the provenance guard; the final suite ran against stable sources.

The correction inputs also passed numerical spot checks: 60 flavor/eta/pT
b-tag evaluations, five pileup evaluations, and 27 JEC evaluations. The asset
suite verifies the recorded SHA256 and available immutable upstream blob hashes.

## Real MiniAOD processing

All following runs exited successfully and passed the strict ROOT validator:

| Input | Events | Reconstruction configurations | Result |
| --- | ---: | ---: | --- |
| `WbWb_4000_1000` | 100 | 12 | 53 baseline events; 10–26 SR events depending on reconstruction; 1,135 valid and 65 guarded event/configuration pairs. |
| `TTJetsMCHT800to1200` | 100 | 1 | 100 valid reconstructions; top-pT and ttbar b-tag weights exercised. |
| `QCDMC_Pt_1800to2400` | 20 | 1 | Exact packaged worker completed; 20 valid reconstructions. |
| `WJetsMC_QQ_HT800toInf` | 20 | 1 | 20 valid reconstructions; W+jets correction inputs exercised. |
| `ST_tW_top_inclMC` | 20 | 1 | All events retained despite no selected reconstruction jets; nonunit generator-weight sum about 648.944 preserved. |

These tests read actual FNAL XRootD files and exercised trigger/filter inputs,
lepton IDs, nominal JEC/JER, PU, prefiring, sample-specific b-tag maps, veto maps,
and reconstruction. They caught and resolved the CMSSW 15 prefiring product's
change from `double` to `float`. The local plugin cache was refreshed after
adding the new producer; the README includes that command.

A small signal/QCD runtime campaign also passed the real evaluator and wrote
its normalized Suu plot and objective diagnostics. It returned an explicit
infeasible pilot result for insufficient background and failed physicality,
rather than manufacturing significance from an empty background template.
These tiny inputs are processing checks, not a sensitivity measurement.

## Dataset and scheduler checks

Every one of the 23 background dataset queries succeeded against DAS,
identifying **8,510 available files**. The repository stores one deterministic
pilot file per background and a query/provenance record for each dataset.
The shipped campaign planned two Optuna trials with 26 tasks each (23
backgrounds and three signal benchmarks), capped at one file and 10,000 events
per sample/task.

The installed native `original_condor_submit -dry-run` accepted a generated
four-task JDL, including distinct working directories, transfers, X.509 proxy,
resources and trial/task ClassAds. Read-only `condor_status` also verified LPC
scheduler discovery. Site wrappers lacking executable shebangs and explicit
scheduler routing were addressed in the controller.

The **exact worker.sh** completed a separate end-to-end local run: build a
1.33 MB archive, verify SHA256, extract into a fresh scratch directory, run
`scram b ProjectRename`, initialize the relocated runtime, process 20 remote
QCD events, run strict schema validation, and write a successful matching
completion receipt. This exercises the code Condor runs without submitting a
job to the scheduler.

## Production scope

The default is a pilot with representative-subset normalization. Increasing
the MC coverage is necessary when SR bins lack ten effective background
entries. Production requires audited full signed generator denominators and
complete processing, plus an independent validation sample for the chosen
configuration. Nominal sensitivity here is a search-planning proxy; it does not
replace the full analysis likelihood with correlated systematics and control
regions. Only the 2017 cut-based SR is implemented; the calibrated reconstruction now supports AK4 and AK8 PUPPI.

The submitted pilot below subsequently exercised Condor matchmaking, worker
transfer, both trial evaluations, and completed-study resume. Long production
campaigns and broad parameter exploration remain outside this small pilot.


## AK4 reconstruction extension

The added radius search uses `ak_radii: [0.4, 0.8]`, dedicated AK4 PUPPI
Summer19UL17 V5 JEC and JRV3 resolution/SF payloads, and generator matching to
`slimmedGenJets` within DeltaR<0.2 for AK4. The fixed PAT baseline and nominal
event weights remain common to both choices. AK4 reconstruction uses pT>50
GeV and abs(eta)<2.5, followed by the scanned additional pT threshold; AK8
retains its previous ET/eta requirements.

The 100-event signal smoke test passed at both radii, with 12 reconstruction
configurations per event. Every AK8 event branch matched the preceding
implementation exactly, including invalid masses, guards, weights, and SR
bits. AK4 and AK8 had identical baseline decisions and event weights. The
AK4 run and the exact packaged worker's separate 100-event high-pT QCD run
both passed strict ROOT validation. The worker verified archive hashes,
relocated CMSSW, selected radius 0.4 from its task parameters, and emitted a
successful completion receipt. No scheduler jobs were submitted.

New regression cases cover categorical radius choice and distinct trial
identities, invalid radius lists, radius changes on resume, metadata/profile
compatibility, and a real AK4 ROOT -> Optuna/controller -> evaluator subprocess
integration. New production jobs require the v2 correction profile; archived
v1 AK8 files remain readable for standalone evaluation. The expanded default
uses a fresh study directory and bundle name.

## Submitted batch pilot

The corrected pilot on 2026-09-10 completed **52/52 jobs with exit code 0**:
clusters **30378816** (AK4) and **30378817** (AK8), procs 0–25 on
`lpcschedd5.fnal.gov`. Each trial used 26 samples capped at 1,000 events,
for 52,000 event processings. Both Optuna trials are COMPLETE; none remain
running. All 89 final automated tests passed and the working CMSSW release
was rebuilt.

The initial attempt exposed a saturated ttbar b-tag efficiency bin, plus LPC
cancellation basename, empty-query, and already-finished-job behavior. The
corrected event-weight helper reproduces the reference's whole-event b-tag
fallback, records its use in two new branches, and has a distinct v3 correction
profile. Scheduler regression tests cover the discovered command edge cases.
The initial failed study and corrected study remain separate for provenance.

Every worker passed strict ROOT validation. An independent audit verified
sample/profile/receipt identities, event counts, finite weights, signed
normalization, selected counts, and every normalized Suu sumw/sumw2 bin.
Baseline decisions and weights matched exactly across all 26 radius pairs.
The b-tag fallback affected 14 TTJets HT>2500 events per radius, one of which
passed the baseline. Peak recorded memory was 1,102 MB; worker wall times
were 63–173 seconds. Restarting the completed controller preserved job IDs,
objective hashes, and evaluation attempt counts without duplicate work.

Both objectives are -1 with feasible=false, and the sensitivity/regret ranking
is empty. Both configurations lack sufficient effective background statistics;
AK4 fails physicality for the 4/1-TeV benchmark, and AK8 fails for all three.
This validates handling of infeasible models, not a sensitivity improvement.

The complete report, plots, logs, task outputs, source manifest, audit JSON,
and integrity-checked SQLite backup are under
`/uscms/home/aji/sensitivity_runs/pilot_20260910_ak4_ak8_v3/`.
See `validation_report.md` there for parameters, counts, provenance and resume
instructions. The original failed attempt is in the adjacent directory
`pilot_20260910_ak4_ak8`.
