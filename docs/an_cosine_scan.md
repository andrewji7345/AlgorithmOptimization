# AN reference cosine threshold scan

This study varies the strict positive/negative cosine boundary in the fixed
`AN23-067-PATAK8-CA8-Thrust-source-port-v1` reconstruction. It starts from corrected
selected PAT AK8 daughters, reclusters CA R=0.8 in the event COM frame, and measures
CA R=0.4 substructure in each superjet rest frame. Only the reference partition
boundary changes. The optional custom reconstruction grid is independent.

The default [configuration](../config/an_cosine_scan_2017.json) covers WbWb 4/1,
6/2 and 8/3 TeV with 10,000 processed events each and 21 cutoffs, 0 through 1 in
0.05 steps. At 0.85, the scan reuses the existing fixed reference reconstruction;
the validator compares its statuses, regions, masses and subjet counts exactly.
The source's strict inequalities and guards of 14 ambiguous CA8 jets and 300
thrust particles apply at every cutoff, including endpoints.

## Efficiency and mass distributions

The primary efficiency is `N(baseline and reference SR at c) / N(baseline)`.
The denominator includes reconstruction failures and is fixed across cutoffs.
SR requires valid reference reconstruction, at least one medium b-tagged AK4,
and at least two CA4 subjets with rest-frame energy above 300 GeV in each SJ.
The AN baseline supplies the fixed trigger, filters, jet/lepton vetoes, HT and
multiplicity requirements. See [reference observables](reference_observables.md).

Primary curves use unweighted counts and 68.27% Wilson intervals. Tables also
contain nominal `genWeight * referenceWeight` ratios and correlated ratio errors.
These conditional efficiencies describe representative signal subsets; they
are not an absolute acceptance or a signal-versus-background optimization.

Mass plots use SR-selected reconstructed SJ and Suu masses, unweighted and
normalized to unit area within the displayed range. Both SJs enter the SJ
histogram. Dashed lines mark nominal Chi and Suu masses. Overlays show cutoffs
0, 0.5, 0.85 and 1; each sample's multipage PDF contains both observables at every
cutoff. No particle-ancestry distribution is used.

## Produce a new scan

Use the project's configured CMSSW Python environment, with PyROOT, NumPy and
Matplotlib available, and an initialized CMS proxy. Build this revision before
creating the CMSSW bundle:

```bash
cd "$CMSSW_BASE/src"
scram b -j 8
edmPluginRefresh "$CMSSW_BASE/lib/$SCRAM_ARCH"
cd SuuAnalysis/ExistingOptimization
python3 run_scripts/sensitivity/pack_cmssw.py \
  --output /path/outside/CMSSW/CMSSW_15_0_19_an_cosine.tar.gz

python3 run_an_cosine_scan.py prepare \
  --config config/an_cosine_scan_2017.json \
  --run-dir /path/to/cosine_scan_run \
  --bundle /path/outside/CMSSW/CMSSW_15_0_19_an_cosine.tar.gz \
  --schedd lpcschedd4.fnal.gov
python3 run_an_cosine_scan.py submit --run-dir /path/to/cosine_scan_run
python3 run_an_cosine_scan.py watch --run-dir /path/to/cosine_scan_run
```

Preparation writes three task manifests, input lists, the Condor JDL, a frozen
source snapshot, and bundle/source checksums. It does not submit. The bundled
CMSSW release must include the scan branches and validator. You may omit
`--schedd` to select and persist an available scheduler at submission.

Run `watch` in a persistent shell or tmux session initialized with the same
CMSSW/Python environment; it polls every minute and renders plots automatically.
To resume with the frozen driver, use
`python3 /path/to/cosine_scan_run/source/run_an_cosine_scan.py watch --run-dir /path/to/cosine_scan_run`.
Use `status` for saved status or `watch --once` for one scheduler poll. A site
override directory at `RUN/condor_config` is passed through
`FERMIHTC_LOCALCONF_OVERRIDE` when present.

Submission intent is saved before contacting Condor. Repeating `submit` cannot
create another cluster after a lost response; `watch` reconciles the token
against queue and history. Scheduler errors preserve jobs and are recorded for
retry. Failed jobs or plotting exceptions persist a `failed` status and their
reason. Correct a failed run in a new directory to preserve its frozen inputs.

## Input coverage and validation

The config records selected CMS logical filenames and their DAS event counts.
The 4/1 TeV file has 18,000 available events, of which the first 10,000 are read.
The 6/2 TeV inputs contain 1,000 + 9,000 events; the 8/3 inputs contain
1,000 + 8,000 + 1,000. Preparation rejects duplicate files and insufficient
declared coverage. Recheck DAS when changing the source catalog.

Workers run both strict schema and reference scan validators and require the
event-tree count and metadata to match the requested event limit before writing
a success receipt. Plotting repeats the coverage check. For other sensitivity
tasks, this worker check can be enabled with `require_full_event_coverage=true`;
the reference cosine scan enables it by default. An all-events limit of -1
checks internal counts without enforcing a positive target.

Each file carries optional metadata vector `referenceCosThrustCuts` (double) and
the matching event vectors:

| Branch | Element type |
| --- | --- |
| `referenceScanRecoStatus`, `referenceScanRegion` | unsigned byte |
| `referenceScanNAmbiguous`, `referenceScanSJ1NCA4E300`, `referenceScanSJ2NCA4E300` | unsigned short |
| `referenceScanSJ1Mass`, `referenceScanSJ2Mass`, `referenceScanSuuMass` | float, GeV |

These extend schema 3 when `referenceCosThrustCuts` is nonempty and require
analysis mode. The fixed reference scalar branches retain their meaning.

```bash
python3 test/validate_compact_scan.py --strict-branches /path/to/output.root
python3 test/validate_reference_cosine_scan.py /path/to/output.root
python3 -m unittest test.test_an_cosine_scan test.test_validate_compact_scan -v
```

## Completed study

The [2026-11-18 report](../reports/an_cosine_scan_2017_20261118/README.md) contains
all three 10,000-event samples, 63 efficiency points, mass distributions,
reconstruction-loss curves, numerical tables and portable input provenance.
It finished at 2026-11-19 07:08:49 UTC (2026-11-18 23:08:49 Pacific).
The original higher-mass single-file jobs reached only 1,000 events; the final
report uses the complete replacement outputs and the validated 4/1 TeV output.

Local ROOT outputs, compiled bundles, scheduler logs and frozen runtime
directories stay outside Git. The dated report is a deliberate small snapshot
of the completed analysis; ordinary generated results remain ignored.
