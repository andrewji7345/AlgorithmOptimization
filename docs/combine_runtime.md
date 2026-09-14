# Expected inference with CMS Combine

`combine_runner.py` converts one datacard into a workspace, runs a blind median
expected 95% CLs upper limit with its 68%/95% bands, calculates expected discovery
significance for an injected signal strength, and runs an independent Asimov
signal-plus-background fit diagnostic. It does not read an observed result into
the objective. The standard single signal-strength parameter `r` scales the
signal template; `r95 * benchmark_cross_section_pb` is an upper limit in the same
cross-section and branching-fraction convention as that template.

The statistical model remains the responsibility of `likelihood_model.py` and
its input manifest. Successful fits do not establish background closure, validate
missing systematic templates, or justify asymptotics in sparse bins.

## Runtime and invocation

The tested isolated installation is CMS Combine **v11.0.0**, source commit
`66f59f9a899b143dc95a1d6d866fab5df8850189`, built in `CMSSW_16_0_0` with
`el9_amd64_gcc13` using `scram b -j 4`. This follows the
[official installation recipe](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/v11.0.0/).
It is independent of the ntuplizer's CMSSW15 area.

On this account, the durable environment wrapper is:

```text
/uscms/home/aji/nobackup/research/analysis_tools/combine_v11/combine-env
```

`runtime_manifest.json` alongside it records the source revision and binary/library
SHA256 hashes. The wrapper clears inherited Python/ROOT library overrides before
activating CMSSW16, restores the caller's working directory, and `exec`s its argv.
It does not submit jobs. Keep this installation in `nobackup`: a full copy exceeds
the remaining HOME quota. The incomplete HOME copy from setup was removed and
HOME writes were verified afterward.

Run the Python controller from an initialized CMSSW environment with NumPy/uproot
available:

```bash
python3 combine_runner.py \
  --card /absolute/path/to/model/datacard.txt \
  --output /absolute/path/to/new/fit_directory \
  --runtime-wrapper /uscms/home/aji/nobackup/research/analysis_tools/combine_v11/combine-env
```

The default mass label is 125. It is a Combine bookkeeping/model argument, not
an assertion that the Suu signal has that mass: the supplied signal template
defines the mass hypothesis. Use `--mass` consistently if a datacard depends on
that argument. `--signal-strength` defaults to 1. The result is a limit on `r`,
not automatically a cross section; the calling evaluator must supply the
benchmark normalization.

The Python API is:

```python
from combine_runner import run_expected, CombineRunError

result = run_expected(card, output_dir,
    command_prefix=["/path/to/combine-env"], timeout_seconds=300,
    mass=125, signal_strength=1.0, r_max=1000.0)
print(result["expected_limit"]["median"], result["expected_significance"])
```

Each output directory must be new or empty. The caller chooses a fresh attempt
directory on retry. Subprocesses receive argument lists without shell expansion.
Every command has a timeout; a timed-out or interrupted command's process group
is killed. Logs, command argv, working directories, runtimes and failure receipts
remain available. A failed command or failed validation raises `CombineRunError`,
whose `.result` holds the same information written to `combine_result.json`.

## Blinding and fit validation

The limit command always uses `-M AsymptoticLimits --run blind --cl 0.95`.
Using only an expected quantile from an ordinary observed-data calculation would
not guarantee blinding: its nuisance parameters can depend on an observed fit.
`--run blind` instead uses the pre-fit background expectation. The significance
and diagnostic commands use `-t -1 --expectSignal 1` (or the explicitly requested
injection), without `--toysFrequentist`. These are the
[documented pre-fit expected calculations](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/v11.0.0/part3/commonstatsmethods/).

The runner reads ROOT result trees, requiring exactly the five expected limit
quantiles, finite positive ordered limits, one finite positive significance,
`fit_status == 0`, an accurate diagnostic covariance matrix (`covariance quality
== 3`), a finite positive uncertainty, and recovery of the injected `r` within
1% of `max(1, r)`. Observed rows, duplicate quantiles, missing files, invalid
fits and results approaching the requested exclusion boundary are rejected.
The diagnostic is an independent signal-plus-background Asimov fit; its status
does not certify every internal profile fit used for a limit. Logs retain
intermediate evaluation warnings for inspection.

Two implementation details were established by actual numerical tests:

* CMSSW15's uproot 5.3.3 could hang on local FitDiagnostics trees through its
  asynchronous fsspec reader. Local result parsing explicitly uses `MemmapSource`.
* A discovery calculation with a needlessly broad `r` range could return zero
  significance despite a good independent fit. Discovery and diagnostic fits use
  `rMax=max(20, 5*injected_r)` independently of the potentially larger exclusion
  search range. A zero significance for a positive injection is rejected for
  investigation rather than silently ranked. The chosen ranges are recorded.

Limit root finding requests relative accuracy 0.001 and absolute accuracy 1e-6.
These settings do not imply that the asymptotic approximation or all fit
operations have that physical accuracy. Very weak signals may need a larger
`--r-max`; large uncertainties and narrow nuisance domains need separate model
validation.

## Actual numerical validation

The final validation used the durable runtime, not mocked Combine output:

| Card | Median expected `r95` | Expected discovery Z | Total wall time |
| --- | ---: | ---: | ---: |
| Known background, s=10 and b=100 | 2.087402 | 0.983954 | approximately 11 s |
| Same model, observation changed from 100 to 10000 | exactly unchanged | exactly unchanged | 10.37 s |
| Doubled signal, s=20 and b=100 | 1.045227 | 1.938343 | 10.23 s |
| Three-bin representative QCD/ttbar/ST/WJets nuisance card | 2.398682 | 0.806854 | 14.30 s |

For the first counting card, the analytic known-background Asimov significance
is 0.983991645; the difference is 0.0038%. The doubled-signal significance also
matches the analytic expression within 0.1%, and its limit halves within 0.3%.
All diagnostic fits recovered r=1 with status zero and covariance quality three.
All five limit quantiles and Z were exactly unchanged by changing the supplied
observation. Workspace construction took about 5–6 seconds; individual fit
commands, including independent CMSSW setup, took about 1.6–3 seconds. This is a
benchmark for these small cards, not a production-runtime guarantee.

Artifacts and the exact numerical summary are preserved under the durable
installation's `validation/` directory. `test/test_combine_runner.py` separately
checks corrupt output rejection, quantile and diagnostic validation, timeouts,
literal subprocess arguments, stale directories, failure receipts and the blind
command policy. Synthetic counting-card results verify the implementation;
physics candidate ranking must be checked using real normalized templates and
the documented nuisance model.
