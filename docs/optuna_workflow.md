# Automated sensitivity optimization

`optimize_sensitivity.py` proposes configurations with Optuna TPE, schedules one
MiniAOD ntuplizing task per sample/file group, waits for every task, and evaluates
the resulting signal and background campaign. The current fixed 2D search uses
[calibration and frozen-reference campaigns](sensitivity_2d_workflow.md), with
physicality off and a worst-plus-mean normalized Asimov utility. Undefined
statistical scores return the explicit infeasible value `-1`; these trials
cannot win the best-feasible summary or sensitivity-regret rankings. The older
three-signal pilot examples below retain their explicitly configured gate.

For fixed-reference studies, the controller resolves references relative to
the campaign, validates their declared physics/sample definitions, fingerprints
the exact document and snapshots its bytes into the study. Changed references
are rejected before further submission/evaluation. Physical event and generator
digests are checked again when scoring completed ntuples. References are never
updated from the running best trial. The main study summary uses the stationary
utility; retrospective rankings use the final cohort's best per-hypothesis scores.

The controller uses Optuna's [ask/tell interface](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/009_ask_and_tell.html)
so evaluating a trial can take hours on an external scheduler. `constant_liar=True`
discourages similar concurrent proposals. The laboratory AK radius is a categorical choice
between 0.4 and 0.8, with dedicated AK4/AK8 PUPPI corrections. It also varies
collection threshold, CA radius, thrust-assignment cut, gate count and gate
threshold; every proposed gate threshold is at least the collection threshold.

Configure `ak_radii: [0.4, 0.8]` for the two-radius search, or `[0.4]` / `[0.8]`
for a fixed-radius comparison. Other radii and duplicate choices are rejected.
The legacy fixed `ak_radius` option remains readable, but cannot be combined
with `ak_radii`. The worker reads the selected radius from each trial; radius
is part of the ntuple identity, Optuna parameters and immutable study definition.

The default uses correction profile v3 (including the reference b-tag guard), a new study/run directory,
and a new bundle filename. Build a fresh archive and start a fresh study when
moving from the earlier AK8-only search. Old v1/v2 ntuples remain readable with matching campaigns, but cannot be mixed
into a v3 campaign. The v1 profile supports AK8 only.

## Prepare one campaign

1. Build the CMSSW package with the sensitivity cfg and `data/analysis_2017`
   corrections installed. Initialize that release's runtime.
2. Install `requirements-sensitivity.txt` into the controller's Python
   environment, retaining access to the CMSSW ROOT runtime for validation.
3. Resolve MiniAOD source files for **every** signal and background in the
   campaign. Each sample needs `input_files: ["root://...", ...]` or an
   `input_file_list` path relative to the campaign JSON. Input lists are copied
   into the immutable per-trial manifests; changing the lists requires a new
   study. The `files` field is replaced with each trial's produced ROOT shards.
   The example pilot selects the first file per sample (`max_files_per_sample: 1`)
   and at most 10,000 events per job. It records the selected files and available
   file count. Remove the file cap and set `max_events_per_job: -1` for a complete
   production campaign with audited full-dataset normalization.
4. Set cross sections, generated weight sums, generated event counts, and
   complete dataset coverage explicitly. The evaluator validates these against
   the unfiltered ntuple metadata. A limited-event development run must say
   `purpose: "pilot"`, with `normalization_scope: "representative_subset"` and
   `sum_gen_weights: "metadata"`, `generated_events: "metadata"` for each
   sample. This uses all processed events for the denominator and marks the
   result as a pilot. Do not promote a partial campaign to production by merely
   changing its label.
5. Edit `config/optuna_2017.json` to point to this campaign, then create a unique
   archive at its configured `backend.cmssw_bundle` location:

   ```bash
   python3 run_scripts/sensitivity/pack_cmssw.py \
     --cmssw-base "$CMSSW_BASE" \
     --output "$CMSSW_BASE/../sensitivity_bundles/CMSSW_15_0_19_sensitivity_ak4_ak8_v3.tar.gz"
   ```

   The output must be outside the CMSSW release, so choose a different absolute
   location when running from inside its source tree and update the config.
   Existing archives are never overwritten. Archives retain calibration ROOT
   files in `data/`, omit legacy scans and scan ROOT outputs, and preserve
   CMSSW metadata and CVMFS symlinks needed for relocation. The controller freezes
   the archive SHA-256; every worker checks its transferred copy.

## Plan, submit, resume

The default command prepares at most `max_parallel_trials` trial directories and
prints a summary. It **does not submit jobs**, renew credentials or call an
evaluator. It does validate campaign definitions and resolve explicit input
lists, so unresolved background placeholders must be filled first.

```bash
python3 optimize_sensitivity.py --config config/optuna_2017.json \
  --run-dir /local/scratch/aji/suu-sensitivity-2017-ak4-ak8
```

Inspect `trial_000000/trial.json`, `campaign.json`, `tasks/*/inputs.txt`, and
`submit.jdl`. Once the proxy and compiled archive are ready, this terminal
command submits and manages the whole configured study without per-trial prompts:

```bash
python3 optimize_sensitivity.py --config config/optuna_2017.json \
  --run-dir /local/scratch/aji/suu-sensitivity-2017-ak4-ak8 --execute
```

The Condor preflight checks installed commands, the archive, and existing X.509
proxy lifetime. Renew the proxy yourself before launch if needed. Choose a
lifetime long enough for the planned campaign; the configured minimum of one
hour is only a startup lower bound. No real jobs are submitted during tests.

On CMS LPC, `backend.schedd: "auto"` queries available non-draining schedds and
chooses one using the site's load weighting. Its identity is persisted in
`scheduler.json`; all later submissions, queue/history queries and removals
explicitly target that same schedd, including after restart. The driver uses the
installed native binaries when LPC wrappers are present, preferring the
correctly named `/usr/libexec/condor/condor_*` entries. `condor_rm` dispatches
by its command basename, so invoking `original_condor_rm` directly fails.
Successful empty JSON queries are treated as empty lists. Cancellation checks
for live jobs and tolerates jobs finishing before removal only after a
successful queue query confirms no live siblings remain. This avoids
the wrappers' shell-only startup and changing scheduler selection. A fixed
`backend.schedd` can be configured instead; other Condor pools can also override
`schedd_constraint` or the `condor_*_executable` paths.

`--execute --once` performs one reconciliation/submission pass and exits; it can
be run periodically by an already authorized service. Repeating a command with
the same run directory resumes its existing study. Ctrl-C stops the controller
and leaves submitted jobs tracked for the next invocation. `n_trials` is the
total number of proposed trials, including failed and infeasible ones, and can
be increased on resume. `max_parallel_trials` limits concurrently running
configurations; each configuration can contain many sample tasks.

Only one controller can hold the run-directory lock. SQLite should reside on a
local filesystem with normal POSIX locking; do not place its database on NFS.
For shared database storage, configure an explicit supported `storage_url` such
as PostgreSQL and retain the single controller. Optuna's
[storage guidance](https://optuna.readthedocs.io/en/stable/faq.html) describes
SQLite concurrency/NFS limitations and explains that ask/tell workflows must
manage failed trials themselves. Preserve the entire run directory, including
the database, if moving or backing up a study; a run directory is permanently
associated with one database/study and its absolute output paths.

## What completion and recovery mean

Each trial's configuration, campaign, task JSON, source file lists and JDL are
immutable. Study provenance freezes the sample normalization, search definition,
worker/evaluator source hashes and custom Python/shell adapters. A changed
campaign, code, CMSSW archive or study identity requires a new study. Resource,
sample splitting and backend changes also require a new study; trial limits and
polling cadence can be changed on resume.

Before calling `condor_submit -terse`, the controller records a unique trial
token and the `submitting` phase. The token is also a custom job ClassAd.
[condor_q](https://htcondor.readthedocs.io/en/latest/man-pages/condor_q.html) and
[condor_history](https://htcondor.readthedocs.io/en/latest/man-pages/condor_history.html)
are queried as JSON by this token. A lost submission response is reconciled even
if the worker has already left the queue. The controller never repeats an
ambiguous submission. Missing history is given a bounded visibility grace
period; unresolved trials are cancelled by token and failed. Keep scheduler
history longer than the maximum expected controller downtime.

Held, removed, signalled and nonzero-exit jobs fail the trial and cancel its
remaining tasks. Timed-out trials are similarly cancelled. Scheduler command
errors get a bounded number of subsequent polling attempts; persistent outages
stop the controller with running jobs preserved for recovery. There is no blind
resubmission of held/failed jobs. Fix the underlying cause before adding more
trials or starting a replacement study.

Only a complete set of successful jobs with nonempty output ROOTs can be
evaluated. Condor workers also write matching token/sample/task completion
receipts and validate the ROOT schema. Each evaluation attempt uses a fresh
directory; the controller accepts an objective only after a successful evaluator
process, matching configuration identity and finite objective validation. A
stale `objective.json` from an interrupted attempt is never used as evidence of
success. The canonical successful result is
`trial_XXXXXX/evaluation/objective.json`; attempt directories contain the plots,
tables and evaluator logs. `study_summary.json` reports trial states and the best
feasible objective. `study_rankings.json` recomputes per-signal sensitivity
regrets across all completed, globally feasible trials.

## Other terminal submission commands

The `command` backend adapts an existing site wrapper without a shell. Replace
the Condor backend config with explicit JSON argument arrays:

```json
{
  "kind": "command",
  "adapter_revision": "site-wrapper-v1",
  "command_timeout_seconds": 60,
  "preflight_argv": ["/absolute/site-wrapper", "preflight"],
  "submit_argv": ["/absolute/site-wrapper", "submit", "--trial", "{trial_json}", "--token", "{token}"],
  "lookup_argv": ["/absolute/site-wrapper", "lookup", "--token", "{token}"],
  "cancel_argv": ["/absolute/site-wrapper", "cancel", "--token", "{token}"]
}
```

Allowed substitutions are `{trial_json}`, `{trial_dir}`, `{token}`, `{job_id}`
(comma-separated recorded IDs), and `{repo}`. Every substituted value remains a
single argv element; shell expressions are never evaluated. Commands must finish
within their timeout and emit one JSON object on stdout, with diagnostics on
stderr. Submit returns `{"job_ids":["123.0","123.1"]}`. Lookup returns
`{"state":"running|complete|failed|missing","job_ids":["123.0"]}` with an
optional `reason`; it must reconcile by **token**, including finished-job
history. Cancel returns `{}` after removing every live job with that token.
The adapter owns credential preflight and must produce all `trial.json` task
outputs before claiming `complete`; the standard evaluator still verifies their
actual schema, sample identity, corrections and complete normalization.

Custom evaluator arguments may use `{python}`, `{repo}`, `{campaign}`,
`{configuration}`, `{output_dir}` and `{trial_dir}`. The evaluator contract is
the same as `evaluate_sensitivity.py`: write a fresh `objective.json` with the
configuration identity, a finite sensitivity `objective`, boolean `feasible`, and
`per_signal` diagnostics; return nonzero for bad inputs or execution failures.
The controller does not contain a fallback or mock physics objective.

## Validation

```bash
python3 -m unittest test.test_optuna_workflow -v
bash -n run_scripts/sensitivity/worker.sh
```

Tests use real Optuna and SQLite with local fake terminal schedulers to cover
planning, sample splitting, resume, lost submit responses, Condor queue/history
reconciliation, finite objectives, infeasible trials, held jobs, timeouts,
scheduler outages, campaign and archive drift, and lock exclusivity. They launch
no production jobs. An isolated smoke test also packaged the compiled release,
ran the exact transferred `worker.sh`, relocated it with `ProjectRename`, and
processed 20 real UL2017 QCD Pt1800–2400 MiniAOD events over XRootD. The worker
completed corrections/prefiring, wrote schema-v3 ROOT output, passed strict
validation on all 20 events, and produced a matching successful completion
receipt. That initial local test did not submit jobs. The subsequent real
52-job pilot completed transfer, processing, evaluation, and resume validation;
see [batch validation results](sensitivity_validation.md#submitted-batch-pilot).
Its sparse MC statistics did not yield a feasible sensitivity ranking.

The installed HTCondor 25.0.12 `original_condor_submit -dry-run` parser was also
run against the generated four-task JDL. It accepted the description and
resolved distinct task directories, transfer lists, proxy, resources and custom
trial/task ClassAds without submitting a cluster.
