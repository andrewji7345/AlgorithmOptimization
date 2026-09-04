# Compact grouped optimization scan

This workflow runs the expanded compact ntuplizer with one Condor job per
signal sample and AK radius. All collection-threshold, event-gate, CA-radius,
and hybrid-assignment points configured by
`runCompactOptimizationScan_cfg.py` are evaluated together inside that job.
This grouping avoids repeating the PUPPI candidate preparation and AK
clustering for configurations that share an AK radius.

The workflow is separate from every legacy/two-threshold scan. It uses its own
scripts, manifest, logs, tarball names, EOS output directory, and ROOT-file
name prefix.

## Job counts and defaults

The submit script discovers and validates all 114 nonempty `*.txt` lists in
`test/signalMCFiles`; sample names are never maintained in a second hard-coded
table.

- `--test` is the safe default: `WbWb_4000_1000` at AK R=0.8, for one job.
- `--full` uses all 114 samples and AK R=0.4 through 1.6 in steps of 0.2,
  for 798 jobs.
- The default limit is 1,000 events per sample. Use `--max-events 10000` for
  a higher-statistics scan. Any positive 32-bit integer is accepted.
- `--ak-radii 0.6,0.8,1.0` can define a smaller pilot or alternate grid.

The remaining scan grids come from the cfg/cfi defaults, so the Condor
manifest does not duplicate their definitions.

## Validate without submitting

From this directory, run:

```bash
bash submit_ntuplizer_scan_compact_grouped.sh --test --dry-run
bash submit_ntuplizer_scan_compact_grouped.sh --full --max-events 10000 --dry-run
```

Dry-run mode performs local validation and atomically writes
`scan_parameters_compact_grouped.tsv`. It does not inspect a proxy, build or
upload a tarball, contact EOS, or invoke `condor_submit`. Inspect the reported
job count and first manifest rows before a real submission.

## Submit

Initialize a CMS proxy, then run one of:

```bash
voms-proxy-init --valid 192:00 -voms cms
bash submit_ntuplizer_scan_compact_grouped.sh --test --max-events 1000
bash submit_ntuplizer_scan_compact_grouped.sh --full --max-events 10000
```

Each invocation generates a UTC/PID run tag. For a meaningful stable label,
pass a unique value such as `--tag global-v1-10k`. The tag appears in the
tarball and every output filename:

```text
compactOptimization_<tag>_<channel>_<mSuu>_<mChi>_akR<radius>_nev<N>.root
```

Decimal points in the radius are replaced with `p`, for example `akR0p8`.
Outputs are copied to:

```text
root://cmseos.fnal.gov//store/user/aji/rootfiles_existingOptimization_compact/
```

The worker intentionally does not use `xrdcp --force`; reusing a tag therefore
fails instead of silently replacing an earlier ntuple. Successful ROOT files
must also pass `validate_compact_scan.py --strict-branches` before upload, so
schema or storage-invariant failures remain in job scratch and are visible in
the Condor error stream. Valid ROOT files remain only on EOS, while Condor
returns log streams under `logs_compact_grouped/`.

Generated local artifacts are:

- `scan_parameters_compact_grouped.tsv`: the active queue manifest;
- `logs_compact_grouped/`: Condor stdout, stderr, and scheduler logs.

The run-specific CMSSW tarball is created under `/tmp`, uploaded to
`/store/user/aji/condor_inputs`, and removed locally. The packaged area
excludes ROOT files, prior tarballs, scan logs/results, and unrelated
ParticleTransformer sources to avoid recapturing large study products.
