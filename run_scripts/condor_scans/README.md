# ExistingOptimization CMSLPC submission

Copy this directory anywhere under your CMSLPC `~/nobackup` area. Submit from
an AlmaLinux 9 interactive node so the jobs automatically use EL9 containers.

The worker nodes do not mount `~/nobackup` or `/uscms_data`. `submit_scan.sh`
therefore creates a compact tarball of `CMSSW_15_0_19`, uploads it once to EOS,
and each worker downloads and unpacks that tarball in its local scratch area.

## Before submission

```bash
cd ~/nobackup/research/CMSSW_15_0_19/src
cmsenv
scram b -j 8

voms-proxy-init --valid 192:00 -voms cms
```

Check these two paths in `submit_scan.sh` if the capitalization in your CMSSW
area differs:

```text
SuuAnalysis/ExistingOptimization/test/runExistingOptimizationNtuplizer_cfg.py
SuuAnalysis/ExistingOptimization/test/signalMCFiles
```

The second path follows the location given with this request. Both paths are
validated before the tarball is made or any jobs are submitted.

## Submit one test point first

```bash
chmod +x run_existingOptimization_point.sh submit_scan.sh
./submit_scan.sh --test
```

This submits only `WbWb_4000_1000`, `pT=100`, `AK=0.4`, `CA=0.4`, and
`cos(thrust)=0.85`. Verify it before launching the full scan:

```bash
condor_q
tail -f logs/job_*.out
xrdfs root://cmseos.fnal.gov ls /store/user/aji/rootfiles_existingOptimization
```

## Submit the full scan

```bash
./submit_scan.sh --full
```

With one thrust value (`0.85`), the full configuration contains 9,792 jobs.
The submit description limits materialization and idle jobs so these are fed
to the scheduler gradually.

To scan additional thrust values, edit this line in `submit_scan.sh`:

```bash
thrust_ints=(80 85 90)
```

That represents values `0.80`, `0.85`, and `0.90`.

## Outputs

Successful ROOT files are written directly to:

```text
/eos/uscms/store/user/aji/rootfiles_existingOptimization/
```

through the XRootD endpoint `root://cmseos.fnal.gov/`. The worker removes its
local ROOT file after a successful copy, preventing Condor from returning a
duplicate through NFS.

## Important scan correction

If instead every AK radius should be paired with every CA radius from 0.4 to
1.6, replace the `ca_start_int` block in `submit_scan.sh` with:

```bash
ca_start_int=4
```

