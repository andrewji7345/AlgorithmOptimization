# UL2017 background pilot inputs

These 23 lists contain **one file per dataset**, resolved from the exact
AN-23-067 background catalog on 2026-09-10. They are inputs for a labeled
representative-subset pilot, not complete production samples. DAS returned
8,510 available files across the 23 datasets.

Each adjacent JSON records the dataset, DAS command, resolution timestamp,
available/selected file counts, source-catalog hash, and list-content hash.
The selected file is the lexicographically first validated DAS result. This
deterministic choice makes repeated trials comparable; representativeness is
an explicit pilot assumption.

From the repository root, refresh the pilot lists with an existing CMS proxy:

```bash
python3 resolve_samples.py --max-files 1
```

Omit `--max-files` to resolve complete dataset file lists. The resolver updates
`config/sensitivity_2017.json` only after every requested query succeeds. It
does not produce generator-weight denominators: these still come from all
processed preselection events in the declared normalization scope.

The catalog and cross sections are in
[`config/backgrounds_2017.json`](../../config/backgrounds_2017.json).
