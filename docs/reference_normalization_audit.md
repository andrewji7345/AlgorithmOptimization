# Background normalization audit, 2017 reference pilot

Read-only audit performed on 2026-09-10. No production, evaluator, catalog, or frozen comparison inputs were changed.

The current catalog uses the supplied AN's updated 2017 cross sections correctly. The WJetsToLNu HT2500+ yield difference is largely explained by an older normalization in the pinned analysis source, rather than evidence for an extra factor in the current catalog. Table 19 must therefore be treated as a diagnostic comparison with potentially different normalization provenance.

## Evidence and scope

- Supplied note: `/uscms/home/aji/nobackup/research/CMSSW_15_0_19/src/AN_23_067.pdf`, text `/tmp/AN_23_067.txt`; 2017 MC normalization table on physical page 15, Table 19 on physical page 44. The version-7 change log explicitly says that background cross-section tables were updated to the era-specific XSDB values. It does not establish that every cutflow table was regenerated.
- Pinned public source: `8ef9b506ef4229a2110ec8e662fd5ae93b73420a` of [SuuToChiChi-analysis-software](https://github.com/emcannaert/SuuToChiChi-analysis-software/tree/8ef9b506ef4229a2110ec8e662fd5ae93b73420a). The local checkout `/uscms_data/d3/aji/SuuToChiChi_analysis_software` reports this commit; relevant normalization and linearizer files are clean. The normalization source was also downloaded independently to `/uscms/home/aji/nobackup/research/sensitivity_runs/reference_validation_20260910/validation/normalization_audit/return_BR_SF.py`.
- [return_BR_SF.py](https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/postprocess/return_BR_SF/return_BR_SF.py#L15) contains historical full-sample scaling constants. [master_linearizer.py](https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/postprocess/master_linearizer.py#L3055) retrieves the four W+jets constants and multiplies their separate histograms by them at lines 3163–3166. `combinedROOT/analyzers/readTree_test.C` also includes the same constants and applies `MC_SF` to cutflow counters at line 1169, but this is not proof that this exact script generated the supplied Table 19.
- Current input: `/uscms/home/aji/nobackup/research/sensitivity_runs/reference_validation_20260910/campaign.json`. All 23 background cross sections exactly match the corresponding updated AN 2017 rows. The dataset paths match the pinned original template and the supplied AN dataset list. Every background has `filter_efficiency=1` and `k_factor=1`; the catalog does not apply an additional W decay branching factor to already decay-specific datasets.
- Actual pilot estimates use both disjoint AK4 reference folds from `production_validated`, pooled using their processed generator-weight denominators. The raw coverage information is in `/uscms/home/aji/nobackup/research/sensitivity_runs/reference_validation_20260910/validation/normalization_audit/reference_coverage_audit.json`.
- Full numerical comparison for all 23 backgrounds, including source hash: `/uscms/home/aji/nobackup/research/sensitivity_runs/reference_validation_20260910/validation/normalization_audit/reference_normalization_comparison.json`.

## WJetsToLNu HT2500+

The AN gives cross section 0.02646 pb and 1,185,699 generated events. At 41,480 pb⁻¹, its full-sample constant is

```
41480 * 0.02646 / 1185699 = 0.0009256656200
```

The pinned `return_BR_SF.py:29` instead uses `0.0002799036518`. Combining that constant with the AN event count implies `0.008001000000858443 pb`, to numerical precision a round older cross section of 0.008001 pb. The ratio between the current AN convention and this historical convention is **3.3070866138**.

Consequently, if Table 19 retained this historical normalization, its 0.28 SR events would become **0.9259842519** with the updated cross section. The two-fold pilot estimate is **1.1055 ± 0.1756** from MC statistics, only approximately 1.02 pilot standard deviations above that rescaled value. It would instead appear approximately 4.7 standard deviations above the unadjusted 0.28 if the note's MC uncertainty were ignored.

The source/AN normalization mismatch is proven by the constants. Its explanation of Table 19 is a strong consistency inference; the exact cutflow-production provenance and the table's own MC uncertainty have not been recovered. Do not change the current catalog to force numerical agreement with the unadjusted table.

## Other backgrounds

The same inversion, `source_constant * AN_generated_events / luminosity`, gives these informative comparisons. This assumes the historical constants used the listed full event counts; they must never be applied directly to the new pilot subset.

| Process | Current AN cross section, pb | Cross section implied by pinned constant, pb | Pinned/current normalization |
|---|---:|---:|---:|
| WJetsToQQ HT800+ | 29.1 | 28.75 | 0.98797 |
| WJetsToLNu HT800–1200 | 4.926 | 5.366 | 1.08932 |
| WJetsToLNu HT1200–2500 | 1.152 | 1.16 | 1.00694 |
| WJetsToLNu HT2500+ | 0.02646 | 0.008001 | 0.30238 |
| TTJets HT800–1200 | 0.5581 | 0.7532 | 1.34958 |
| TTJets HT1200–2500 | 0.09876 | 0.1316 | 1.33252 |
| TTJets HT2500+ | 0.001124 | 0.001407 | 1.25178 |
| ST s-channel hadronic | 7.104 | 11.24 | 1.58221 |
| ST s-channel leptonic | 3.549 | 3.74 | 1.05382 |
| ST tW antitop | 32.51 | 34.97 | 1.07567 |
| ST tW top | 32.45 | 34.91 | 1.07581 |

The ten QCD differences are between −0.48% and +1.23%; ST t-channel top agrees and antitop differs by −0.014%. There is no evidence from these comparisons for missing WJetsQQ branching or filtering factors. Its source normalization differs by only 1.2%. Zero selected WJetsQQ events in the current small pilot remain an MC coverage problem, not a normalization test.

As a further consistency check, the TTJets HT1200–2500 Table-19 value 22.35 becomes 16.7727 under the same historical-to-current rescaling. The pooled pilot gives 16.4740 ± 1.6465. This independent agreement strengthens the older-cutflow-normalization interpretation. It does not prove selection equivalence or make it appropriate to renormalize all note tables without their original provenance.

## Generator weights and source caveats

Current weighting uses the signed generator weight for each event and its signed sum over **every processed event**, multiplied by the current cross section, luminosity, and per-event corrections. QCD, TTJets, and WJets events in the returned shards all have unit generator weights, so signed-weight conventions do not explain their yield differences. Single-top samples do have nonunit and sometimes negative generator weights. For example, the s-channel hadronic shard has ±10.9187 generator weights and a mean of approximately 7.18995; the t-channel top shard has ±120.394 and a mean near 119.756. Counting entries instead of summing their signed weights would change those estimates. The original `rootProcessor.C` nominal event correction uses pileup and prefiring and does not read a generator-weight branch, which is another reason not to claim full single-top yield reproduction merely by copying historical constants.

The pinned `return_BR_SF` function also has a literal-string `elif` in its ST dispatch, causing every ST sample except t-channel top to take the t-channel antitop branch. The audited `master_linearizer.py` ST block uses separate hard-coded ST dictionaries instead, so this dispatch defect cannot be asserted to affect that path. The current catalog/evaluator do not import the legacy function. No source bugs were copied or changed in this audit.

Recommended action: retain the current updated-AN catalog and signed subset normalization, preserve this discrepancy explanation in comparison provenance, and keep absolute AN cutflow reproduction explicitly unaudited. Continue the targeted MC expansion with the same corrected convention for both the reference anchor and every hybrid candidate.
