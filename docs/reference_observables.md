# Reference reconstruction and likelihood observables

The additive `analysisObservableVersion=1` contract supplements schema 3. Nominal
hybrid reconstruction and the correction prescription ending in `btagGuard-v3`
retain their previous meaning. New likelihood comparisons use the separately
identified reference reconstruction, all four analysis regions, and alternate
absolute event weights. These additions make a controlled comparison possible;
they do not certify reproduction of the complete published analysis.

## Reconstruction source and differences

The source is `preprocess/plugins/clusteringAlgorithmAll.cc` and its `sortJets.cc`
helper at commit `8ef9b506ef4229a2110ec8e662fd5ae93b73420a` of the
[original analysis](https://github.com/emcannaert/SuuToChiChi-analysis-software/tree/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/preprocess/plugins).
The metadata name is `AN23-067-PATAK8-CA8-Thrust-source-port-v1`.

There are two independent anchors for the reclustering mode: the supplied AN
§4.2, lines 591–592 explicitly describes Cambridge/Aachen R=0.8 clustering in
the event MPP frame, and the pinned
[`templates/createCfgTemplate.py:13`](https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/templates/createCfgTemplate.py#L13)
sets `skipReclustering=False`, then writes that value into the analyzer PSet at
line 377. The alternate branch that skips event-frame reclustering is therefore
not the mode described by the AN or that template.

The public template sets `useOptimizedWP=True` at line 12, but its initial
`slimmedSelection=True` is **overridden inside the systematic loop at line 269**:
nominal analyzers and every signal analyzer receive `slimmedSelection=False`.
This distinction matters when reading
[`clusteringAlgorithmAll.cc:394–460`](https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/preprocess/plugins/clusteringAlgorithmAll.cc#L394).
With the optimized switch true but slimming false, the nominal/signal analyzer
keeps its looser preselection defaults: HT ≥ 1500 GeV, AK8 ET ≥ 300 GeV, at least
two AK8 jets and at least one heavy AK8 jet, with the alternative dijet-mass path.
The final `rootProcessor.C:1461–1499` then applies HT ≥ 1600 GeV, at least three
AK8 jets and at least two heavy AK8 jets (or the dijet alternative), consistent
with those requirements in the supplied AN. These upstream cuts do not explain
the observed nominal signal anti-tag leakage.

The same public template *does* introduce a different preselection for
**nonnominal background** analyzers: they retain `slimmedSelection=True` together
with `useOptimizedWP=True`, which selects HT ≥ 2200 GeV, AK8 ET ≥ 400 GeV and at
least three heavy AK8 jets (or the dijet alternative). This inconsistency is
specific to the public template's shifted-background production. The current
reference deliberately keeps the supplied AN baseline for every kinematic state
and recomputes acceptance migrations through the same cuts. It does not copy a
variation-dependent tightening of the selection.

The committed template is also not directly executable: stray unindented
identifiers at lines 48–55 cause a Python indentation error at line 58. The
generated `allCfgs` files are absent from the pinned Git tree. Its visible values
establish public configuration intent, not which executable configuration
produced the historical AN histograms. Agreement with an independent port of
the clustering source likewise does not establish historical configuration or
full analysis closure. In particular, the template's stronger shifted-background
cuts must not be presented as an explanation of the nominal signal discrepancy.

The upstream filter audit found no omitted tighter lepton cut that could resolve
that discrepancy: the early `leptonVeto` uses medium muon ID, electron wp90 and
tighter tau IDs, while the final processor vetoes the looser lepton definitions
used here. `hadronFilter` is commented out in the template's path construction
(lines 456–458), and pileup jet ID is evaluated only below 50 GeV whereas selected
AK4 jets have pT above that threshold. Other source/AN details remain distinct:
the final processor has no explicit four-AK4 requirement and loops only over AK8
jets for its veto-map rejection, while the AN-based baseline here requires four
AK4 jets and checks both selected collections. Those stricter choices do not
establish an explanation for higher absolute signal anti-tag acceptance.

Jet ordering is another end-to-end validation item. The public template uses
`updatedPatJetsAK8UpdatedJEC` and updated AK4 collections; CMSSW's PATJetUpdater
sorts by corrected pT by default. The current baseline keeps the input MiniAOD
collection order while recalculating corrections. If those new corrections
change the ordering, the leading-four AK4 pairing or the first-two AK8 eta rule
could change. Its numerical impact has not been measured. None of these audit
findings was used to alter the frozen validation campaign or to claim agreement
with unavailable historical production files.

The reference starts from daughters of the selected PAT AK8 jets used by the
fixed AN baseline. Each daughter receives its PUPPI weight and its parent jet's
JEC/JER scale. Their summed four-vector defines the event COM boost. In that
frame, Cambridge/Aachen R=0.8 clustering uses `inclusive_jets(10)`, which is a
10 GeV **pT** threshold in FastJet, notwithstanding the source's neighboring
comment about energy. Clusters are sorted by energy.

Only daughters of clusters with at least five constituents enter CMS
`PhysicsTools/CandUtils/Thrust`. The source's explicit 300-particle guard is
retained. Clusters with at least two constituents enter the hybrid partition:
cosine strictly above +0.85 or below −0.85 fixes a side, and the remaining jets
are assigned to minimize `abs(m1-m2)/min(m1,m2)`. The original nested-loop order,
direct four-vector accumulation and fourteen-ambiguous-jet maximum are used for
this reference. The scan's own configurable ambiguity guard remains separate.
Clusters with only two to four constituents affect that partition but, following
the source, are then omitted from the final masses and superjet substructure.
Each superjet's constituents are boosted to its rest frame and reclustered with
Cambridge/Aachen R=0.4.

Several differences are intentional and must be considered when auditing
agreement with an old ntuple. This port uses the project's deterministic
event-and-jet JER seeds instead of the original phi-only seeds. It stores
superjets ordered by lab pT, while the source stores thrust-side order; symmetric
region definitions and the average superjet mass are unaffected. It identifies
cluster membership directly instead of matching four-vectors by an angular
tolerance. Zero-weight PUPPI daughters are retained as in the source. Numerical failures and
guarded events remain in the output with a nonzero reference status, preserving
preselection normalization; the original analyzer returns without writing them.
The shared baseline continues to use this project's existing UL17 correction
implementation. This is a source port for reference comparisons, not a claim of
bitwise equivalence to historical ROOT files.

Reference status uses the existing 0–8 reconstruction status table. Status 6
includes either the source's 300-thrust-particle guard or its fourteen-ambiguity
limit. Invalid reference masses are NaN and `referenceRegion=0`.

## Region observables

`sj1NCA4E50`, `sj2NCA4E50`, `sj1MassE100`, and `sj2MassE100` are configuration
arrays. The corresponding `referenceSJ*` branches are scalar values for the
fixed source port. `MassE100` is the invariant mass of the sum of rest-frame CA4
four-vectors whose energies exceed 100 GeV, and is zero when that set is empty.
The existing NCA4E300 branches remain available.

The region boundaries follow AN §5.1/§5.4 and `combinedROOT/rootProcessor.C`:

| Classification | Medium b tags at AK4 pT > 70 GeV | Region code |
|---|---|---|
| Both superjets have at least two CA4 jets with E > 300 GeV | At least one | 1, SR |
| Both tagged | Zero | 2, CR |
| One tagged, one anti-tagged | At least one | 3, AT1b |
| One tagged, one anti-tagged | Zero | 4, AT0b |

An anti-tag has zero CA4 jets with E > 50 GeV and `MassE100 < 150 GeV`. The latter
requirement is redundant under the former, but both are preserved explicitly.
Every region requires the baseline and valid reconstruction. Hybrid models also
require their reconstruction jet-veto flag. The reference needs no extra veto
because its constituent parents already belong to the baseline collection.
Code zero means no selected region. The hybrid arrays are
`passesSignalRegion`, `passesControlRegion`, `passesAT1b`, and `passesAT0b`.

`analysisBTagJetPt`, `analysisBTagJetEta` and `analysisBTagJetDiscriminator` store
the selected PAT AK4 jets entering the b-tag weight. Their length equals
`analysisNAK4`. These branches allow counting thresholds to be audited; changing
the b-tag working point still requires matching scale factors and efficiency
maps and is not implemented by merely applying another discriminator cut.

## Weight conventions

`analysisWeight` retains the previous v3 nominal convention, including nominal
top-pT reweighting for ttbar. The source processor instead uses **no top-pT
correction in its nominal prediction**, and reads the stored top-pT weight only
for its one-sided variation (`rootProcessor.C:73,835,996,1179`). Consequently,
`referenceWeight` stores pileup × prefiring × medium-b-tag event weight without
that correction. The likelihood comparison must use `referenceWeight` for both
the source port and every hybrid model, so their normalization conventions agree.
Generator weights are separate and must be multiplied exactly once.

The `weightVariationNames` metadata orders fourteen absolute weights in both
`referenceWeightVariations` and `analysisWeightVariations`:

1. `pileupUp`, `pileupDown`: official UL17 payload variations.
2. `prefiringUp`, `prefiringDown`: the prefiring producer's varied probabilities.
3. `btagHFCorrelatedUp/Down`, `btagHFUncorrelatedUp/Down`,
   `btagLFCorrelatedUp/Down`, `btagLFUncorrelatedUp/Down`: medium DeepJet method-1a
   event weights using the source's flavor split and official correlated or
   uncorrelated correction labels.
4. `topPtUp`, `topPtDown`: for `referenceWeightVariations`, up applies the
   source's `sqrt(SF(top1) SF(top2))`, with `SF=exp(0.0615-0.0005 min(pT,500))`,
   while down equals nominal (the original one-sided convention). The legacy
   `analysisWeightVariations` instead supplies squared correction versus no
   correction around its already-weighted nominal; it must not be mistaken for
   the source convention or used in the reference likelihood.

The vectors contain total alternate weights, not ratios. This handles zero
nominal weights without division. B-tag scale-factor variations can extrapolate
below zero at very high pT; as in the source, the **final event probability ratio**
receives the existing finite/0–100 guard and fallback to one. Individual jet
factors are not clipped. `btagWeightVariationFallback` records the eight alternate
b-tag fallback decisions, in the same order as positions 4–11 of the weight
vectors. The nominal decision remains `analysisBTagWeightFallback`.

## Kinematic shape variations

`analysisSystematic=nominal|JECUp|JECDown|JERUp|JERDown` selects one state per job.
All events, including those that migrate across a selection boundary, are kept.
JEC/JER shifts rerun baseline selections, custom reconstruction, source-port
reconstruction, masses and regions; they are not post-hoc histogram rescalings.

The total JEC uncertainties for AK4 PFchs, AK4 PUPPI and AK8 PUPPI are extracted
from the original analysis's pinned `Summer19UL17_V5_MC.tar.gz`. SHA256 hashes and
provenance are in `data/analysis_2017/provenance.json`. JEC shifts affect the
appropriate nominally corrected jets and soft-drop subjets, and the jets used by
the prefiring producer. JER up/down uses the official scale-factor variations
with the same random seed per event and jet. Prefiring remains evaluated before
JER, matching the existing prescription. Weight variations are evaluated using
the shifted selected objects within each kinematic state.

This is a representative **total-JEC envelope**, not the final AN's multiple JEC
source correlation model. Similarly the JER envelope is global rather than the
AN's additional eta split. PDF/renormalization/factorization variations,
year-combination correlations, original two-dimensional superbinning and data
closure are not supplied by these ntuple additions. Their absence must remain
explicit in any final likelihood validation claim.

## Development validation

The production C++ region helper is tested at every tag, anti-tag, validity and
b-tag category boundary, including a sweep over physically consistent count
combinations. The CMSSW plugin was compiled and exercised on 100 events from the
4/1-TeV WBWB signal in all five kinematic states, and on 100 events from the
highest-HT ttbar sample. A matched configuration reproduced all 17 existing
scalar and nine existing reconstruction/region branches exactly against the
previous v3 pilot. Thirteen reference branches agreed exactly between AK4 and
AK8 reconstruction jobs on the same signal events. The ttbar test also exercised
alternate b-tag event-weight fallbacks in six of its 100 events.

An independent reconstruction audit captured the corrected PAT AK8 daughter
four-vectors for those 100 signal events. A separate executable used the
unmodified, pinned original `sortJets.cc`, CMS Thrust and independently
transcribed source clustering/matching loops. Statuses and all nine mass/tag
observables agreed with the port for every event (mass tolerance 1e-5 relative,
0.0005 GeV absolute). One event hit the source complexity guard. A second capture
explicitly retained zero-weight PUPPI daughters; there were none in the selected
AK8 jets of this sample, so it did not change any result.

The audit found **11 AT1b and 23 SR events out of 100 generated signal events**,
with 53 passing the nominal baseline. These anti-tag events already have one
low-mass superjet before the final constituent-multiplicity filter; none of their
CA8 clusters is removed by that filter. The independent source implementation
reproduces them. This disagrees with a naive application of the AN's general
1–2% signal-contamination expectation to this small, low-mass benchmark sample.
It is an unresolved physics-closure check, not evidence of AN-level closure.
Larger benchmark-specific samples and an end-to-end comparison with historical
analysis outputs remain necessary before calling the reference fully validated.
