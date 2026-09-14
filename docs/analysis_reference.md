# UL17 analysis reference and scope

The sensitivity workflow targets the supplied **AN-23-067**, with its final
cut-based selections, for 2017 Ultra Legacy MC. The source-code reference is
[SuuToChiChi-analysis-software at 8ef9b506ef4229a2110ec8e662fd5ae93b73420a](https://github.com/emcannaert/SuuToChiChi-analysis-software/tree/8ef9b506ef4229a2110ec8e662fd5ae93b73420a).
The note was supplied locally as `AN_23_067.pdf`; it is not redistributed here.
Its printed page numbers differ from PDF page indices by two.

## Selection reproduced

| Stage | Requirement | Reference |
|---|---|---|
| Trigger | `HLT_PFHT1050_v*` OR `HLT_PFJet500_v*` | AN §3.7, §5.1; `clusteringAlgorithmAll.cc:670-676` |
| Noise filters | goodVertices, globalSuperTightHalo2016, HBHENoise, HBHENoiseIso, EcalDeadCellTriggerPrimitive, BadPFMuon, BadPFMuonDz, eeBadSc, ecalBadCalib | AN §3.8; `preprocess/plugins/noiseFilter.cc:55-70` |
| Electron veto | pT >12 GeV, abs(eta)<2.5, `mvaEleID-Fall17-iso-V2-wpLoose` | AN §3.1-3.2; `clusteringAlgorithmAll.cc:1846` |
| Muon veto | pT >8 GeV, abs(eta)<2.4, CutBasedIdLoose, PFIsoMedium | AN §3.1, §3.3; `clusteringAlgorithmAll.cc:1838` |
| Tau veto | pT >20 GeV, abs(eta)<2.3, decay mode not 5/6/7, decayModeFindingNewDMs, DeepTau2017v2p1 VVLoose VSe / VLoose VSjet / VLoose VSmu | AN §3.1, §3.4; `clusteringAlgorithmAll.cc:1877` |
| AK4 jets | PF CHS, corrected and smeared pT >50 GeV, abs(eta)<2.5, tight ID, pass veto map | AN §3.1, §3.5 |
| AK8 jets | PF PUPPI, corrected and smeared sqrt(pT²+m²)>300 GeV; first two selected jets abs(eta)<2.5, subsequent jets abs(eta)<1.4; tight ID and veto map | AN §3.1; `clusteringAlgorithmAll.cc:1591-1608,3102` |
| Tight jet ID | NHF<0.9, NEMF<0.9, CEMF<0.8, muon energy fraction<0.8, CHF>0, number of constituents>1 | AN Table 13 |
| Heavy AK8 | pT >500 GeV, corrected SoftDrop mass>45 GeV, same ID and eta requirements | AN §3.1; `clusteringAlgorithmAll.cc:3058` |
| HT | Sum of corrected/smeared AK4 pT for pT>50 GeV and abs(eta)<2.5, before tight jet-ID selection | AN §3.1; `clusteringAlgorithmAll.cc:2055-2058` |
| Event baseline | HT>1600 GeV, >=3 AK8, >=4 AK4, and either >=2 heavy AK8 or both selected dijet masses>1000 GeV | AN §5.1 |
| Dijet pairing | Three partitions of first four selected AK4 jets; minimize sqrt(DeltaR(pair1)²+DeltaR(pair2)²) | `clusteringAlgorithmAll.cc:2771-2803` |
| Medium b tag | DeepJet probb+probbb+problepb>0.3040, corrected AK4 pT>70 GeV | AN §3.5.8, §5.1; `clusteringAlgorithmAll.cc:673` |
| Tagged superjet | >=2 CA R=0.4 jets with E>300 GeV in that superjet's rest frame | AN §5.1; `rootProcessor.C:2037-2039` |
| Signal region | At least one medium b-tagged AK4 and both superjets tagged | AN §5.1 |

The CA R=0.4 tag is a fixed analysis-region definition, independent of the
intermediate clustering radii being optimized. The final SR does **not** require
`SJ_mass_100>400 GeV`: those checks are commented out in the current source and
are absent from the final note summary. The control region has no medium b tags
and two tagged superjets; anti-tag regions instead require one tagged superjet
and one with no E>50 GeV CA4 jets and mass(E>100 GeV CA4 sum)<150 GeV.

The early `leptonVeto.cc` filter uses tighter IDs, but the final
`rootProcessor.C:1076` applies the looser lepton counts listed above. Applying
the final veto directly reproduces their combined veto selection. Tight jet-ID
boundaries follow AN Table 13: the old C++ `isgoodjet` accidentally accepts
zero CHF and one constituent, despite calling its criterion tight ID.

The physicality gate is an optimization constraint: the reconstructed chi mass
window is [0.5,1.5] times the generated target mass. It must not select signal
events for the sensitivity histogram using generator truth. Background and
signal sensitivity use the same reconstructed SR selection.

## Nominal calibration and weights

`data/analysis_2017/provenance.json` records every payload's SHA256, original
source, and any extraction. `globaltag_mapping.txt` records the jet conditions
from a live `conddb` query of the original `106X_mc2017_realistic_v10` GlobalTag.

| Object | Nominal JEC | JER resolution / scale factor |
|---|---|---|
| Baseline AK4 CHS | Summer19UL17_V5_MC, L1FastJet + L2Relative + L3Absolute | Summer19UL17_JRV3_MC, AK4PFchs |
| Baseline/reconstruction AK8 PUPPI | Summer19UL17_V5_MC, L2Relative + L3Absolute | Summer19UL17_JRV3_MC, **AK8PF**, matching original GT labels |
| Added reconstruction AK4 PUPPI | Summer19UL17_V5_MC, L2Relative + L3Absolute | Summer19UL17_JRV3_MC, AK4PFPuppi |
| SoftDropPuppi subjets | Summer19UL17_V5_MC, AK4PFPuppi L1FastJet + L2Relative + L3Absolute | Original analysis uses AK4PFchs JER with stochastic smearing |

JEC rescales the raw four-vector. JER broadens MC resolution to match data;
they are separate corrections. The reference Python configuration explicitly
includes L3Absolute (`templates/createCfgTemplate.py:152-175`), while the note
lists only L1/L2. These UL17 L3 payloads evaluate to unity, so including the
source configuration's L3 does not alter the numerical result.

The exact AK8 **AK8PF** JER nomenclature is intentional: it is the label used
in `clusteringAlgorithmAll.cc:2840-2841`, confirmed against the GlobalTag. It
must not be silently replaced with an AK8PFPuppi or AK4 JER payload. The JRV3
text files were obtained from the primary
[CMS JetMET resolution database](https://github.com/cms-jet/JRDatabase/tree/master/textFiles/Summer19UL17_JRV3_MC)
with immutable blob hashes recorded in the manifest.

For PAT jets the nominal hybrid JER follows the stored generator-jet association
when available, otherwise stochastic Gaussian smearing. For the manually
reclustered jets there is no PAT association. AK4 uses `slimmedGenJets` and
AK8 uses `slimmedGenJetsAK8`; nearest matching requires DeltaR<R/2
(0.2 for AK4, 0.4 for AK8) and abs(pT_reco-pT_gen)<3*sigma*pT_reco. Unmatched jets are
smeared stochastically. Random seeds include event identity and jet direction
and are reproducible across processes and trials, improving on the source's
phi-only seed. Factors are applied to **both the reconstructed jet and its
constituent four-vectors** before later frame changes and clustering.
The calibrated reconstruction supports laboratory AK radii 0.4 and 0.8 only.
Other radii are rejected. AK4 uses its own PUPPI JEC and JRV3 PUPPI resolution/SF
payloads, verified against the official CMS JRDatabase and recorded with blob
hashes. The standard AN baseline AK4 CHS corrections remain separate. PUPPI
reconstruction omits L1 pileup corrections at both radii; the provided AK4
PUPPI L1 payload is unity. There is no radius interpolation or extra correction
applied to the COM-frame CA jets.

The event baseline is calculated from the fixed standard PAT jet collections,
so every trial starts from the same AN-selected population. The configurable
reconstruction independently clusters the PUPPI particles at the chosen radius.
AK8 requires fixed ET>300 GeV and its first-two/subsequent eta requirements.
AK4 requires pT>50 GeV and abs(eta)<2.5 for all reconstruction jets. Both use
tight ID and the trial's additional pT threshold and intermediate clustering
parameters. The AK4 option is an extension of the reconstruction search; the
common AN baseline still requires the original standard AK8 and AK4 objects.
Custom reconstruction jets are sorted by corrected pT, whereas the historical
PAT baseline preserves its input order.
The retained jets must also pass the veto map for that trial's SR decision;
vetoed jets discarded by a higher trial threshold do not veto that trial.
Consequently the custom model can change the reconstructed SR efficiency while
preserving the common analysis baseline. Exact event-by-event equivalence to
the original PAT reconstruction is not assumed.

The original nominal `updateJetCollection` JEC uses `fixedGridRhoFastjetAll`;
the source's JER and SoftDrop subjet correction use `fixedGridRhoAll`. These
inputs remain explicit in `analysis2017_cfi.py`.

Nominal weights multiply generator event weight, pileup, fixed-WP medium b-tag
weight, and L1 prefiring. TT samples additionally use the source's
sqrt(exp(0.0615-0.0005*min(pT_top,500))*exp(0.0615-0.0005*min(pT_antitop,500))).
The non-generator factors are produced by the selection helper; the compact
ntuplizer owns generator weights and preselection normalization counters.

Pileup uses `Collisions17_UltraLegacy_goldenJSON`. B tagging uses
`deepJet_mujets` for b/c and `deepJet_incl` for light jets with the original
sample-category efficiency maps (`QCDMC`, `TTbarMC`, `STMC`, `WJetsMC`,
`SuuToChiChi`). The reduced JSON retains those correction entries exactly and
omits unused entries. Fixed-WP method 1a forms the products of tagging and non-tagging probabilities
in data and MC, then divides the event probabilities. The final analysis
[`rootProcessor.C:1234-1236`](https://github.com/emcannaert/SuuToChiChi-analysis-software/blob/8ef9b506ef4229a2110ec8e662fd5ae93b73420a/combinedROOT/rootProcessor.C#L1234)
replaces a nonfinite, negative, or greater-than-100 **event b-tag weight** with
1. The v3 correction profile reproduces this rule, including zero-denominator
bins. It does not clip efficiencies or sanitize individual jet weights.
`analysisBTagWeight` and `analysisBTagWeightFallback` record the result and
fallback for each event; PU, prefiring, top-pT, and generator factors are retained.
Invalid map values outside [0,1], missing payloads, and nonfinite/negative SFs
still raise errors. This guard was exercised by the first batch pilot's
TTJets HT>2500 sample, whose map contains an efficiency of exactly 1. Matching the original implementation,
weights use all selected pT>50 GeV AK4 jets; the region's tagged count uses
pT>70 GeV. High-pT efficiency lookup is capped at each eta row's last nonempty
bin. Missing inputs, invalid maps, and nonfinite corrections raise errors.
The original efficiency maps span abs(eta)<=2.4; selected jets in 2.4-2.5 use
the nearest eta edge bin. This explicit overflow convention replaces unsafe
out-of-range indexing in the original analyzer and should be included in
later b-tag closure validation.

The official CMSSW prefiring producer uses `DataEraECAL=UL2017BtoF`,
`DataEraMuon=20172018`, `UseJetEMPt=False`, both uncertainties 0.2, and AK4 jets
after nominal JEC but before JER, as in `createCfgTemplate.py:231-262`.
`AN2017CorrectedJets` provides that calibrated collection from vendored files.
CMSSW 15 publishes `nonPrefiringProb` as a float, while the original CMSSW 10
analyzer consumed a double. The helper uses the CMSSW 15 float product; this
release adaptation was confirmed with an actual MiniAOD smoke run.

## Samples and normalization

`config/backgrounds_2017.json` contains the 23 background datasets in the
supplied note's Table 9, printed page 15 / PDF page 17. The table was visually
checked against the PDF. Full MINIAODSIM dataset paths come from the pinned
CRAB configuration template, lines 356-393, and can be resolved using DAS.

The chosen samples are ten disjoint QCD pT bins from 170 GeV to infinity,
three TTJets HT bins from 800 GeV to infinity, four WJets samples, and six
single-top channels. The older QCD HT samples and inclusive TT samples are
alternatives; combining them with this set would double count backgrounds.
WW/ZZ appear in the repository but are not part of the supplied note's main
background table. This scan reproduces the note's chosen model, not an
independent claim that all omitted processes have zero yield.

The catalog provides cross sections in pb. Luminosity is 41480 pb^-1, from
the analysis's unrounded 41.48 fb^-1 value. Normalize a generated sample with
`luminosity_pb * cross_section_pb / sum_gen_weights`, where the sum covers the
declared input scope **before every selection**. DAS event counts do not give
signed generator-weight sums. Never divide by selected events. Partial files
are suitable only for clearly labeled pilot estimates, with explicit subset
normalization and enough background statistics to prevent sparse-bin artifacts.

The old `postprocess/return_BR_SF/return_BR_SF.py` constants are per-event
weights tied to historical available event counts, and some differ from the
updated note's cross sections. They cannot be reused for new file subsets.

`config/signal_benchmarks_2017.json` gives three WBWB examples (4/1, 6/2, 8/3
TeV). The explicit theory benchmark is y_uu=y_chi=2, BR(chi->Wb)=0.5,
BR(W->hadrons)=0.6741, using LO production cross sections from the original
`return_signal_SF.py` and AN Table 3. It includes the Suu->chichi phase-space
branching fraction and the fully hadronic decay fraction, without applying
an extra NLO K factor. These are per-decay cross sections, not the sum of six
decay channels. Full-analysis sensitivity requires the desired signal decays
and mass points to be represented with a consistent benchmark.

This implementation supplies nominal MC optimization sensitivity. The note's
final inference additionally uses its data-driven background estimate, shape
and normalization nuisance parameters, region correlations, and Combine fit.
An optimization score from nominal signal/background MC is not that final
expected exclusion limit. Validate a selected model on independent MC before
interpreting improvements from a search over many configurations.
