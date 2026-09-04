import FWCore.ParameterSet.Config as cms

existingOptimizationNtuplizer = cms.EDAnalyzer(
    "ExistingOptimizationNtuplizer",

    packedPFCandidates = cms.InputTag("packedPFCandidates"),

    vertices = cms.InputTag("offlineSlimmedPrimaryVertices"),

    ak4Jets = cms.InputTag("slimmedJets"),

    ak8Jets = cms.InputTag("slimmedJetsAK8"),

    met = cms.InputTag("slimmedMETs"),

    rho = cms.InputTag("fixedGridRhoFastjetAll"),

    genParticles = cms.InputTag("prunedGenParticles"),

    genJets = cms.InputTag("slimmedGenJets"),

    genAK8Jets = cms.InputTag("slimmedGenJetsAK8"),

    isMC = cms.bool(True),

    # minEventJets=0 preserves the original single-threshold reconstruction.
    # Dedicated two-threshold jobs override it to 4 and lower jetPtCut.
    jetPtCut = cms.double(300.0),
    eventJetPtCut = cms.double(300.0),
    minEventJets = cms.uint32(0),

    akRadius = cms.double(0.8),
    caRadius = cms.double(0.8),
    cosThrust = cms.double(0.85)
)
