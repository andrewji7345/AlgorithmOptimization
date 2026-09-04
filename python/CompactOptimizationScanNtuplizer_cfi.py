import FWCore.ParameterSet.Config as cms


compactOptimizationScanNtuplizer = cms.EDAnalyzer(
    "CompactOptimizationScanNtuplizer",
    packedPFCandidates=cms.InputTag("packedPFCandidates"),
    generatorInfo=cms.InputTag("generator"),
    sampleName=cms.string("unknown"),
    akRadius=cms.double(0.8),

    # T_keep: all uncorrected, custom PUPPI AK jets above this threshold
    # contribute constituents to the reconstructed COM system.
    collectionPtCuts=cms.vdouble(
        100, 120, 140, 160, 180, 200, 220, 240,
        260, 280, 300, 320, 340, 360, 380, 400,
    ),
    caRadii=cms.vdouble(0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6),

    # c=0 is explicitly pure thrust. c=1 (pure exhaustive mass assignment)
    # is supported but intentionally absent from production defaults because
    # high CA multiplicities make it expensive; the per-event guard still
    # protects any custom scan that includes it.
    cosThrustCuts=cms.vdouble(
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
        0.6, 0.7, 0.8, 0.9, 0.95,
    ),

    # These gates are metadata only. The event tree stores sorted AK pT once,
    # so an evaluator can construct every legal (n, T_gate, T_keep) choice
    # without repeating reconstruction or multiplying the output size.
    defaultGateJetCounts=cms.vuint32(0, 1, 2, 3, 4, 5, 6),
    defaultGatePtCuts=cms.vdouble(
        100, 120, 140, 160, 180, 200, 220, 240,
        260, 280, 300, 320, 340, 360, 380, 400,
    ),

    maxAmbiguousCAJets=cms.uint32(12),
    enforceLegacyRadiusConstraint=cms.bool(True),
    compressionLevel=cms.uint32(6),
)
