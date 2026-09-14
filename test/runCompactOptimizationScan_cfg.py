from pathlib import Path

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


options = VarParsing("analysis")
options.setDefault("maxEvents", 1000)
options.register(
    "inputRootFiles",
    "SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Text file containing MiniAOD ROOT file names",
)
options.register(
    "outputRootFile",
    "compact_optimization_scan.root",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Compact ROOT output file",
)
options.register(
    "sampleName",
    "",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Sample name stored in metadata (default: input-list stem)",
)
options.register(
    "akRadius",
    0.8,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    "One anti-kT radius for this grouped job",
)
options.register(
    "collectionPtCuts",
    "100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Comma-separated T_keep grid in GeV",
)
options.register(
    "caRadii",
    "0.4,0.6,0.8,1.0,1.2,1.4,1.6",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Comma-separated Cambridge/Aachen radii",
)
options.register(
    "cosThrustCuts",
    "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Comma-separated hybrid assignment cuts; 0=pure thrust, 1=pure mass",
)
options.register(
    "defaultGateJetCounts",
    "0,1,2,3,4,5,6",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Comma-separated gate multiplicities stored in metadata",
)
options.register(
    "defaultGatePtCuts",
    "100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400",
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    "Comma-separated T_gate grid in GeV stored in metadata",
)
options.register(
    "maxAmbiguousCAJets",
    12,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "Per-configuration exhaustive-assignment guard",
)
options.register(
    "enforceLegacyRadiusConstraint",
    True,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.bool,
    "Keep only CA >= max(0.4, AK-0.2)",
)
options.register(
    "compressionLevel",
    6,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    "ROOT ZSTD compression level",
)
options.register("analysisMode", False, VarParsing.multiplicity.singleton,
                 VarParsing.varType.bool, "Apply the calibrated AN-23-067 UL2017 selection")
options.register("sampleKind", "signal", VarParsing.multiplicity.singleton,
                 VarParsing.varType.string, "signal or background")
options.register("analysisSystematic", "nominal", VarParsing.multiplicity.singleton,
                 VarParsing.varType.string, "nominal, JECUp, JECDown, JERUp or JERDown")
options.parseArguments()
if not options.analysisMode and options.analysisSystematic != "nominal":
    raise ValueError("Kinematic systematic variations require analysisMode=True")


def parse_csv(raw, converter, option_name):
    try:
        values = [converter(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"Invalid {option_name} value in {raw!r}") from error
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def read_file_list(path):
    with open(path, encoding="utf-8") as handle:
        files = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not files:
        raise ValueError(f"Input list is empty: {path}")
    return files


collection_pt_cuts = parse_csv(options.collectionPtCuts, float, "collectionPtCuts")
ca_radii = parse_csv(options.caRadii, float, "caRadii")
cos_thrust_cuts = parse_csv(options.cosThrustCuts, float, "cosThrustCuts")
gate_jet_counts = parse_csv(options.defaultGateJetCounts, int, "defaultGateJetCounts")
gate_pt_cuts = parse_csv(options.defaultGatePtCuts, float, "defaultGatePtCuts")
sample_name = options.sampleName or Path(options.inputRootFiles).stem

process = cms.Process("COMPACTSCAN")
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 100
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(options.maxEvents))
process.options = cms.untracked.PSet(
    numberOfThreads=cms.untracked.uint32(1),
    numberOfStreams=cms.untracked.uint32(0),
    wantSummary=cms.untracked.bool(True),
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(*read_file_list(options.inputRootFiles)),
)
process.TFileService = cms.Service(
    "TFileService",
    fileName=cms.string(options.outputRootFile),
    closeFileFast=cms.untracked.bool(True),
)

from SuuAnalysis.ExistingOptimization.CompactOptimizationScanNtuplizer_cfi import (
    compactOptimizationScanNtuplizer,
)

process.compactScan = compactOptimizationScanNtuplizer.clone(
    sampleName=cms.string(sample_name),
    sampleKind=cms.string(options.sampleKind),
    analysisMode=cms.bool(options.analysisMode),
    akRadius=cms.double(options.akRadius),
    collectionPtCuts=cms.vdouble(*collection_pt_cuts),
    caRadii=cms.vdouble(*ca_radii),
    cosThrustCuts=cms.vdouble(*cos_thrust_cuts),
    defaultGateJetCounts=cms.vuint32(*gate_jet_counts),
    defaultGatePtCuts=cms.vdouble(*gate_pt_cuts),
    maxAmbiguousCAJets=cms.uint32(options.maxAmbiguousCAJets),
    enforceLegacyRadiusConstraint=cms.bool(options.enforceLegacyRadiusConstraint),
    compressionLevel=cms.uint32(options.compressionLevel),
)

if options.analysisMode:
    from SuuAnalysis.ExistingOptimization.analysis2017_cfi import configure_analysis2017
    if options.sampleKind == "signal":
        category = "SuuToChiChi"
    elif sample_name.startswith("QCD"):
        category = "QCDMC"
    elif sample_name.startswith("TT"):
        category = "TTbarMC"
    elif sample_name.startswith("ST"):
        category = "STMC"
    elif sample_name.startswith("WJets"):
        category = "WJetsMC"
    else:
        raise ValueError("No validated b-tag efficiency map for background " + sample_name)
    analysis_config, analysis_sequence = configure_analysis2017(process, category, options.analysisSystematic)
    process.compactScan.analysis = analysis_config
    # These are producers, not filters: retain every event for MC normalization.
    process.path = cms.Path(analysis_sequence * process.compactScan)
else:
    process.path = cms.Path(process.compactScan)
