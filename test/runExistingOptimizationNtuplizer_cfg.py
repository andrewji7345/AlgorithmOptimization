import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

options = VarParsing('analysis')

options.register(
    'inputRootFiles',
    'SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_4000_1000.txt', # to test ntuplizer
    #'SuuAnalysis/ExistingOptimization/test/signalMCFiles/WbWb_all.txt', # for realsies
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Text file containing list of ROOT files'
)

options.register(
    'outputRootFile',
    'rootfiles_existingOptimization/WbWb_4000_1000.root', # to test ntuplizer
    #'rootfiles_existingOptimization/WbWb_all.root', # for realsies
    VarParsing.multiplicity.singleton,
    VarParsing.varType.string,
    'Output ROOT file'
)

options.register(
    'jetPtCut',
    300.0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'Minimum pT for jets passed to the reconstruction (legacy default: 300 GeV)'
)

options.register(
    'eventJetPtCut',
    300.0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'pT threshold used by the event-level jet multiplicity requirement'
)

options.register(
    'minEventJets',
    0,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.int,
    'Minimum number of jets above eventJetPtCut (0 disables the event gate)'
)

options.register(
    'akRadius',
    0.8,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'AK jet clustering radius'
)

options.register(
    'caRadius',
    0.8,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'CA jet clustering radius'
)

options.register(
    'cosThrust',
    0.85,
    VarParsing.multiplicity.singleton,
    VarParsing.varType.float,
    'Cosine similarity to assign CA jet to SJ'
)

options.parseArguments()

def readFileList(fname):
    with open(fname) as f:
        return [line.strip() for line in f
                if line.strip() and not line.startswith("#")]

process = cms.Process("NTUPLE")

process.load("FWCore.MessageService.MessageLogger_cfi")

process.MessageLogger.cerr.FwkReport.reportEvery = 1000

process.maxEvents = cms.untracked.PSet(
    #input = cms.untracked.int32(-1) # for realsies
    input = cms.untracked.int32(1000) # for ntuplizer evaluation
)

process.source = cms.Source(
    "PoolSource",
    fileNames = cms.untracked.vstring(
        *readFileList(options.inputRootFiles)
    )
)

process.TFileService = cms.Service(
    "TFileService",
    fileName = cms.string(options.outputRootFile)
)

process.load("SuuAnalysis.ExistingOptimization.ExistingOptimizationNtuplizer_cfi")

process.existingOptimizationNtuplizer.jetPtCut = cms.double(options.jetPtCut)

process.existingOptimizationNtuplizer.eventJetPtCut = cms.double(options.eventJetPtCut)

process.existingOptimizationNtuplizer.minEventJets = cms.uint32(options.minEventJets)

process.existingOptimizationNtuplizer.akRadius = cms.double(options.akRadius)

process.existingOptimizationNtuplizer.caRadius = cms.double(options.caRadius)

process.existingOptimizationNtuplizer.cosThrust = cms.double(options.cosThrust)

process.p = cms.Path(
    process.existingOptimizationNtuplizer
)
