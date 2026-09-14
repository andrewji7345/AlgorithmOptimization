"""Nominal AN-23-067 UL17 MC selection, using vendored calibration payloads."""

import FWCore.ParameterSet.Config as cms

_DATA = "SuuAnalysis/ExistingOptimization/data/analysis_2017/"
_CATEGORIES = ("SuuToChiChi", "QCDMC", "TTbarMC", "STMC", "WJetsMC")


def _jec(jet, levels):
    return cms.vstring(*(_DATA + "Summer19UL17_V5_MC_" + level + "_" + jet + ".txt" for level in levels))


def configure_analysis2017(process, sample_category="SuuToChiChi", systematic="nominal"):
    """Return (selection PSet, prerequisite sequence) for the compact ntuplizer.

    The prefiring producer consumes jets corrected with nominal JEC before JER,
    matching the original analysis. All correction resources resolve at startup.
    """
    if sample_category not in _CATEGORIES:
        raise ValueError("Unknown 2017 sample category: " + sample_category)
    if systematic not in ("nominal", "JECUp", "JECDown", "JERUp", "JERDown"):
        raise ValueError("Unknown 2017 kinematic systematic: " + systematic)
    ak4_jec = _jec("AK4PFchs", ("L1FastJet", "L2Relative", "L3Absolute"))
    process.an2017CorrectedAK4 = cms.EDProducer(
        "AN2017CorrectedJets",
        src=cms.InputTag("slimmedJets"),
        rho=cms.InputTag("fixedGridRhoFastjetAll"),
        jecFiles=ak4_jec,
        jecUncertainty=cms.FileInPath(_DATA + "Summer19UL17_V5_MC_Uncertainty_AK4PFchs.txt"),
        systematic=cms.string(systematic),
    )
    from PhysicsTools.PatUtils.l1PrefiringWeightProducer_cfi import l1PrefiringWeightProducer

    process.an2017Prefiring = l1PrefiringWeightProducer.clone(
        TheJets=cms.InputTag("an2017CorrectedAK4"),
        DataEraECAL=cms.string("UL2017BtoF"),
        DataEraMuon=cms.string("20172018"),
        UseJetEMPt=cms.bool(False),
        PrefiringRateSystematicUnctyECAL=cms.double(0.2),
        PrefiringRateSystematicUnctyMuon=cms.double(0.2),
    )
    selection = cms.PSet(
        systematic=cms.string(systematic),
        ak4Jets=cms.InputTag("slimmedJets"),
        ak8Jets=cms.InputTag("slimmedJetsAK8"),
        genAK4Jets=cms.InputTag("slimmedGenJets"),
        genAK8Jets=cms.InputTag("slimmedGenJetsAK8"),
        muons=cms.InputTag("slimmedMuons"),
        electrons=cms.InputTag("slimmedElectrons"),
        taus=cms.InputTag("slimmedTaus"),
        triggerResults=cms.InputTag("TriggerResults", "", "HLT"),
        filterResults=cms.InputTag("TriggerResults", "", "PAT"),
        rho=cms.InputTag("fixedGridRhoFastjetAll"),
        jerRho=cms.InputTag("fixedGridRhoAll"),
        pileup=cms.InputTag("slimmedAddPileupInfo"),
        prefiringWeight=cms.InputTag("an2017Prefiring", "nonPrefiringProb"),
        prefiringWeightUp=cms.InputTag("an2017Prefiring", "nonPrefiringProbUp"),
        prefiringWeightDown=cms.InputTag("an2017Prefiring", "nonPrefiringProbDown"),
        genParticles=cms.InputTag("prunedGenParticles"),
        applyTopPtWeight=cms.bool(sample_category == "TTbarMC"),
        ak4JECFiles=ak4_jec,
        recoAK4JECFiles=_jec("AK4PFPuppi", ("L2Relative", "L3Absolute")),
        ak8JECFiles=_jec("AK8PFPuppi", ("L2Relative", "L3Absolute")),
        subjetJECFiles=_jec("AK4PFPuppi", ("L1FastJet", "L2Relative", "L3Absolute")),
        jecAK4Uncertainty=cms.FileInPath(_DATA + "Summer19UL17_V5_MC_Uncertainty_AK4PFchs.txt"),
        jecAK8Uncertainty=cms.FileInPath(_DATA + "Summer19UL17_V5_MC_Uncertainty_AK8PFPuppi.txt"),
        jecRecoAK4Uncertainty=cms.FileInPath(_DATA + "Summer19UL17_V5_MC_Uncertainty_AK4PFPuppi.txt"),
        jerAK4Resolution=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_PtResolution_AK4PFchs.txt"),
        jerAK4ScaleFactor=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_SF_AK4PFchs.txt"),
        jerAK8Resolution=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_PtResolution_AK8PF.txt"),
        jerAK8ScaleFactor=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_SF_AK8PF.txt"),
        jerRecoAK4Resolution=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_PtResolution_AK4PFPuppi.txt"),
        jerRecoAK4ScaleFactor=cms.FileInPath(_DATA + "Summer19UL17_JRV3_MC_SF_AK4PFPuppi.txt"),
        btagFile=cms.FileInPath(_DATA + "btagging.json"),
        btagEfficiencyFile=cms.FileInPath(_DATA + "btag_efficiency_map_" + sample_category + "_combined_2017.root"),
        pileupFile=cms.FileInPath(_DATA + "puWeights.json"),
        jetVetoFile=cms.FileInPath(_DATA + "hotjets-UL17_v2.root"),
        jetVetoHistogram=cms.string("h2hot_ul17_plus_hep17_plus_hbpw89"),
    )
    return selection, cms.Sequence(process.an2017CorrectedAK4 * process.an2017Prefiring)
