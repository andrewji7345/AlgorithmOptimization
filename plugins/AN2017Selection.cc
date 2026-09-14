#include "SuuAnalysis/ExistingOptimization/interface/AN2017Selection.h"
#include "SuuAnalysis/ExistingOptimization/interface/AN2017BTagWeight.h"

#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/FileInPath.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "CondFormats/JetMETObjects/interface/FactorizedJetCorrector.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectorParameters.h"
#include "CondFormats/JetMETObjects/interface/JetCorrectionUncertainty.h"
#include "JetMETCorrections/Modules/interface/JetResolution.h"
#include "correction.h"
#include "TFile.h"
#include "TH2.h"
#include "TLorentzVector.h"
#include "TRandom3.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {
std::unique_ptr<FactorizedJetCorrector> corrector(const edm::ParameterSet& cfg, const std::string& key) {
  std::vector<JetCorrectorParameters> levels;
  for (const auto& name : cfg.getParameter<std::vector<std::string>>(key))
    levels.emplace_back(edm::FileInPath(name).fullPath());
  if (levels.empty()) throw cms::Exception("Configuration") << "No JEC levels for " << key;
  return std::make_unique<FactorizedJetCorrector>(levels);
}

double jec(FactorizedJetCorrector& corr, const TLorentzVector& raw, double area, double rho) {
  if (!(raw.Pt() > 0.) || !std::isfinite(raw.E()))
    throw cms::Exception("InvalidJet") << "Cannot correct a nonfinite or zero-pT jet";
  corr.setJetPt(raw.Pt());
  corr.setJetEta(raw.Eta());
  corr.setJetE(raw.E());
  corr.setJetA(area);
  corr.setRho(rho);
  const double factor = corr.getCorrection();
  if (!(factor > 0.) || !std::isfinite(factor))
    throw cms::Exception("InvalidCorrection") << "Invalid JEC factor: " << factor;
  return factor;
}

double jecShift(JetCorrectionUncertainty& uncertainty, const TLorentzVector& p,
                const std::string& systematic) {
  if (systematic != "JECUp" && systematic != "JECDown") return 1.;
  uncertainty.setJetPt(p.Pt());
  uncertainty.setJetEta(p.Eta());
  const double delta = uncertainty.getUncertainty(systematic == "JECUp");
  const double factor = 1. + (systematic == "JECUp" ? delta : -delta);
  if (!std::isfinite(delta) || delta < 0. || !std::isfinite(factor) || factor <= 0.)
    throw cms::Exception("InvalidCorrection") << "Invalid total JEC uncertainty";
  return factor;
}

template <class P4> TLorentzVector tlv(const P4& p) {
  return TLorentzVector(p.px(), p.py(), p.pz(), p.energy());
}

std::unique_ptr<TH2> histogram(const edm::FileInPath& path, const std::string& name) {
  std::unique_ptr<TFile> file(TFile::Open(path.fullPath().c_str(), "READ"));
  if (!file || file->IsZombie()) throw cms::Exception("Configuration") << "Cannot open " << path.fullPath();
  const auto* source = dynamic_cast<TH2*>(file->Get(name.c_str()));
  if (!source) throw cms::Exception("Configuration") << "Missing histogram " << name << " in " << path.fullPath();
  std::unique_ptr<TH2> result(static_cast<TH2*>(source->Clone()));
  result->SetDirectory(nullptr);
  return result;
}

// Tight ID as explicitly documented in AN Table 13 (strict boundaries).
bool tightID(const pat::Jet& jet) {
  return jet.isPFJet() && jet.neutralHadronEnergyFraction() < 0.9 &&
         jet.neutralEmEnergyFraction() < 0.9 && jet.numberOfDaughters() > 1 &&
         jet.chargedHadronEnergyFraction() > 0. && jet.muonEnergyFraction() < 0.8 &&
         jet.chargedEmEnergyFraction() < 0.8;
}

double deepJetScore(const pat::Jet& jet) {
  double result = 0.;
  for (const auto& name : {"pfDeepFlavourJetTags:probb", "pfDeepFlavourJetTags:probbb",
                            "pfDeepFlavourJetTags:problepb"}) {
    const auto& values = jet.getPairDiscri();
    const auto found = std::find_if(values.begin(), values.end(), [&](const auto& entry) { return entry.first == name; });
    if (found == values.end() || !std::isfinite(found->second) || found->second < 0.)
      throw cms::Exception("MissingAnalysisInput") << "Missing/invalid DeepJet discriminator " << name;
    result += found->second;
  }
  return result;
}

bool triggerDecision(const edm::Event& event, const edm::TriggerResults& decisions,
                     const std::vector<std::string>& required, bool requireAll) {
  const auto& names = event.triggerNames(decisions);
  bool result = requireAll;
  for (const auto& target : required) {
    bool found = false, pass = false;
    for (unsigned i = 0; i < decisions.size(); ++i) {
      const auto& name = names.triggerName(i);
      if (name == target || (target.size() > 2 && target.substr(target.size() - 2) == "_v" &&
                             name.compare(0, target.size(), target) == 0)) {
        found = true;
        pass = pass || decisions.accept(i);
      }
    }
    if (!found) throw cms::Exception("MissingAnalysisInput") << "Required trigger/filter absent: " << target;
    result = requireAll ? result && pass : result || pass;
  }
  return result;
}
}  // namespace

class AN2017Selection::Impl {
public:
  Impl(const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc)
      : ak4_(cc.consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("ak4Jets"))),
        ak8_(cc.consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("ak8Jets"))),
        genAK4_(cc.consumes<std::vector<reco::GenJet>>(cfg.getParameter<edm::InputTag>("genAK4Jets"))),
        genAK8_(cc.consumes<std::vector<reco::GenJet>>(cfg.getParameter<edm::InputTag>("genAK8Jets"))),
        muons_(cc.consumes<std::vector<pat::Muon>>(cfg.getParameter<edm::InputTag>("muons"))),
        electrons_(cc.consumes<std::vector<pat::Electron>>(cfg.getParameter<edm::InputTag>("electrons"))),
        taus_(cc.consumes<std::vector<pat::Tau>>(cfg.getParameter<edm::InputTag>("taus"))),
        hlt_(cc.consumes<edm::TriggerResults>(cfg.getParameter<edm::InputTag>("triggerResults"))),
        filters_(cc.consumes<edm::TriggerResults>(cfg.getParameter<edm::InputTag>("filterResults"))),
        rhoToken_(cc.consumes<double>(cfg.getParameter<edm::InputTag>("rho"))),
        jerRhoToken_(cc.consumes<double>(cfg.getParameter<edm::InputTag>("jerRho"))),
        pileup_(cc.consumes<std::vector<PileupSummaryInfo>>(cfg.getParameter<edm::InputTag>("pileup"))),
        prefire_(cc.consumes<float>(cfg.getParameter<edm::InputTag>("prefiringWeight"))),
        prefireUp_(cc.consumes<float>(cfg.getParameter<edm::InputTag>("prefiringWeightUp"))),
        prefireDown_(cc.consumes<float>(cfg.getParameter<edm::InputTag>("prefiringWeightDown"))),
        genParticles_(cc.consumes<std::vector<reco::GenParticle>>(cfg.getParameter<edm::InputTag>("genParticles"))),
        applyTopPt_(cfg.getParameter<bool>("applyTopPtWeight")),
        systematic_(cfg.getParameter<std::string>("systematic")),
        ak4JEC_(corrector(cfg, "ak4JECFiles")), ak8JEC_(corrector(cfg, "ak8JECFiles")),
        subjetJEC_(corrector(cfg, "subjetJECFiles")),
        recoAK4JEC_(corrector(cfg, "recoAK4JECFiles")),
        ak4Uncertainty_(cfg.getParameter<edm::FileInPath>("jecAK4Uncertainty").fullPath()),
        ak8Uncertainty_(cfg.getParameter<edm::FileInPath>("jecAK8Uncertainty").fullPath()),
        recoAK4Uncertainty_(cfg.getParameter<edm::FileInPath>("jecRecoAK4Uncertainty").fullPath()),
        ak4Res_(cfg.getParameter<edm::FileInPath>("jerAK4Resolution").fullPath()),
        ak4SF_(cfg.getParameter<edm::FileInPath>("jerAK4ScaleFactor").fullPath()),
        ak8Res_(cfg.getParameter<edm::FileInPath>("jerAK8Resolution").fullPath()),
        ak8SF_(cfg.getParameter<edm::FileInPath>("jerAK8ScaleFactor").fullPath()),
        recoAK4Res_(cfg.getParameter<edm::FileInPath>("jerRecoAK4Resolution").fullPath()),
        recoAK4SF_(cfg.getParameter<edm::FileInPath>("jerRecoAK4ScaleFactor").fullPath()),
        btag_(correction::CorrectionSet::from_file(cfg.getParameter<edm::FileInPath>("btagFile").fullPath())),
        pileupCorrections_(correction::CorrectionSet::from_file(cfg.getParameter<edm::FileInPath>("pileupFile").fullPath())),
        veto_(histogram(cfg.getParameter<edm::FileInPath>("jetVetoFile"), cfg.getParameter<std::string>("jetVetoHistogram"))) {
    const auto path = cfg.getParameter<edm::FileInPath>("btagEfficiencyFile");
    efficiencies_[0] = histogram(path, "h_effLightJets_med");
    efficiencies_[1] = histogram(path, "h_effcJets_med");
    efficiencies_[2] = histogram(path, "h_effbJets_med");
    if (systematic_ != "nominal" && systematic_ != "JECUp" && systematic_ != "JECDown" &&
        systematic_ != "JERUp" && systematic_ != "JERDown")
      throw cms::Exception("Configuration") << "Unknown AN2017 kinematic systematic: " << systematic_;
  }

  double smear(const TLorentzVector& p, const JME::JetResolution& resolution,
               const JME::JetResolutionScaleFactor& scaleFactor, const TLorentzVector* gen,
               unsigned salt) const {
    JME::JetParameters params;
    params.setJetPt(p.Pt()).setJetEta(p.Eta()).setRho(jerRho_);
    const double sigma = resolution.getResolution(params);
    const auto variation = systematic_ == "JERUp" ? Variation::UP :
                           systematic_ == "JERDown" ? Variation::DOWN : Variation::NOMINAL;
    const double sf = scaleFactor.getScaleFactor(params, variation);
    if (!std::isfinite(sigma) || sigma < 0. || !std::isfinite(sf) || sf <= 0.)
      throw cms::Exception("InvalidCorrection") << "Invalid JER payload result";
    if (gen) return std::max(0., 1. + (sf - 1.) * (p.Pt() - gen->Pt()) / p.Pt());
    // Same event/jet receives the same draw in every trial and every process.
    // Incorporating event identity avoids the reference code's phi-only collisions.
    uint64_t hash = eventSeed_ ^ (uint64_t(std::llround((p.Phi() + 4.) * 1.e6)) << 1) ^ salt;
    hash ^= hash >> 33; hash *= 0xff51afd7ed558ccdULL; hash ^= hash >> 33;
    const unsigned seed = static_cast<unsigned>(hash ^ (hash >> 32));
    TRandom3 rng(seed ? seed : 1U);
    return std::max(0., 1. + rng.Gaus(0., sigma) * std::sqrt(std::max(sf * sf - 1., 0.)));
  }

  TLorentzVector corrected(const pat::Jet& jet, bool ak8, unsigned salt) {
    const auto raw = tlv(jet.correctedP4("Uncorrected"));
    auto p = jec(ak8 ? *ak8JEC_ : *ak4JEC_, raw, jet.jetArea(), rho_) * raw;
    p *= jecShift(ak8 ? ak8Uncertainty_ : ak4Uncertainty_, p, systematic_);
    TLorentzVector gen;
    if (jet.genJet()) gen = tlv(jet.genJet()->p4());
    return smear(p, ak8 ? ak8Res_ : ak4Res_, ak8 ? ak8SF_ : ak4SF_,
                 jet.genJet() ? &gen : nullptr, salt) * p;
  }

  bool vetoPass(const TLorentzVector& p) const {
    return veto_->GetBinContent(veto_->GetXaxis()->FindFixBin(p.Eta()),
                                veto_->GetYaxis()->FindFixBin(p.Phi())) <= 0.;
  }

  void addBtagWeight(an2017::BTagWeight& weight, std::array<an2017::BTagWeight, 8>& varied,
                    const pat::Jet& jet, const TLorentzVector& p) const {
    const int flavor = std::abs(jet.hadronFlavour());
    if (flavor != 0 && flavor != 4 && flavor != 5)
      throw cms::Exception("InvalidJet") << "Unsupported hadron flavor " << flavor;
    const auto& map = *efficiencies_[flavor == 5 ? 2 : flavor == 4 ? 1 : 0];
    const int ybin = std::clamp(map.GetYaxis()->FindFixBin(p.Eta()), 1, map.GetNbinsY());
    int last = map.GetNbinsX();
    while (last > 0 && map.GetBinContent(last, ybin) <= 0.) --last;
    if (!last) throw cms::Exception("InvalidCorrection") << "Empty btag efficiency eta row";
    const int xbin = std::clamp(map.GetXaxis()->FindFixBin(p.Pt()), 1, last);
    const double eff = map.GetBinContent(xbin, ybin);
    const double sf = btag_->at(flavor ? "deepJet_mujets" : "deepJet_incl")
                          ->evaluate({"central", "M", flavor, std::abs(p.Eta()), p.Pt()});
    const double discriminator = deepJetScore(jet);
    const bool tagged = discriminator > 0.3040;
    if (!std::isfinite(eff) || eff < 0. || eff > 1. || !std::isfinite(sf) || sf <= 0.)
      throw cms::Exception("InvalidCorrection") << "Invalid btag efficiency/SF: " << eff << "/" << sf;
    weight.add(tagged, eff, sf);
    const std::array<std::string, 4> labels{{"up_correlated", "down_correlated",
                                           "up_uncorrelated", "down_uncorrelated"}};
    for (unsigned i = 0; i < varied.size(); ++i) {
      const bool varyThisFlavor = (i < 4) == (flavor != 0);
      const double shifted = varyThisFlavor ? btag_->at(flavor ? "deepJet_mujets" : "deepJet_incl")
          ->evaluate({labels[i % 4], "M", flavor, std::abs(p.Eta()), p.Pt()}) : sf;
      // Official down variations can extrapolate below zero at high pT.
      // Match rootProcessor.C: inspect the final event probability ratio, not
      // individual per-jet factors (which may cancel across untagged jets).
      varied[i].add(tagged, eff, shifted);
    }
  }

  double softDropMass(const pat::Jet& jet) {
    if (!jet.hasSubjets("SoftDropPuppi"))
      throw cms::Exception("MissingAnalysisInput") << "Missing SoftDropPuppi subjet collection";
    TLorentzVector sum;
    unsigned index = 0;
    for (const auto& subjet : jet.subjets("SoftDropPuppi")) {
      const auto raw = tlv(subjet->correctedP4("Uncorrected"));
      auto p = jec(*subjetJEC_, raw, subjet->jetArea(), jerRho_) * raw;
      p *= jecShift(recoAK4Uncertainty_, p, systematic_);
      sum += smear(p, ak4Res_, ak4SF_, nullptr, 300 + index++) * p;
    }
    return std::max(0., sum.M());
  }

  AN2017Selection::EventInfo evaluate(const edm::Event& event) {
    AN2017Selection::EventInfo result;
    rho_ = event.get(rhoToken_);
    jerRho_ = event.get(jerRhoToken_);
    eventSeed_ = (uint64_t(event.id().run()) << 32) ^ event.id().event() ^
                 (uint64_t(event.luminosityBlock()) << 16);
    genAK4Jets_.clear();
    for (const auto& jet : event.get(genAK4_)) genAK4Jets_.push_back(tlv(jet.p4()));
    genJets_.clear();
    for (const auto& jet : event.get(genAK8_)) genJets_.push_back(tlv(jet.p4()));
    evaluated_ = true;
    result.trigger = triggerDecision(event, event.get(hlt_), {"HLT_PFHT1050_v", "HLT_PFJet500_v"}, false);
    result.filters = triggerDecision(event, event.get(filters_),
       {"Flag_goodVertices", "Flag_globalSuperTightHalo2016Filter", "Flag_HBHENoiseFilter",
        "Flag_HBHENoiseIsoFilter", "Flag_EcalDeadCellTriggerPrimitiveFilter", "Flag_BadPFMuonFilter",
        "Flag_BadPFMuonDzFilter", "Flag_eeBadScFilter", "Flag_ecalBadCalibFilter"}, true);
    result.leptonVeto = true;
    for (const auto& muon : event.get(muons_))
      if (muon.pt() > 8. && std::abs(muon.eta()) < 2.4 && muon.passed(reco::Muon::CutBasedIdLoose) &&
          muon.passed(reco::Muon::PFIsoMedium)) result.leptonVeto = false;
    for (const auto& electron : event.get(electrons_)) {
      if (!electron.isElectronIDAvailable("mvaEleID-Fall17-iso-V2-wpLoose"))
        throw cms::Exception("MissingAnalysisInput") << "Missing UL17 electron ID";
      if (electron.pt() > 12. && std::abs(electron.eta()) < 2.5 &&
          electron.electronID("mvaEleID-Fall17-iso-V2-wpLoose")) result.leptonVeto = false;
    }
    for (const auto& tau : event.get(taus_)) {
      const std::array<std::string, 4> ids{{"decayModeFindingNewDMs", "byVVLooseDeepTau2017v2p1VSe",
                                           "byVLooseDeepTau2017v2p1VSjet", "byVLooseDeepTau2017v2p1VSmu"}};
      bool pass = true;
      for (const auto& id : ids) {
        if (!tau.isTauIDAvailable(id)) throw cms::Exception("MissingAnalysisInput") << "Missing UL17 tau ID " << id;
        pass = pass && tau.tauID(id) > 0.5;
      }
      if (tau.pt() > 20. && std::abs(tau.eta()) < 2.3 && tau.decayMode() != 5 && tau.decayMode() != 6 &&
          tau.decayMode() != 7 && pass) result.leptonVeto = false;
    }

    bool foundPileup = false;
    std::array<double, 3> pileupWeights{{1., 1., 1.}};
    for (const auto& pu : event.get(pileup_)) {
      if (pu.getBunchCrossing() != 0) continue;
      const std::array<std::string, 3> labels{{"nominal", "up", "down"}};
      for (unsigned i=0; i<labels.size(); ++i)
        pileupWeights[i] = pileupCorrections_->at("Collisions17_UltraLegacy_goldenJSON")
                            ->evaluate({double(pu.getTrueNumInteractions()), labels[i]});
      result.weight *= pileupWeights[0];
      foundPileup = true;
      break;
    }
    if (!foundPileup) throw cms::Exception("MissingAnalysisInput") << "No in-time pileup summary";
    result.weight *= event.get(prefire_);
    double topPtWeight = 1.;
    if (applyTopPt_) {
      double product = 1.; unsigned tops = 0;
      for (const auto& particle : event.get(genParticles_))
        if (std::abs(particle.pdgId()) == 6 && particle.statusFlags().isLastCopy()) {
          product *= std::exp(0.0615 - 0.0005 * std::min(particle.pt(), 500.));
          ++tops;
        }
      if (tops != 2) throw cms::Exception("MissingAnalysisInput") << "TTbar weighting expects two last-copy tops, got " << tops;
      topPtWeight = std::sqrt(product);
      result.weight *= topPtWeight;
    }

    result.jetVeto = true;
    an2017::BTagWeight btag;
    std::array<an2017::BTagWeight, 8> variedBtag;
    std::vector<TLorentzVector> selectedAK4;
    unsigned index = 0;
    for (const auto& jet : event.get(ak4_)) {
      const auto p = corrected(jet, false, 100 + index++);
      // HT in the reference is deliberately accumulated before tight-ID selection.
      if (p.Pt() > 50. && std::abs(p.Eta()) < 2.5) result.ht += p.Pt();
      if (p.Pt() <= 50. || std::abs(p.Eta()) >= 2.5 || !tightID(jet)) continue;
      ++result.nAK4;
      selectedAK4.push_back(p);
      result.jetVeto = result.jetVeto && vetoPass(p);
      addBtagWeight(btag, variedBtag, jet, p);
      const double disc = deepJetScore(jet);
      result.btagJetPt.push_back(p.Pt());
      result.btagJetEta.push_back(p.Eta());
      result.btagJetDiscriminator.push_back(disc);
      if (p.Pt() > 70. && disc > 0.3040) ++result.nBTags;
    }
    result.btagWeight = btag.value();
    result.btagWeightFallback = btag.fallback();
    result.weight *= result.btagWeight;
    // Absolute alternate total weights avoid ratios at zero nominal weight.
    const double prefire = event.get(prefire_);
    const double b = result.btagWeight;
    result.weightVariations = {
      pileupWeights[1] * prefire * topPtWeight * b,
      pileupWeights[2] * prefire * topPtWeight * b,
      pileupWeights[0] * event.get(prefireUp_) * topPtWeight * b,
      pileupWeights[0] * event.get(prefireDown_) * topPtWeight * b};
    for (const auto& varied : variedBtag)
      result.weightVariations.push_back(pileupWeights[0] * prefire * topPtWeight * varied.value());
    for (const auto& varied : variedBtag)
      result.btagWeightVariationFallback.push_back(varied.fallback());
    // Backward-compatible v3 top-pT convention; the reference convention below
    // follows rootProcessor.C, whose nominal prediction has no top-pT reweighting.
    result.weightVariations.push_back(pileupWeights[0] * prefire * topPtWeight * topPtWeight * b);
    result.weightVariations.push_back(pileupWeights[0] * prefire * b);
    result.referenceWeight = pileupWeights[0] * prefire * b;
    result.referenceWeightVariations = {
      pileupWeights[1] * prefire * b, pileupWeights[2] * prefire * b,
      pileupWeights[0] * event.get(prefireUp_) * b,
      pileupWeights[0] * event.get(prefireDown_) * b};
    for (const auto& varied : variedBtag)
      result.referenceWeightVariations.push_back(pileupWeights[0] * prefire * varied.value());
    result.referenceWeightVariations.push_back(result.referenceWeight * topPtWeight);
    result.referenceWeightVariations.push_back(result.referenceWeight);
    for (const auto value : result.weightVariations)
      if (!std::isfinite(value) || value < 0.)
        throw cms::Exception("InvalidCorrection") << "Nonfinite or negative alternate event weight";
    // The reference uses collection pT order for the leading quartet and AK8 eta rule.
    index = 0;
    for (const auto& jet : event.get(ak8_)) {
      const auto p = corrected(jet, true, 200);
      const double etaMax = result.nAK8 < 2 ? 2.5 : 1.4;
      if (std::abs(p.Eta()) >= etaMax || !tightID(jet)) continue;
      if (p.Pt() > 500. && softDropMass(jet) > 45.) ++result.nHeavyAK8;
      if (std::hypot(p.Pt(), p.M()) <= 300. || p.M() < 0.) continue;
      ++result.nAK8;
      result.jetVeto = result.jetVeto && vetoPass(p);
      const double rawE = jet.correctedP4("Uncorrected").energy();
      if (!(rawE > 0.)) throw cms::Exception("InvalidJet") << "Invalid PAT AK8 raw energy";
      const double scale = p.E() / rawE;
      for (unsigned j=0; j<jet.numberOfDaughters(); ++j) {
        const auto* candidate = dynamic_cast<const pat::PackedCandidate*>(jet.daughter(j));
        if (!candidate) throw cms::Exception("MissingAnalysisInput") << "PAT AK8 daughter is not packed";
        const double weight = candidate->puppiWeight();
        if (!std::isfinite(weight) || weight < 0.)
          throw cms::Exception("InvalidJet") << "Invalid reference daughter PUPPI weight";
        // The source keeps zero-weight daughters too. Preserve that choice
        // because constituent multiplicity participates in reference cuts.
        result.referenceConstituents.push_back(scale * weight * tlv(candidate->p4()));
      }
    }
    bool dijet = false;
    if (selectedAK4.size() >= 4) {
      const std::array<std::array<unsigned, 4>, 3> pairs{{{{0, 1, 2, 3}}, {{0, 2, 1, 3}}, {{0, 3, 1, 2}}}};
      double best = std::numeric_limits<double>::infinity();
      for (const auto& pairing : pairs) {
        const auto& a = selectedAK4[pairing[0]]; const auto& b = selectedAK4[pairing[1]];
        const auto& c = selectedAK4[pairing[2]]; const auto& d = selectedAK4[pairing[3]];
        const double distance = std::hypot(a.DeltaR(b), c.DeltaR(d));
        if (distance < best) { best = distance; dijet = (a + b).M() > 1000. && (c + d).M() > 1000.; }
      }
    }
    result.baseline = result.trigger && result.filters && result.leptonVeto && result.jetVeto &&
                      result.ht > 1600. && result.nAK4 >= 4 && result.nAK8 >= 3 &&
                      (result.nHeavyAK8 >= 2 || dijet);
    if (!std::isfinite(result.weight) || result.weight < 0.)
      throw cms::Exception("InvalidCorrection") << "Nonfinite or negative nominal event weight";
    return result;
  }

  double correctReclusteredJet(const TLorentzVector& raw, double radius) {
    const bool isAK4 = std::abs(radius - 0.4) < 1.e-6;
    if (!isAK4 && std::abs(radius - 0.8) >= 1.e-6)
      throw cms::Exception("Configuration") << "Reclustered corrections require AK R=0.4 or 0.8";
    auto& correction = isAK4 ? *recoAK4JEC_ : *ak8JEC_;
    const auto& resolution = isAK4 ? recoAK4Res_ : ak8Res_;
    const auto& scaleFactor = isAK4 ? recoAK4SF_ : ak8SF_;
    const auto& genJets = isAK4 ? genAK4Jets_ : genJets_;
    if (!evaluated_) throw cms::Exception("LogicError") << "Call evaluate(event) before correcting a reclustered jet";
    // PUPPI JEC omits L1 pileup correction, so this fixed area has no effect on payload evaluation.
    double factor = jec(correction, raw, M_PI * radius * radius, rho_);
    factor *= jecShift(isAK4 ? recoAK4Uncertainty_ : ak8Uncertainty_, factor * raw, systematic_);
    const auto p = factor * raw;
    JME::JetParameters params;
    params.setJetPt(p.Pt()).setJetEta(p.Eta()).setRho(jerRho_);
    const double sigma = resolution.getResolution(params);
    const TLorentzVector* matched = nullptr;
    double bestDR = radius / 2.;
    for (const auto& gen : genJets) {
      const double dr = p.DeltaR(gen);
      if (dr < bestDR && std::abs(p.Pt() - gen.Pt()) < 3. * sigma * p.Pt()) {
        matched = &gen; bestDR = dr;
      }
    }
    return factor * smear(p, resolution, scaleFactor, matched, isAK4 ? 400 : 200);
  }

private:
  edm::EDGetTokenT<std::vector<pat::Jet>> ak4_, ak8_;
  edm::EDGetTokenT<std::vector<reco::GenJet>> genAK4_, genAK8_;
  edm::EDGetTokenT<std::vector<pat::Muon>> muons_;
  edm::EDGetTokenT<std::vector<pat::Electron>> electrons_;
  edm::EDGetTokenT<std::vector<pat::Tau>> taus_;
  edm::EDGetTokenT<edm::TriggerResults> hlt_, filters_;
  edm::EDGetTokenT<double> rhoToken_, jerRhoToken_;
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> pileup_;
  edm::EDGetTokenT<float> prefire_, prefireUp_, prefireDown_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genParticles_;
  bool applyTopPt_;
  std::string systematic_;
  std::unique_ptr<FactorizedJetCorrector> ak4JEC_, ak8JEC_, subjetJEC_, recoAK4JEC_;
  JetCorrectionUncertainty ak4Uncertainty_, ak8Uncertainty_, recoAK4Uncertainty_;
  JME::JetResolution ak4Res_;
  JME::JetResolutionScaleFactor ak4SF_;
  JME::JetResolution ak8Res_;
  JME::JetResolutionScaleFactor ak8SF_;
  JME::JetResolution recoAK4Res_;
  JME::JetResolutionScaleFactor recoAK4SF_;
  std::unique_ptr<correction::CorrectionSet> btag_, pileupCorrections_;
  std::unique_ptr<TH2> veto_;
  std::array<std::unique_ptr<TH2>, 3> efficiencies_;
  double rho_ = 0., jerRho_ = 0.;
  uint64_t eventSeed_ = 0;
  bool evaluated_ = false;
  std::vector<TLorentzVector> genJets_, genAK4Jets_;
};

AN2017Selection::AN2017Selection(const edm::ParameterSet& cfg, edm::ConsumesCollector&& cc)
    : impl_(std::make_unique<Impl>(cfg, std::move(cc))) {}
AN2017Selection::~AN2017Selection() = default;
AN2017Selection::EventInfo AN2017Selection::evaluate(const edm::Event& event) { return impl_->evaluate(event); }
double AN2017Selection::correctReclusteredJet(const TLorentzVector& raw, double radius) { return impl_->correctReclusteredJet(raw, radius); }
bool AN2017Selection::passesReclusteredJetVeto(const TLorentzVector& corrected) const {
  return impl_->vetoPass(corrected);
}

// Official L1 prefiring producer needs jets with nominal JEC, before JER smearing.
class AN2017CorrectedJets : public edm::stream::EDProducer<> {
public:
  explicit AN2017CorrectedJets(const edm::ParameterSet& cfg)
      : jets_(consumes<std::vector<pat::Jet>>(cfg.getParameter<edm::InputTag>("src"))),
        rho_(consumes<double>(cfg.getParameter<edm::InputTag>("rho"))), corr_(corrector(cfg, "jecFiles")),
        uncertainty_(cfg.getParameter<edm::FileInPath>("jecUncertainty").fullPath()),
        systematic_(cfg.getParameter<std::string>("systematic")) {
    produces<std::vector<pat::Jet>>();
  }
  void produce(edm::Event& event, const edm::EventSetup&) override {
    auto output = std::make_unique<std::vector<pat::Jet>>();
    for (const auto& jet : event.get(jets_)) {
      auto copy = jet;
      const auto raw = jet.correctedP4("Uncorrected");
      double factor = jec(*corr_, tlv(raw), jet.jetArea(), event.get(rho_));
      factor *= jecShift(uncertainty_, factor * tlv(raw), systematic_);
      copy.setP4(factor * raw);
      output->push_back(std::move(copy));
    }
    event.put(std::move(output));
  }
private:
  edm::EDGetTokenT<std::vector<pat::Jet>> jets_;
  edm::EDGetTokenT<double> rho_;
  std::unique_ptr<FactorizedJetCorrector> corr_;
  JetCorrectionUncertainty uncertainty_;
  std::string systematic_;
};
DEFINE_FWK_MODULE(AN2017CorrectedJets);
