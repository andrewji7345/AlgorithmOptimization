#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "Compression.h"
#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TVector3.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"

#include "fastjet/ClusterSequence.hh"
#include "fastjet/Error.hh"
#include "fastjet/JetDefinition.hh"
#include "fastjet/PseudoJet.hh"

namespace {

enum class RecoStatus : std::uint8_t {
  valid = 0,
  noSelectedJets = 1,
  noSelectedConstituents = 2,
  invalidCOM = 3,
  invalidThrust = 4,
  noCAJets = 5,
  complexityGuard = 6,
  noValidPartition = 7,
  numericalFailure = 8
};

constexpr float kInvalidMass = std::numeric_limits<float>::quiet_NaN();

bool finiteP4(const TLorentzVector &p4) {
  return std::isfinite(p4.Px()) && std::isfinite(p4.Py()) &&
         std::isfinite(p4.Pz()) && std::isfinite(p4.E());
}

bool finiteVector(const TVector3 &vector) {
  return std::isfinite(vector.X()) && std::isfinite(vector.Y()) &&
         std::isfinite(vector.Z());
}

template <typename T> void sortAndUnique(std::vector<T> &values) {
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

struct WeightedParticle {
  TLorentzVector labP4;
};

struct AKJetCache {
  TLorentzVector labP4;
  std::vector<std::uint32_t> constituentIndices;
};

struct SelectedSystem {
  RecoStatus status = RecoStatus::valid;
  TVector3 beta;
  TVector3 thrustAxis;
  std::vector<TLorentzVector> comConstituents;
};

struct AssignmentResult {
  RecoStatus status = RecoStatus::numericalFailure;
  std::uint16_t nAmbiguous = 0;
  float leadingPtMass = kInvalidMass;
  float subleadingPtMass = kInvalidMass;
};

struct CAJetCache {
  RecoStatus status = RecoStatus::numericalFailure;
  TVector3 beta;
  std::vector<TLorentzVector> comJets;
  std::vector<double> cosToThrust;
  std::map<std::vector<std::int8_t>, AssignmentResult> assignmentCache;
};

struct BaseConfiguration {
  double collectionPtCut = 0.;
  double caRadius = 0.;
  std::size_t collectionPtIndex = 0;
  std::size_t caRadiusIndex = 0;
};

} // namespace

class CompactOptimizationScanNtuplizer
    : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit CompactOptimizationScanNtuplizer(const edm::ParameterSet &);
  ~CompactOptimizationScanNtuplizer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;

  std::vector<WeightedParticle>
  makeWeightedParticles(const pat::PackedCandidateCollection &) const;
  std::vector<AKJetCache>
  clusterAKJets(const std::vector<WeightedParticle> &) const;
  SelectedSystem
  buildSelectedSystem(std::size_t, const std::vector<AKJetCache> &,
                      const std::vector<WeightedParticle> &) const;
  CAJetCache clusterCAJets(const SelectedSystem &, double) const;
  AssignmentResult assignCAJets(const CAJetCache &,
                                const std::vector<std::int8_t> &) const;
  std::vector<std::int8_t> classificationForCut(const CAJetCache &,
                                                double) const;

  static bool findThrustAxis(const std::vector<TLorentzVector> &, TVector3 &);
  static bool physicalMass(const TLorentzVector &, double &);
  static AssignmentResult failure(RecoStatus, std::size_t = 0);

  edm::EDGetTokenT<pat::PackedCandidateCollection> packedPFToken_;
  edm::EDGetTokenT<GenEventInfoProduct> generatorInfoToken_;

  double akRadius_;
  std::vector<double> collectionPtCuts_;
  std::vector<double> caRadii_;
  std::vector<double> cosThrustCuts_;
  std::vector<unsigned int> configuredGateJetCounts_;
  std::vector<double> defaultGatePtCuts_;
  unsigned int maxAmbiguousCAJets_;
  bool enforceLegacyRadiusConstraint_;
  unsigned int compressionLevel_;

  std::vector<BaseConfiguration> baseConfigurations_;
  TTree *metadataTree_ = nullptr;
  TTree *eventsTree_ = nullptr;

  // Metadata (one row per output file).
  std::uint32_t schemaVersion_ = 1;
  std::string sampleName_;
  bool puppiWeighted_ = true;
  bool useJEC_ = false;
  std::string caAlgorithm_ = "cambridge_y_phi";
  std::string massObjective_ = "abs(m1-m2)/(m1+m2)";
  std::string gateRecoConstraint_ =
      "collectionPtCut<=gatePtCut; nGateJets=0 is ungated";
  std::vector<std::uint16_t> defaultGateJetCounts_;
  std::vector<float> metadataCollectionPtCuts_;
  std::vector<float> metadataCaRadii_;
  std::vector<float> metadataCosThrustCuts_;
  std::vector<float> metadataGatePtCuts_;
  std::vector<float> baseCollectionPtCut_;
  std::vector<float> baseCaRadius_;
  std::vector<std::uint32_t> configId_;
  std::vector<std::uint32_t> configBaseIndex_;
  std::vector<float> configCollectionPtCut_;
  std::vector<float> configCaRadius_;
  std::vector<float> configCosThrust_;
  std::vector<std::uint8_t> statusCodes_;
  std::vector<std::string> statusNames_;
  std::uint64_t processedEvents_ = 0;
  double sumWeights_ = 0.;
  double sumWeights2_ = 0.;

  // Event data. Gates are reconstructed offline from sorted AK pT, so the
  // reconstruction payload is never duplicated across gate configurations.
  std::uint32_t run_ = 0;
  std::uint32_t lumi_ = 0;
  std::uint64_t event_ = 0;
  float genWeight_ = 1.f;
  std::vector<float> akJetPt_;
  std::vector<std::uint16_t> nCAJets_;
  std::vector<std::uint8_t> recoStatus_;
  std::vector<std::uint16_t> nAmbiguous_;
  std::vector<float> sj1Mass_;
  std::vector<float> sj2Mass_;
};

CompactOptimizationScanNtuplizer::CompactOptimizationScanNtuplizer(
    const edm::ParameterSet &config)
    : packedPFToken_(consumes<pat::PackedCandidateCollection>(
          config.getParameter<edm::InputTag>("packedPFCandidates"))),
      generatorInfoToken_(consumes<GenEventInfoProduct>(
          config.getParameter<edm::InputTag>("generatorInfo"))),
      akRadius_(config.getParameter<double>("akRadius")),
      collectionPtCuts_(
          config.getParameter<std::vector<double>>("collectionPtCuts")),
      caRadii_(config.getParameter<std::vector<double>>("caRadii")),
      cosThrustCuts_(config.getParameter<std::vector<double>>("cosThrustCuts")),
      configuredGateJetCounts_(config.getParameter<std::vector<unsigned int>>(
          "defaultGateJetCounts")),
      defaultGatePtCuts_(
          config.getParameter<std::vector<double>>("defaultGatePtCuts")),
      maxAmbiguousCAJets_(
          config.getParameter<unsigned int>("maxAmbiguousCAJets")),
      enforceLegacyRadiusConstraint_(
          config.getParameter<bool>("enforceLegacyRadiusConstraint")),
      compressionLevel_(config.getParameter<unsigned int>("compressionLevel")),
      sampleName_(config.getParameter<std::string>("sampleName")) {
  usesResource(TFileService::kSharedResource);

  sortAndUnique(collectionPtCuts_);
  sortAndUnique(caRadii_);
  sortAndUnique(cosThrustCuts_);
  sortAndUnique(configuredGateJetCounts_);
  sortAndUnique(defaultGatePtCuts_);

  if (!std::isfinite(akRadius_) || akRadius_ <= 0.) {
    throw cms::Exception("Configuration")
        << "akRadius must be finite and positive";
  }
  if (collectionPtCuts_.empty() || caRadii_.empty() || cosThrustCuts_.empty()) {
    throw cms::Exception("Configuration")
        << "collectionPtCuts, caRadii, and cosThrustCuts must all be nonempty";
  }
  for (const double cut : collectionPtCuts_) {
    if (!std::isfinite(cut) || cut <= 0.) {
      throw cms::Exception("Configuration")
          << "Every collectionPtCuts entry must be finite and positive";
    }
  }
  for (const double radius : caRadii_) {
    if (!std::isfinite(radius) || radius <= 0.) {
      throw cms::Exception("Configuration")
          << "Every caRadii entry must be finite and positive";
    }
  }
  for (const double cut : cosThrustCuts_) {
    if (!std::isfinite(cut) || cut < 0. || cut > 1.) {
      throw cms::Exception("Configuration")
          << "Every cosThrustCuts entry must lie in [0, 1]";
    }
  }
  if (maxAmbiguousCAJets_ > 20) {
    throw cms::Exception("Configuration")
        << "maxAmbiguousCAJets must not exceed 20; larger exhaustive searches "
           "are unsafe";
  }
  if (compressionLevel_ > 22) {
    throw cms::Exception("Configuration")
        << "compressionLevel must lie in [0, 22] for ZSTD";
  }

  for (const unsigned int count : configuredGateJetCounts_) {
    if (count > std::numeric_limits<std::uint16_t>::max()) {
      throw cms::Exception("Configuration")
          << "defaultGateJetCounts entries must fit in uint16";
    }
    defaultGateJetCounts_.push_back(static_cast<std::uint16_t>(count));
  }
  if (defaultGateJetCounts_.empty() || defaultGateJetCounts_.front() != 0) {
    throw cms::Exception("Configuration")
        << "defaultGateJetCounts must be nonempty and include the canonical "
           "ungated value 0";
  }
  if (defaultGatePtCuts_.empty()) {
    throw cms::Exception("Configuration")
        << "defaultGatePtCuts must be nonempty";
  }
  for (const double cut : defaultGatePtCuts_) {
    if (!std::isfinite(cut) || cut < collectionPtCuts_.front()) {
      throw cms::Exception("Configuration")
          << "Every defaultGatePtCuts entry must be finite and at least the "
             "smallest collectionPtCut ("
          << collectionPtCuts_.front() << " GeV)";
    }
  }

  if (enforceLegacyRadiusConstraint_) {
    const double minimumCARadius = std::max(0.4, akRadius_ - 0.2);
    caRadii_.erase(std::remove_if(caRadii_.begin(), caRadii_.end(),
                                  [minimumCARadius](double radius) {
                                    return radius + 1.e-9 < minimumCARadius;
                                  }),
                   caRadii_.end());
    if (caRadii_.empty()) {
      throw cms::Exception("Configuration")
          << "The CA >= max(0.4, AK-0.2) radius constraint removed every "
             "requested CA radius";
    }
  }

  metadataCollectionPtCuts_.reserve(collectionPtCuts_.size());
  for (const double value : collectionPtCuts_) {
    metadataCollectionPtCuts_.push_back(static_cast<float>(value));
  }
  metadataCaRadii_.reserve(caRadii_.size());
  for (const double value : caRadii_) {
    metadataCaRadii_.push_back(static_cast<float>(value));
  }
  metadataCosThrustCuts_.reserve(cosThrustCuts_.size());
  for (const double value : cosThrustCuts_) {
    metadataCosThrustCuts_.push_back(static_cast<float>(value));
  }
  metadataGatePtCuts_.reserve(defaultGatePtCuts_.size());
  for (const double value : defaultGatePtCuts_) {
    metadataGatePtCuts_.push_back(static_cast<float>(value));
  }

  for (std::size_t iPt = 0; iPt < collectionPtCuts_.size(); ++iPt) {
    for (std::size_t iRadius = 0; iRadius < caRadii_.size(); ++iRadius) {
      BaseConfiguration base;
      base.collectionPtCut = collectionPtCuts_[iPt];
      base.caRadius = caRadii_[iRadius];
      base.collectionPtIndex = iPt;
      base.caRadiusIndex = iRadius;
      baseConfigurations_.push_back(base);
      baseCollectionPtCut_.push_back(static_cast<float>(base.collectionPtCut));
      baseCaRadius_.push_back(static_cast<float>(base.caRadius));
    }
  }

  const std::size_t nConfigurations =
      baseConfigurations_.size() * cosThrustCuts_.size();
  if (nConfigurations > std::numeric_limits<std::uint32_t>::max()) {
    throw cms::Exception("Configuration")
        << "The reconstruction grid exceeds uint32 indexing";
  }
  configId_.reserve(nConfigurations);
  configBaseIndex_.reserve(nConfigurations);
  configCollectionPtCut_.reserve(nConfigurations);
  configCaRadius_.reserve(nConfigurations);
  configCosThrust_.reserve(nConfigurations);
  for (std::size_t iBase = 0; iBase < baseConfigurations_.size(); ++iBase) {
    for (const double cosCut : cosThrustCuts_) {
      const auto id = static_cast<std::uint32_t>(configId_.size());
      configId_.push_back(id);
      configBaseIndex_.push_back(static_cast<std::uint32_t>(iBase));
      configCollectionPtCut_.push_back(
          static_cast<float>(baseConfigurations_[iBase].collectionPtCut));
      configCaRadius_.push_back(
          static_cast<float>(baseConfigurations_[iBase].caRadius));
      configCosThrust_.push_back(static_cast<float>(cosCut));
    }
  }

  statusCodes_ = {static_cast<std::uint8_t>(RecoStatus::valid),
                  static_cast<std::uint8_t>(RecoStatus::noSelectedJets),
                  static_cast<std::uint8_t>(RecoStatus::noSelectedConstituents),
                  static_cast<std::uint8_t>(RecoStatus::invalidCOM),
                  static_cast<std::uint8_t>(RecoStatus::invalidThrust),
                  static_cast<std::uint8_t>(RecoStatus::noCAJets),
                  static_cast<std::uint8_t>(RecoStatus::complexityGuard),
                  static_cast<std::uint8_t>(RecoStatus::noValidPartition),
                  static_cast<std::uint8_t>(RecoStatus::numericalFailure)};
  statusNames_ = {"valid",
                  "no_selected_jets",
                  "no_selected_constituents",
                  "invalid_com",
                  "invalid_thrust",
                  "no_ca_jets",
                  "complexity_guard",
                  "no_valid_partition",
                  "numerical_failure"};
}

void CompactOptimizationScanNtuplizer::beginJob() {
  edm::Service<TFileService> fs;
  fs->file().SetCompressionSettings(
      ROOT::CompressionSettings(ROOT::RCompressionSetting::EAlgorithm::kZSTD,
                                static_cast<int>(compressionLevel_)));

  metadataTree_ =
      fs->make<TTree>("Metadata", "Compact scan schema and configuration");
  metadataTree_->Branch("schemaVersion", &schemaVersion_);
  metadataTree_->Branch("sampleName", &sampleName_);
  metadataTree_->Branch("akRadius", &akRadius_);
  metadataTree_->Branch("maxAmbiguousCAJets", &maxAmbiguousCAJets_);
  metadataTree_->Branch("puppiWeighted", &puppiWeighted_);
  metadataTree_->Branch("useJEC", &useJEC_);
  metadataTree_->Branch("caAlgorithm", &caAlgorithm_);
  metadataTree_->Branch("massObjective", &massObjective_);
  metadataTree_->Branch("gateRecoConstraint", &gateRecoConstraint_);
  metadataTree_->Branch("enforceLegacyRadiusConstraint",
                        &enforceLegacyRadiusConstraint_);
  metadataTree_->Branch("collectionPtCuts", &metadataCollectionPtCuts_);
  metadataTree_->Branch("caRadii", &metadataCaRadii_);
  metadataTree_->Branch("cosThrustCuts", &metadataCosThrustCuts_);
  metadataTree_->Branch("defaultGateJetCounts", &defaultGateJetCounts_);
  metadataTree_->Branch("defaultGatePtCuts", &metadataGatePtCuts_);
  metadataTree_->Branch("baseCollectionPtCut", &baseCollectionPtCut_);
  metadataTree_->Branch("baseCaRadius", &baseCaRadius_);
  metadataTree_->Branch("configId", &configId_);
  metadataTree_->Branch("configBaseIndex", &configBaseIndex_);
  metadataTree_->Branch("configCollectionPtCut", &configCollectionPtCut_);
  metadataTree_->Branch("configCaRadius", &configCaRadius_);
  metadataTree_->Branch("configCosThrust", &configCosThrust_);
  metadataTree_->Branch("statusCodes", &statusCodes_);
  metadataTree_->Branch("statusNames", &statusNames_);
  metadataTree_->Branch("processedEvents", &processedEvents_);
  metadataTree_->Branch("sumWeights", &sumWeights_);
  metadataTree_->Branch("sumWeights2", &sumWeights2_);

  eventsTree_ =
      fs->make<TTree>("Events", "Compact factorized optimization scan");
  eventsTree_->SetAutoFlush(-5 * 1024 * 1024);
  eventsTree_->Branch("run", &run_);
  eventsTree_->Branch("lumi", &lumi_);
  eventsTree_->Branch("event", &event_);
  eventsTree_->Branch("genWeight", &genWeight_);
  eventsTree_->Branch("akJetPt", &akJetPt_);
  eventsTree_->Branch("nCAJets", &nCAJets_);
  eventsTree_->Branch("recoStatus", &recoStatus_);
  eventsTree_->Branch("nAmbiguous", &nAmbiguous_);
  eventsTree_->Branch("sj1Mass", &sj1Mass_);
  eventsTree_->Branch("sj2Mass", &sj2Mass_);
}

std::vector<WeightedParticle>
CompactOptimizationScanNtuplizer::makeWeightedParticles(
    const pat::PackedCandidateCollection &candidates) const {
  std::vector<WeightedParticle> particles;
  particles.reserve(candidates.size());
  for (const auto &candidate : candidates) {
    const double weight = candidate.puppiWeight();
    if (!std::isfinite(weight) || weight <= 0.) {
      continue;
    }
    TLorentzVector p4;
    p4.SetPxPyPzE(weight * candidate.px(), weight * candidate.py(),
                  weight * candidate.pz(), weight * candidate.energy());
    if (!finiteP4(p4) || p4.E() <= 0.) {
      continue;
    }
    particles.push_back({p4});
  }
  return particles;
}

std::vector<AKJetCache> CompactOptimizationScanNtuplizer::clusterAKJets(
    const std::vector<WeightedParticle> &particles) const {
  std::vector<AKJetCache> result;
  if (particles.empty()) {
    return result;
  }

  std::vector<fastjet::PseudoJet> inputs;
  inputs.reserve(particles.size());
  for (std::size_t index = 0; index < particles.size(); ++index) {
    const auto &p4 = particles[index].labP4;
    fastjet::PseudoJet pseudojet(p4.Px(), p4.Py(), p4.Pz(), p4.E());
    pseudojet.set_user_index(static_cast<int>(index));
    inputs.push_back(pseudojet);
  }

  const double minimumPt = collectionPtCuts_.front();
  const fastjet::JetDefinition definition(fastjet::antikt_algorithm, akRadius_);
  const fastjet::ClusterSequence sequence(inputs, definition);
  const auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets(minimumPt));
  result.reserve(jets.size());
  for (const auto &jet : jets) {
    if (!std::isfinite(jet.perp()) || jet.perp() <= minimumPt) {
      continue;
    }
    AKJetCache record;
    record.labP4.SetPxPyPzE(jet.px(), jet.py(), jet.pz(), jet.e());
    if (!finiteP4(record.labP4)) {
      throw cms::Exception("NumericalFailure")
          << "FastJet returned a non-finite AK jet";
    }
    const auto constituents = jet.constituents();
    record.constituentIndices.reserve(constituents.size());
    for (const auto &constituent : constituents) {
      const int index = constituent.user_index();
      if (index < 0 || static_cast<std::size_t>(index) >= particles.size()) {
        throw cms::Exception("NumericalFailure")
            << "AK constituent index is out of range";
      }
      record.constituentIndices.push_back(static_cast<std::uint32_t>(index));
    }
    result.push_back(std::move(record));
  }
  return result;
}

bool CompactOptimizationScanNtuplizer::findThrustAxis(
    const std::vector<TLorentzVector> &particles, TVector3 &axis) {
  if (particles.empty()) {
    return false;
  }
  const auto hardest = std::max_element(
      particles.begin(), particles.end(),
      [](const TLorentzVector &lhs, const TLorentzVector &rhs) {
        return lhs.Vect().Mag2() < rhs.Vect().Mag2();
      });
  if (hardest == particles.end() || !finiteVector(hardest->Vect()) ||
      hardest->Vect().Mag2() <= 0.) {
    return false;
  }
  axis = hardest->Vect().Unit();
  constexpr int maxIterations = 100;
  constexpr double tolerance = 1.e-6;
  bool converged = false;
  for (int iteration = 0; iteration < maxIterations; ++iteration) {
    TVector3 nextAxis;
    for (const auto &particle : particles) {
      const TVector3 momentum = particle.Vect();
      nextAxis += momentum.Dot(axis) >= 0. ? momentum : -momentum;
    }
    if (!finiteVector(nextAxis) || nextAxis.Mag2() <= 0.) {
      return false;
    }
    nextAxis = nextAxis.Unit();
    if ((nextAxis - axis).Mag() < tolerance) {
      axis = nextAxis;
      converged = true;
      break;
    }
    axis = nextAxis;
  }
  return converged && finiteVector(axis) && axis.Mag2() > 0.;
}

SelectedSystem CompactOptimizationScanNtuplizer::buildSelectedSystem(
    std::size_t nSelectedAK, const std::vector<AKJetCache> &akJets,
    const std::vector<WeightedParticle> &particles) const {
  SelectedSystem result;
  if (nSelectedAK == 0) {
    result.status = RecoStatus::noSelectedJets;
    return result;
  }

  std::vector<std::uint32_t> indices;
  for (std::size_t jetIndex = 0; jetIndex < nSelectedAK; ++jetIndex) {
    const auto &constituents = akJets[jetIndex].constituentIndices;
    indices.insert(indices.end(), constituents.begin(), constituents.end());
  }
  if (indices.empty()) {
    result.status = RecoStatus::noSelectedConstituents;
    return result;
  }

  TLorentzVector total;
  for (const std::uint32_t index : indices) {
    if (index >= particles.size()) {
      result.status = RecoStatus::numericalFailure;
      return result;
    }
    total += particles[index].labP4;
  }
  if (!finiteP4(total) || total.E() <= 0.) {
    result.status = RecoStatus::invalidCOM;
    return result;
  }
  result.beta = total.BoostVector();
  const double beta2 = result.beta.Mag2();
  if (!finiteVector(result.beta) || !std::isfinite(beta2) || beta2 < 0. ||
      beta2 >= 1.) {
    result.status = RecoStatus::invalidCOM;
    return result;
  }

  result.comConstituents.reserve(indices.size());
  for (const std::uint32_t index : indices) {
    TLorentzVector boosted = particles[index].labP4;
    boosted.Boost(-result.beta);
    if (!finiteP4(boosted) || boosted.E() <= 0.) {
      result.status = RecoStatus::invalidCOM;
      result.comConstituents.clear();
      return result;
    }
    result.comConstituents.push_back(boosted);
  }
  if (!findThrustAxis(result.comConstituents, result.thrustAxis)) {
    result.status = RecoStatus::invalidThrust;
    result.comConstituents.clear();
    return result;
  }
  result.status = RecoStatus::valid;
  return result;
}

CAJetCache
CompactOptimizationScanNtuplizer::clusterCAJets(const SelectedSystem &system,
                                                double caRadius) const {
  CAJetCache result;
  if (system.status != RecoStatus::valid) {
    result.status = system.status;
    return result;
  }
  result.beta = system.beta;

  std::vector<fastjet::PseudoJet> inputs;
  inputs.reserve(system.comConstituents.size());
  for (const auto &p4 : system.comConstituents) {
    inputs.emplace_back(p4.Px(), p4.Py(), p4.Pz(), p4.E());
  }
  const fastjet::JetDefinition definition(fastjet::cambridge_algorithm,
                                          caRadius);
  const fastjet::ClusterSequence sequence(inputs, definition);
  const auto jets = fastjet::sorted_by_pt(sequence.inclusive_jets());
  if (jets.empty()) {
    result.status = RecoStatus::noCAJets;
    return result;
  }
  if (jets.size() > std::numeric_limits<std::uint16_t>::max()) {
    result.status = RecoStatus::numericalFailure;
    return result;
  }

  result.comJets.reserve(jets.size());
  result.cosToThrust.reserve(jets.size());
  for (const auto &jet : jets) {
    TLorentzVector p4;
    p4.SetPxPyPzE(jet.px(), jet.py(), jet.pz(), jet.e());
    const TVector3 momentum = p4.Vect();
    const double magnitude = momentum.Mag();
    if (!finiteP4(p4) || p4.E() <= 0. || !finiteVector(momentum) ||
        !std::isfinite(magnitude) || magnitude <= 0.) {
      result.status = RecoStatus::numericalFailure;
      result.comJets.clear();
      result.cosToThrust.clear();
      return result;
    }
    const double rawCosine = momentum.Dot(system.thrustAxis) / magnitude;
    if (!std::isfinite(rawCosine)) {
      result.status = RecoStatus::numericalFailure;
      result.comJets.clear();
      result.cosToThrust.clear();
      return result;
    }
    result.comJets.push_back(p4);
    result.cosToThrust.push_back(std::clamp(rawCosine, -1., 1.));
  }
  result.status = RecoStatus::valid;
  return result;
}

std::vector<std::int8_t>
CompactOptimizationScanNtuplizer::classificationForCut(const CAJetCache &caJets,
                                                       double cosCut) const {
  std::vector<std::int8_t> labels;
  labels.reserve(caJets.cosToThrust.size());
  if (cosCut == 0.) {
    for (const double cosine : caJets.cosToThrust) {
      labels.push_back(cosine >= 0. ? 1 : 2);
    }
    return labels;
  }
  if (cosCut == 1.) {
    labels.assign(caJets.cosToThrust.size(), 0);
    return labels;
  }
  for (const double cosine : caJets.cosToThrust) {
    if (cosine >= cosCut) {
      labels.push_back(1);
    } else if (cosine <= -cosCut) {
      labels.push_back(2);
    } else {
      labels.push_back(0);
    }
  }
  return labels;
}

bool CompactOptimizationScanNtuplizer::physicalMass(const TLorentzVector &p4,
                                                    double &mass) {
  if (!finiteP4(p4) || p4.E() <= 0.) {
    return false;
  }
  const double mass2 = p4.M2();
  const double scale = std::max(1., p4.E() * p4.E() + p4.Vect().Mag2());
  if (!std::isfinite(mass2) || mass2 < -1.e-9 * scale) {
    return false;
  }
  mass = std::sqrt(std::max(0., mass2));
  return std::isfinite(mass);
}

AssignmentResult
CompactOptimizationScanNtuplizer::failure(RecoStatus status,
                                          std::size_t nAmbiguous) {
  AssignmentResult result;
  result.status = status;
  result.nAmbiguous = static_cast<std::uint16_t>(std::min(
      nAmbiguous,
      static_cast<std::size_t>(std::numeric_limits<std::uint16_t>::max())));
  return result;
}

AssignmentResult CompactOptimizationScanNtuplizer::assignCAJets(
    const CAJetCache &caJets, const std::vector<std::int8_t> &labels) const {
  if (caJets.status != RecoStatus::valid) {
    return failure(caJets.status);
  }
  if (labels.size() != caJets.comJets.size()) {
    return failure(RecoStatus::numericalFailure);
  }

  TLorentzVector fixed0;
  TLorentzVector fixed1;
  std::size_t nFixed0 = 0;
  std::size_t nFixed1 = 0;
  std::vector<TLorentzVector> ambiguous;
  ambiguous.reserve(labels.size());
  for (std::size_t index = 0; index < labels.size(); ++index) {
    if (labels[index] == 1) {
      fixed0 += caJets.comJets[index];
      ++nFixed0;
    } else if (labels[index] == 2) {
      fixed1 += caJets.comJets[index];
      ++nFixed1;
    } else if (labels[index] == 0) {
      ambiguous.push_back(caJets.comJets[index]);
    } else {
      return failure(RecoStatus::numericalFailure);
    }
  }

  const std::size_t nAmbiguous = ambiguous.size();
  if (nAmbiguous > maxAmbiguousCAJets_) {
    return failure(RecoStatus::complexityGuard, nAmbiguous);
  }
  if (nAmbiguous >= std::numeric_limits<std::uint64_t>::digits) {
    return failure(RecoStatus::complexityGuard, nAmbiguous);
  }

  const std::uint64_t nMasks = std::uint64_t{1} << nAmbiguous;
  std::vector<TLorentzVector> subsetSums(nMasks);
  for (std::uint64_t mask = 1; mask < nMasks; ++mask) {
    const std::uint64_t leastBit = mask & (~mask + 1);
    const unsigned int bitIndex =
        static_cast<unsigned int>(__builtin_ctzll(leastBit));
    subsetSums[mask] = subsetSums[mask ^ leastBit] + ambiguous[bitIndex];
  }
  TLorentzVector totalAmbiguous;
  for (const auto &jet : ambiguous) {
    totalAmbiguous += jet;
  }

  double bestScore = std::numeric_limits<double>::infinity();
  TLorentzVector best0;
  TLorentzVector best1;
  bool found = false;
  const bool removeComplementSymmetry =
      (nFixed0 == 0 && nFixed1 == 0 && nAmbiguous > 0);
  for (std::uint64_t mask = 0; mask < nMasks; ++mask) {
    if (removeComplementSymmetry && (mask & std::uint64_t{1}) == 0) {
      continue;
    }
    const std::size_t assigned0 =
        static_cast<std::size_t>(__builtin_popcountll(mask));
    const std::size_t assigned1 = nAmbiguous - assigned0;
    if (nFixed0 + assigned0 == 0 || nFixed1 + assigned1 == 0) {
      continue;
    }
    const TLorentzVector side0 = fixed0 + subsetSums[mask];
    const TLorentzVector side1 = fixed1 + (totalAmbiguous - subsetSums[mask]);
    double mass0 = 0.;
    double mass1 = 0.;
    if (!physicalMass(side0, mass0) || !physicalMass(side1, mass1)) {
      continue;
    }
    const double denominator = mass0 + mass1;
    if (!std::isfinite(denominator) || denominator <= 0.) {
      continue;
    }
    const double score = std::abs(mass0 - mass1) / denominator;
    if (!std::isfinite(score)) {
      continue;
    }
    if (score < bestScore) {
      bestScore = score;
      best0 = side0;
      best1 = side1;
      found = true;
    }
  }
  if (!found) {
    return failure(RecoStatus::noValidPartition, nAmbiguous);
  }

  double mass0 = 0.;
  double mass1 = 0.;
  if (!physicalMass(best0, mass0) || !physicalMass(best1, mass1)) {
    return failure(RecoStatus::numericalFailure, nAmbiguous);
  }
  TLorentzVector lab0 = best0;
  TLorentzVector lab1 = best1;
  lab0.Boost(caJets.beta);
  lab1.Boost(caJets.beta);
  if (!finiteP4(lab0) || !finiteP4(lab1)) {
    return failure(RecoStatus::numericalFailure, nAmbiguous);
  }

  AssignmentResult result;
  result.status = RecoStatus::valid;
  result.nAmbiguous = static_cast<std::uint16_t>(nAmbiguous);
  if (lab0.Pt() >= lab1.Pt()) {
    result.leadingPtMass = static_cast<float>(mass0);
    result.subleadingPtMass = static_cast<float>(mass1);
  } else {
    result.leadingPtMass = static_cast<float>(mass1);
    result.subleadingPtMass = static_cast<float>(mass0);
  }
  return result;
}

void CompactOptimizationScanNtuplizer::analyze(const edm::Event &input,
                                               const edm::EventSetup &) {
  run_ = input.id().run();
  lumi_ = input.luminosityBlock();
  event_ = input.id().event();
  genWeight_ = 1.f;
  const auto generatorInfo = input.getHandle(generatorInfoToken_);
  if (generatorInfo.isValid() && std::isfinite(generatorInfo->weight())) {
    genWeight_ = static_cast<float>(generatorInfo->weight());
  }
  ++processedEvents_;
  sumWeights_ += genWeight_;
  sumWeights2_ += static_cast<double>(genWeight_) * genWeight_;

  akJetPt_.clear();
  nCAJets_.assign(baseConfigurations_.size(), 0);
  recoStatus_.assign(configId_.size(),
                     static_cast<std::uint8_t>(RecoStatus::numericalFailure));
  nAmbiguous_.assign(configId_.size(), 0);
  sj1Mass_.assign(configId_.size(), kInvalidMass);
  sj2Mass_.assign(configId_.size(), kInvalidMass);

  const auto &candidates = input.get(packedPFToken_);
  try {
    const auto particles = makeWeightedParticles(candidates);
    const auto akJets = clusterAKJets(particles);
    akJetPt_.reserve(akJets.size());
    for (const auto &jet : akJets) {
      akJetPt_.push_back(static_cast<float>(jet.labP4.Pt()));
    }

    std::vector<std::size_t> selectedCounts(collectionPtCuts_.size(), 0);
    for (std::size_t iPt = 0; iPt < collectionPtCuts_.size(); ++iPt) {
      while (selectedCounts[iPt] < akJets.size() &&
             akJetPt_[selectedCounts[iPt]] >
                 static_cast<float>(collectionPtCuts_[iPt])) {
        ++selectedCounts[iPt];
      }
    }

    std::map<std::size_t, SelectedSystem> selectedSystemCache;
    std::map<std::pair<std::size_t, std::size_t>, CAJetCache> caCache;
    for (std::size_t iBase = 0; iBase < baseConfigurations_.size(); ++iBase) {
      const auto &base = baseConfigurations_[iBase];
      const std::size_t nSelected = selectedCounts[base.collectionPtIndex];
      auto selectedIt = selectedSystemCache.find(nSelected);
      if (selectedIt == selectedSystemCache.end()) {
        selectedIt =
            selectedSystemCache
                .emplace(nSelected,
                         buildSelectedSystem(nSelected, akJets, particles))
                .first;
      }

      const auto caKey = std::make_pair(nSelected, base.caRadiusIndex);
      auto caIt = caCache.find(caKey);
      if (caIt == caCache.end()) {
        try {
          caIt = caCache
                     .emplace(caKey,
                              clusterCAJets(selectedIt->second, base.caRadius))
                     .first;
        } catch (const fastjet::Error &) {
          CAJetCache failed;
          failed.status = RecoStatus::numericalFailure;
          caIt = caCache.emplace(caKey, std::move(failed)).first;
        }
      }

      CAJetCache &caResult = caIt->second;
      if (caResult.status == RecoStatus::valid) {
        nCAJets_[iBase] = static_cast<std::uint16_t>(caResult.comJets.size());
      }
      for (std::size_t iCos = 0; iCos < cosThrustCuts_.size(); ++iCos) {
        const std::size_t configIndex = iBase * cosThrustCuts_.size() + iCos;
        AssignmentResult assignment;
        if (caResult.status != RecoStatus::valid) {
          assignment = failure(caResult.status);
        } else {
          const auto labels =
              classificationForCut(caResult, cosThrustCuts_[iCos]);
          auto cached = caResult.assignmentCache.find(labels);
          if (cached == caResult.assignmentCache.end()) {
            cached = caResult.assignmentCache
                         .emplace(labels, assignCAJets(caResult, labels))
                         .first;
          }
          assignment = cached->second;
        }
        recoStatus_[configIndex] = static_cast<std::uint8_t>(assignment.status);
        nAmbiguous_[configIndex] = assignment.nAmbiguous;
        sj1Mass_[configIndex] = assignment.leadingPtMass;
        sj2Mass_[configIndex] = assignment.subleadingPtMass;
      }
    }
  } catch (const fastjet::Error &) {
    // Retain the event with numerical_failure for every configuration. This
    // preserves denominator accounting instead of aborting or dropping it.
  } catch (const cms::Exception &error) {
    if (error.category() != "NumericalFailure") {
      throw;
    }
  }

  eventsTree_->Fill();
}

void CompactOptimizationScanNtuplizer::endJob() { metadataTree_->Fill(); }

void CompactOptimizationScanNtuplizer::fillDescriptions(
    edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription description;
  description.add<edm::InputTag>("packedPFCandidates",
                                 edm::InputTag("packedPFCandidates"));
  description.add<edm::InputTag>("generatorInfo", edm::InputTag("generator"));
  description.add<std::string>("sampleName", "unknown");
  description.add<double>("akRadius", 0.8);
  description.add<std::vector<double>>(
      "collectionPtCuts", {100., 120., 140., 160., 180., 200., 220., 240., 260.,
                           280., 300., 320., 340., 360., 380., 400.});
  description.add<std::vector<double>>("caRadii",
                                       {0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6});
  description.add<std::vector<double>>(
      "cosThrustCuts",
      {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95});
  description.add<std::vector<unsigned int>>("defaultGateJetCounts",
                                             {0, 1, 2, 3, 4, 5, 6});
  description.add<std::vector<double>>(
      "defaultGatePtCuts", {100., 120., 140., 160., 180., 200., 220., 240.,
                            260., 280., 300., 320., 340., 360., 380., 400.});
  description.add<unsigned int>("maxAmbiguousCAJets", 12);
  description.add<bool>("enforceLegacyRadiusConstraint", true);
  description.add<unsigned int>("compressionLevel", 6);
  descriptions.add("compactOptimizationScanNtuplizer", description);
}

DEFINE_FWK_MODULE(CompactOptimizationScanNtuplizer);
