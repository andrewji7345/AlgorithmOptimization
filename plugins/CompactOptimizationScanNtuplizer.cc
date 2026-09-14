#include <memory>
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
#include "SuuAnalysis/ExistingOptimization/interface/AN2017Selection.h"
#include "SuuAnalysis/ExistingOptimization/interface/AN2017Regions.h"
#include "PhysicsTools/CandUtils/interface/Thrust.h"
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "TVector3.h"

#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
  int pdgId = 0;
  int charge = 0;
};

struct AKJetCache {
  TLorentzVector labP4;
  std::vector<std::uint32_t> constituentIndices;
  bool passesVeto = true;
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
  float suuMass = kInvalidMass;
  std::uint16_t leadingNCA4E300 = 0;
  std::uint16_t subleadingNCA4E300 = 0;
  std::uint16_t leadingNCA4E50 = 0, subleadingNCA4E50 = 0;
  float leadingMassE100 = kInvalidMass, subleadingMassE100 = kInvalidMass;
};

struct CAJetCache {
  RecoStatus status = RecoStatus::numericalFailure;
  TVector3 beta;
  std::vector<TLorentzVector> comJets;
  std::vector<std::vector<TLorentzVector>> comJetConstituents;
  std::vector<double> cosToThrust;
  std::map<std::vector<std::int8_t>, AssignmentResult> assignmentCache;
  bool sourceReference = false;
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
  clusterAKJets(std::vector<WeightedParticle> &) const;
  SelectedSystem
  buildSelectedSystem(std::size_t, const std::vector<AKJetCache> &,
                      const std::vector<WeightedParticle> &) const;
  CAJetCache clusterCAJets(const SelectedSystem &, double) const;
  AssignmentResult assignCAJets(const CAJetCache &,
                                const std::vector<std::int8_t> &) const;
  std::vector<std::int8_t> classificationForCut(const CAJetCache &,
                                                double) const;
  AssignmentResult reconstructReference(const std::vector<TLorentzVector>&) const;

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
  bool analysisMode_ = false;
  std::unique_ptr<AN2017Selection> analysis_;
  std::string analysisSelection_ = "none";
  std::string correctionPrescription_ = "none";
  std::string sampleKind_ = "signal";
  bool passesTrigger_ = false, passesFilters_ = false;
  bool passesLeptonVeto_ = false, passesJetVeto_ = false, passesBaseline_ = false;
  float analysisHT_ = 0.f, analysisWeight_ = 1.f;
  float analysisBTagWeight_ = 1.f;
  bool analysisBTagWeightFallback_ = false;
  std::uint16_t analysisNAK4_ = 0, analysisNAK8_ = 0, analysisNHeavyAK8_ = 0, analysisNBTags_ = 0;
  std::vector<std::uint8_t> passesSignalRegion_;
  std::vector<std::uint8_t> passesRecoJetVeto_;
  std::vector<std::uint16_t> sj1NCA4E300_, sj2NCA4E300_;
  std::vector<std::uint16_t> sj1NCA4E50_, sj2NCA4E50_;
  std::vector<float> sj1MassE100_, sj2MassE100_;
  std::vector<std::uint8_t> passesControlRegion_, passesAT0b_, passesAT1b_;
  std::uint32_t analysisObservableVersion_ = 1;
  std::string analysisSystematic_ = "nominal";
  std::string referenceReconstruction_ = "AN23-067-PATAK8-CA8-Thrust-source-port-v1";
  std::vector<std::string> weightVariationNames_{
      "pileupUp", "pileupDown", "prefiringUp", "prefiringDown",
      "btagHFCorrelatedUp", "btagHFCorrelatedDown", "btagHFUncorrelatedUp", "btagHFUncorrelatedDown",
      "btagLFCorrelatedUp", "btagLFCorrelatedDown", "btagLFUncorrelatedUp", "btagLFUncorrelatedDown",
      "topPtUp", "topPtDown"};
  std::vector<float> analysisWeightVariations_, referenceWeightVariations_;
  std::vector<std::uint8_t> btagWeightVariationFallback_;
  std::vector<float> analysisBTagJetPt_, analysisBTagJetEta_, analysisBTagJetDiscriminator_;
  float referenceWeight_ = 1.f;
  std::uint8_t referenceRecoStatus_ = 8, referenceRegion_ = 0;
  float referenceSJ1Mass_ = kInvalidMass, referenceSJ2Mass_ = kInvalidMass, referenceSuuMass_ = kInvalidMass;
  float referenceSJ1MassE100_ = kInvalidMass, referenceSJ2MassE100_ = kInvalidMass;
  std::uint16_t referenceSJ1NCA4E50_ = 0, referenceSJ2NCA4E50_ = 0;
  std::uint16_t referenceSJ1NCA4E300_ = 0, referenceSJ2NCA4E300_ = 0;

  std::vector<BaseConfiguration> baseConfigurations_;
  TTree *metadataTree_ = nullptr;
  TTree *eventsTree_ = nullptr;

  // Metadata (one row per output file).
  std::uint32_t schemaVersion_ = 2;
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
  std::vector<float> suuMass_;
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
  analysisMode_ = config.getParameter<bool>("analysisMode");
  sampleKind_ = config.getParameter<std::string>("sampleKind");
  if (sampleKind_ != "signal" && sampleKind_ != "background")
    throw cms::Exception("Configuration") << "sampleKind must be signal or background";
  if (analysisMode_) {
    if (std::abs(akRadius_ - 0.4) > 1.e-6 && std::abs(akRadius_ - 0.8) > 1.e-6)
      throw cms::Exception("Configuration") << "Calibrated sensitivity supports AK R=0.4 or 0.8 only; arbitrary-radius JEC is unavailable";
    analysis_ = std::make_unique<AN2017Selection>(config.getParameter<edm::ParameterSet>("analysis"), consumesCollector());
    analysisSystematic_ = config.getParameter<edm::ParameterSet>("analysis").getParameter<std::string>("systematic");
    schemaVersion_ = 3;
    useJEC_ = true;
    analysisSelection_ = "AN-23-067-UL2017-cutbased-v1";
    correctionPrescription_ = "UL2017-AK4CHS-baseline-AK4AK8Puppi-reco-JEC-JER-btagGuard-v3";
  } else if (sampleKind_ != "signal") {
    throw cms::Exception("Configuration") << "Background production requires analysisMode";
  }

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
  eventsTree_->Branch("suuMass", &suuMass_);
  if (analysisMode_) {
    metadataTree_->Branch("analysisSelection", &analysisSelection_);
    metadataTree_->Branch("correctionPrescription", &correctionPrescription_);
    metadataTree_->Branch("sampleKind", &sampleKind_);
    metadataTree_->Branch("analysisObservableVersion", &analysisObservableVersion_);
    metadataTree_->Branch("analysisSystematic", &analysisSystematic_);
    metadataTree_->Branch("referenceReconstruction", &referenceReconstruction_);
    metadataTree_->Branch("weightVariationNames", &weightVariationNames_);
    eventsTree_->Branch("analysisWeight", &analysisWeight_);
    eventsTree_->Branch("analysisBTagWeight", &analysisBTagWeight_);
    eventsTree_->Branch("analysisBTagWeightFallback", &analysisBTagWeightFallback_);
    eventsTree_->Branch("passesBaseline", &passesBaseline_);
    eventsTree_->Branch("passesSignalRegion", &passesSignalRegion_);
    eventsTree_->Branch("passesRecoJetVeto", &passesRecoJetVeto_);
    eventsTree_->Branch("passesTrigger", &passesTrigger_);
    eventsTree_->Branch("passesFilters", &passesFilters_);
    eventsTree_->Branch("passesLeptonVeto", &passesLeptonVeto_);
    eventsTree_->Branch("passesJetVeto", &passesJetVeto_);
    eventsTree_->Branch("analysisHT", &analysisHT_);
    eventsTree_->Branch("analysisNAK4", &analysisNAK4_);
    eventsTree_->Branch("analysisNAK8", &analysisNAK8_);
    eventsTree_->Branch("analysisNHeavyAK8", &analysisNHeavyAK8_);
    eventsTree_->Branch("analysisNBTags", &analysisNBTags_);
    eventsTree_->Branch("sj1NCA4E300", &sj1NCA4E300_);
    eventsTree_->Branch("sj2NCA4E300", &sj2NCA4E300_);
    eventsTree_->Branch("sj1NCA4E50", &sj1NCA4E50_);
    eventsTree_->Branch("sj2NCA4E50", &sj2NCA4E50_);
    eventsTree_->Branch("sj1MassE100", &sj1MassE100_);
    eventsTree_->Branch("sj2MassE100", &sj2MassE100_);
    eventsTree_->Branch("passesControlRegion", &passesControlRegion_);
    eventsTree_->Branch("passesAT0b", &passesAT0b_);
    eventsTree_->Branch("passesAT1b", &passesAT1b_);
    eventsTree_->Branch("analysisWeightVariations", &analysisWeightVariations_);
    eventsTree_->Branch("referenceWeight", &referenceWeight_);
    eventsTree_->Branch("referenceWeightVariations", &referenceWeightVariations_);
    eventsTree_->Branch("btagWeightVariationFallback", &btagWeightVariationFallback_);
    eventsTree_->Branch("analysisBTagJetPt", &analysisBTagJetPt_);
    eventsTree_->Branch("analysisBTagJetEta", &analysisBTagJetEta_);
    eventsTree_->Branch("analysisBTagJetDiscriminator", &analysisBTagJetDiscriminator_);
    eventsTree_->Branch("referenceRecoStatus", &referenceRecoStatus_);
    eventsTree_->Branch("referenceRegion", &referenceRegion_);
    eventsTree_->Branch("referenceSJ1Mass", &referenceSJ1Mass_);
    eventsTree_->Branch("referenceSJ2Mass", &referenceSJ2Mass_);
    eventsTree_->Branch("referenceSuuMass", &referenceSuuMass_);
    eventsTree_->Branch("referenceSJ1MassE100", &referenceSJ1MassE100_);
    eventsTree_->Branch("referenceSJ2MassE100", &referenceSJ2MassE100_);
    eventsTree_->Branch("referenceSJ1NCA4E50", &referenceSJ1NCA4E50_);
    eventsTree_->Branch("referenceSJ2NCA4E50", &referenceSJ2NCA4E50_);
    eventsTree_->Branch("referenceSJ1NCA4E300", &referenceSJ1NCA4E300_);
    eventsTree_->Branch("referenceSJ2NCA4E300", &referenceSJ2NCA4E300_);
  }
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
    particles.push_back({p4, candidate.pdgId(), candidate.charge()});
  }
  return particles;
}

std::vector<AKJetCache> CompactOptimizationScanNtuplizer::clusterAKJets(
    std::vector<WeightedParticle> &particles) const {
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

  // Select on corrected pT; no raw threshold may discard a jet migrating up.
  const double minimumPt = analysisMode_ ? 0. : collectionPtCuts_.front();
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
    if (analysisMode_) {
      if (std::abs(record.labP4.Eta()) > 2.5) continue;
      const double factor = analysis_->correctReclusteredJet(record.labP4, akRadius_);
      if (!std::isfinite(factor) || factor < 0.)
        throw cms::Exception("InvalidCorrection") << "Non-finite/negative AK jet correction";
      record.labP4 *= factor;
      record.passesVeto = analysis_->passesReclusteredJetVeto(record.labP4);
      for (auto index : record.constituentIndices) particles[index].labP4 *= factor;
    }
    result.push_back(std::move(record));
  }
  if (analysisMode_) {
    std::stable_sort(result.begin(), result.end(), [](const auto &a, const auto &b) {
      return a.labP4.Pt() > b.labP4.Pt();
    });
    // Radius-specific object requirements apply after corrections, in pT order.
    const bool isAK4 = std::abs(akRadius_ - 0.4) < 1.e-6;
    std::vector<AKJetCache> accepted;
    for (auto &jet : result) {
      if (jet.labP4.Pt() <= collectionPtCuts_.front() ||
          (isAK4 ? jet.labP4.Pt() <= 50. : std::hypot(jet.labP4.Pt(), jet.labP4.M()) <= 300.) ||
          std::abs(jet.labP4.Eta()) >= (isAK4 || accepted.size() < 2 ? 2.5 : 1.4)) continue;
      double totalE=0., nh=0., ne=0., mu=0., ce=0., ch=0.;
      for (auto index : jet.constituentIndices) {
        const auto &part=particles[index]; const double energy=part.labP4.E();
        totalE += energy;
        const int id=std::abs(part.pdgId);
        if (id==130 || id==2112) nh += energy;
        if (id==22) ne += energy;
        if (id==13) mu += energy;
        if (id==11) ce += energy;
        if (part.charge != 0 && id != 11 && id != 13) ch += energy;
      }
      if (totalE<=0. || jet.constituentIndices.size()<2 || ch<=0. ||
          nh/totalE>=0.9 || ne/totalE>=0.9 || mu/totalE>=0.8 || ce/totalE>=0.8) continue;
      accepted.push_back(std::move(jet));
    }
    return accepted;
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
    if (analysisMode_) {
      std::vector<TLorentzVector> constituents;
      for (const auto &part : jet.constituents())
        constituents.emplace_back(part.px(), part.py(), part.pz(), part.e());
      result.comJetConstituents.push_back(std::move(constituents));
    }
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

AssignmentResult CompactOptimizationScanNtuplizer::reconstructReference(
    const std::vector<TLorentzVector>& particles) const {
  if (particles.empty()) return failure(RecoStatus::noSelectedConstituents);
  TLorentzVector total;
  for (const auto& part : particles) total += part;
  if (!finiteP4(total) || total.E() <= 0. || total.BoostVector().Mag2() >= 1.)
    return failure(RecoStatus::invalidCOM);
  CAJetCache cache;
  cache.sourceReference = true;
  cache.beta = total.BoostVector();
  std::vector<fastjet::PseudoJet> inputs;
  for (auto part : particles) {
    part.Boost(-cache.beta);
    if (!finiteP4(part)) return failure(RecoStatus::invalidCOM);
    inputs.emplace_back(part.Px(), part.Py(), part.Pz(), part.E());
  }
  const fastjet::ClusterSequence sequence(inputs, fastjet::JetDefinition(fastjet::cambridge_algorithm, 0.8));
  // inclusive_jets(10) is a pT threshold in FastJet, despite the source's
  // adjacent comment describing an energy threshold.
  const auto jets = fastjet::sorted_by_E(sequence.inclusive_jets(10.));
  std::vector<reco::LeafCandidate> thrustParticles;
  for (const auto& jet : jets) {
    const auto parts = jet.constituents();
    if (parts.size() < 5) continue;
    for (const auto& part : parts) {
      thrustParticles.emplace_back(1, reco::Candidate::LorentzVector(part.px(), part.py(), part.pz(), part.E()));
      // Preserve the source's explicit old FastJet/ROOT guard, but retain an
      // invalid-reference event in the ntuple so the MC denominator is intact.
      if (thrustParticles.size() > 300) return failure(RecoStatus::complexityGuard);
    }
  }
  if (thrustParticles.empty()) return failure(RecoStatus::invalidThrust);
  const Thrust thrust(thrustParticles.begin(), thrustParticles.end());
  const TVector3 axis(thrust.axis().X(), thrust.axis().Y(), thrust.axis().Z());
  if (!finiteVector(axis) || axis.Mag2() <= 0.) return failure(RecoStatus::invalidThrust);
  std::vector<std::int8_t> labels;
  for (const auto& jet : jets) {
    const auto parts = jet.constituents();
    if (parts.size() < 2) continue;
    const TLorentzVector p(jet.px(), jet.py(), jet.pz(), jet.E());
    const double cosine = std::cos(p.Vect().Angle(axis));
    if (!std::isfinite(cosine)) return failure(RecoStatus::numericalFailure);
    cache.comJets.push_back(p);
    labels.push_back(cosine > 0.85 ? 1 : cosine < -0.85 ? 2 : 0);
    std::vector<TLorentzVector> keptParts;
    // In the source, 2-4 constituent CA8 jets affect the mass-balancing
    // assignment, but are excluded from final superjet masses and substructure.
    if (parts.size() >= 5)
      for (const auto& part : parts) keptParts.emplace_back(part.px(), part.py(), part.pz(), part.E());
    cache.comJetConstituents.push_back(std::move(keptParts));
  }
  if (cache.comJets.empty()) return failure(RecoStatus::noCAJets);
  cache.status = RecoStatus::valid;
  return assignCAJets(cache, labels);
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
  if (nAmbiguous > (caJets.sourceReference ? 14U : maxAmbiguousCAJets_)) {
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
  std::uint64_t bestMask = 0;
  const bool removeComplementSymmetry =
      (!caJets.sourceReference && nFixed0 == 0 && nFixed1 == 0 && nAmbiguous > 0);
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
    TLorentzVector side0 = fixed0 + subsetSums[mask];
    TLorentzVector side1 = fixed1 + (totalAmbiguous - subsetSums[mask]);
    if (caJets.sourceReference) {
      // Match sortJets.cc's nested-loop ordering and direct accumulation. Its
      // first ambiguity is the most significant bit; zero chooses negative SJ.
      side0 = fixed0; side1 = fixed1;
      for (std::size_t j=0; j<nAmbiguous; ++j)
        ((mask >> (nAmbiguous - 1 - j)) & 1U ? side0 : side1) += ambiguous[j];
    }
    double mass0 = 0.;
    double mass1 = 0.;
    if (!physicalMass(side0, mass0) || !physicalMass(side1, mass1)) {
      continue;
    }
    const double denominator = caJets.sourceReference ? std::min(mass0, mass1) : mass0 + mass1;
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
      bestMask = mask;
    }
  }
  if (!found) {
    return failure(RecoStatus::noValidPartition, nAmbiguous);
  }

  if (caJets.sourceReference) {
    best0 = TLorentzVector(); best1 = TLorentzVector();
    std::size_t j = 0;
    for (std::size_t i=0; i<labels.size(); ++i) {
      const int side = labels[i] == 1 ? 0 : labels[i] == 2 ? 1 :
          ((bestMask >> (nAmbiguous - 1 - j++)) & 1U ? 0 : 1);
      if (!caJets.comJetConstituents[i].empty()) (side == 0 ? best0 : best1) += caJets.comJets[i];
    }
  }

  double mass0 = 0.;
  double mass1 = 0.;
  if (!physicalMass(best0, mass0) || !physicalMass(best1, mass1)) {
    return failure(RecoStatus::numericalFailure, nAmbiguous);
  }
  // The invariant mass of the reconstructed pair is not mass0 + mass1.
  // Compute it in the selected-system COM frame before the lab boost.
  double suuMass = 0.;
  if (!physicalMass(best0 + best1, suuMass) ||
      !std::isfinite(static_cast<float>(suuMass))) {
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
  result.suuMass = static_cast<float>(suuMass);
  if (lab0.Pt() >= lab1.Pt()) {
    result.leadingPtMass = static_cast<float>(mass0);
    result.subleadingPtMass = static_cast<float>(mass1);
  } else {
    result.leadingPtMass = static_cast<float>(mass1);
    result.subleadingPtMass = static_cast<float>(mass0);
  }
  if (analysisMode_) {
    std::vector<fastjet::PseudoJet> sides[2];
    const TLorentzVector systems[2] = {best0, best1};
    std::size_t ambiguousIndex = 0;
    for (std::size_t i=0; i<labels.size(); ++i) {
      const int side = labels[i]==1 ? 0 : labels[i]==2 ? 1 :
                       ((bestMask >> (caJets.sourceReference ? nAmbiguous - 1 - ambiguousIndex++ : ambiguousIndex++)) & 1U ? 0 : 1);
      if (systems[side].M2() <= 0.) return failure(RecoStatus::numericalFailure, nAmbiguous);
      for (auto part : caJets.comJetConstituents[i]) {
        part.Boost(-systems[side].BoostVector());
        if (!finiteP4(part)) return failure(RecoStatus::numericalFailure, nAmbiguous);
        sides[side].emplace_back(part.Px(), part.Py(), part.Pz(), part.E());
      }
    }
    std::uint16_t counts[2] = {0,0}, counts50[2] = {0,0};
    float masses100[2] = {0.f, 0.f};
    for (int side=0; side<2; ++side) {
      const fastjet::JetDefinition definition(fastjet::cambridge_algorithm, 0.4);
      const fastjet::ClusterSequence sequence(sides[side], definition);
      TLorentzVector sum100;
      for (const auto &jet : sequence.inclusive_jets()) {
        if (jet.E() > 300.) ++counts[side];
        if (jet.E() > 50.) ++counts50[side];
        if (jet.E() > 100.) sum100 += TLorentzVector(jet.px(), jet.py(), jet.pz(), jet.E());
      }
      // Empty E>100 collection has invariant mass zero (the anti-tag case).
      const double mass2 = sum100.M2();
      if (!finiteP4(sum100) || mass2 < -1.e-9 * std::max(1., sum100.E()*sum100.E()))
        return failure(RecoStatus::numericalFailure, nAmbiguous);
      masses100[side] = std::sqrt(std::max(0., mass2));
    }
    const bool leadingFirst = lab0.Pt() >= lab1.Pt();
    result.leadingNCA4E300 = counts[leadingFirst ? 0 : 1];
    result.subleadingNCA4E300 = counts[leadingFirst ? 1 : 0];
    result.leadingNCA4E50 = counts50[leadingFirst ? 0 : 1];
    result.subleadingNCA4E50 = counts50[leadingFirst ? 1 : 0];
    result.leadingMassE100 = masses100[leadingFirst ? 0 : 1];
    result.subleadingMassE100 = masses100[leadingFirst ? 1 : 0];
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
  if (analysisMode_ && (!generatorInfo.isValid() || !std::isfinite(generatorInfo->weight())))
    throw cms::Exception("MissingInput") << "Sensitivity requires finite preselection generator weights";
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
  suuMass_.assign(configId_.size(), kInvalidMass);
  if (analysisMode_) {
    const auto info = analysis_->evaluate(input);
    passesTrigger_ = info.trigger; passesFilters_ = info.filters;
    passesLeptonVeto_ = info.leptonVeto; passesJetVeto_ = info.jetVeto;
    passesBaseline_ = info.baseline; analysisHT_ = info.ht;
    analysisWeight_ = info.weight;
    analysisBTagWeight_ = info.btagWeight;
    analysisBTagWeightFallback_ = info.btagWeightFallback;
    analysisWeightVariations_.assign(info.weightVariations.begin(), info.weightVariations.end());
    referenceWeight_ = info.referenceWeight;
    referenceWeightVariations_.assign(info.referenceWeightVariations.begin(), info.referenceWeightVariations.end());
    btagWeightVariationFallback_ = info.btagWeightVariationFallback;
    analysisBTagJetPt_ = info.btagJetPt;
    analysisBTagJetEta_ = info.btagJetEta;
    analysisBTagJetDiscriminator_ = info.btagJetDiscriminator;
    analysisNAK4_ = info.nAK4; analysisNAK8_ = info.nAK8;
    analysisNHeavyAK8_ = info.nHeavyAK8; analysisNBTags_ = info.nBTags;
    passesSignalRegion_.assign(configId_.size(), 0);
    passesRecoJetVeto_.assign(configId_.size(), 1);
    sj1NCA4E300_.assign(configId_.size(), 0);
    sj2NCA4E300_.assign(configId_.size(), 0);
    sj1NCA4E50_.assign(configId_.size(), 0);
    sj2NCA4E50_.assign(configId_.size(), 0);
    sj1MassE100_.assign(configId_.size(), kInvalidMass);
    sj2MassE100_.assign(configId_.size(), kInvalidMass);
    passesControlRegion_.assign(configId_.size(), 0);
    passesAT0b_.assign(configId_.size(), 0);
    passesAT1b_.assign(configId_.size(), 0);
    AssignmentResult reference;
    try {
      reference = reconstructReference(info.referenceConstituents);
    } catch (const fastjet::Error&) {
      reference = failure(RecoStatus::numericalFailure);
    }
    referenceRecoStatus_ = static_cast<std::uint8_t>(reference.status);
    referenceSJ1Mass_ = reference.leadingPtMass;
    referenceSJ2Mass_ = reference.subleadingPtMass;
    referenceSuuMass_ = reference.suuMass;
    referenceSJ1NCA4E50_ = reference.leadingNCA4E50;
    referenceSJ2NCA4E50_ = reference.subleadingNCA4E50;
    referenceSJ1NCA4E300_ = reference.leadingNCA4E300;
    referenceSJ2NCA4E300_ = reference.subleadingNCA4E300;
    referenceSJ1MassE100_ = reference.leadingMassE100;
    referenceSJ2MassE100_ = reference.subleadingMassE100;
    referenceRegion_ = an2017::region(passesBaseline_, reference.status == RecoStatus::valid,
        true, analysisNBTags_, reference.leadingNCA4E300, reference.subleadingNCA4E300,
        reference.leadingNCA4E50, reference.subleadingNCA4E50,
        reference.leadingMassE100, reference.subleadingMassE100);
  }

  const auto &candidates = input.get(packedPFToken_);
  try {
    auto particles = makeWeightedParticles(candidates);
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
      const bool recoJetVeto = std::all_of(akJets.begin(), akJets.begin() + nSelected,
                                         [](const auto &jet) { return jet.passesVeto; });
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
        suuMass_[configIndex] = assignment.suuMass;
        if (analysisMode_) {
          sj1NCA4E300_[configIndex] = assignment.leadingNCA4E300;
          sj2NCA4E300_[configIndex] = assignment.subleadingNCA4E300;
          sj1NCA4E50_[configIndex] = assignment.leadingNCA4E50;
          sj2NCA4E50_[configIndex] = assignment.subleadingNCA4E50;
          sj1MassE100_[configIndex] = assignment.leadingMassE100;
          sj2MassE100_[configIndex] = assignment.subleadingMassE100;
          passesRecoJetVeto_[configIndex] = recoJetVeto;
          const auto region = an2017::region(passesBaseline_, assignment.status == RecoStatus::valid,
              recoJetVeto, analysisNBTags_, assignment.leadingNCA4E300, assignment.subleadingNCA4E300,
              assignment.leadingNCA4E50, assignment.subleadingNCA4E50,
              assignment.leadingMassE100, assignment.subleadingMassE100);
          passesSignalRegion_[configIndex] = region == an2017::SR;
          passesControlRegion_[configIndex] = region == an2017::CR;
          passesAT1b_[configIndex] = region == an2017::AT1b;
          passesAT0b_[configIndex] = region == an2017::AT0b;
        }
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
  description.add<bool>("analysisMode", false);
  description.add<std::string>("sampleKind", "signal");
  edm::ParameterSetDescription analysisDescription;
  analysisDescription.setAllowAnything();
  description.add<edm::ParameterSetDescription>("analysis", analysisDescription);
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
