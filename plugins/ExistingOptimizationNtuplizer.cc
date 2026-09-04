// Record all relevant event information to a root file.

#include <memory>
#include <vector>
#include <unordered_map>
#include <set>
#include <cmath>
#include <string>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <limits>
#include <stdexcept>
#include "TLorentzVector.h"
#include "TVector3.h"

// Framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

// Data formats
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/MET.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/Jet.h"

// ROOT
#include "TTree.h"

// TFileService
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// Clustering
#include "fastjet/PseudoJet.hh"
#include "fastjet/ClusterSequence.hh"

//--------------------------------------------------
// Helper function: find last copy of a particle
//--------------------------------------------------

const reco::Candidate* findLastCopy(const reco::Candidate* p)
{
  const reco::Candidate* current = p;

  bool foundSameParticle = true;

  while (foundSameParticle) {

    foundSameParticle = false;

    for (unsigned int i=0; i<current->numberOfDaughters(); i++) {

      const reco::Candidate* d = current->daughter(i);

      if (d->pdgId() == current->pdgId()) {

        current = d;
        foundSameParticle = true;
        break;
      }
    }
  }

  return current;
}

//--------------------------------------------------
// Helper struct: Store 4-vec and boost vec of a particle
//--------------------------------------------------

struct ParticleKinematics {
    float px = 0;
    float py = 0;
    float pz = 0;
    float E = 0;
    float pt = 0;
    float eta = 0;
    float phi = 0;
    float mass = 0;
};

//--------------------------------------------------
// Helper function: fill 4-vec and boost vec of a particle
//--------------------------------------------------

void fillParticle(const reco::Candidate* src,
                  ParticleKinematics& dst)
{
    if (!src) return;

    dst.px   = src->px();
    dst.py   = src->py();
    dst.pz   = src->pz();
    dst.E    = src->energy();

    dst.pt   = src->pt();
    dst.eta  = src->eta();
    dst.phi  = src->phi();
    dst.mass = src->mass();
}

//--------------------------------------------------
// Helper function: branch particle 4-vec and boost vec in ttree
//--------------------------------------------------

void branchParticle(TTree* tree,
                    const std::string& name,
                    ParticleKinematics& p)
{
    tree->Branch((name+"_px").c_str(),   &p.px);
    tree->Branch((name+"_py").c_str(),   &p.py);
    tree->Branch((name+"_pz").c_str(),   &p.pz);
    tree->Branch((name+"_E").c_str(),    &p.E);

    tree->Branch((name+"_pt").c_str(),   &p.pt);
    tree->Branch((name+"_eta").c_str(),  &p.eta);
    tree->Branch((name+"_phi").c_str(),  &p.phi);
    tree->Branch((name+"_mass").c_str(), &p.mass);
}

//--------------------------------------------------
// Helper function: recursively find the truth label for a genParticle's origin
//--------------------------------------------------

std::set<int> getTruthLabel(const reco::Candidate* p,

                              const reco::Candidate* gen_W_q0_ptr[2],
                              const reco::Candidate* gen_W_q1_ptr[2],
                              const reco::Candidate* gen_b_ptr[2],
                              const reco::Candidate* gen_Z_q0_ptr[2],
                              const reco::Candidate* gen_Z_q1_ptr[2],
                              const reco::Candidate* gen_h_q0_ptr[2],
                              const reco::Candidate* gen_h_q1_ptr[2],
                              const reco::Candidate* gen_t_W_q0_ptr[2],
                              const reco::Candidate* gen_t_W_q1_ptr[2],
                              const reco::Candidate* gen_t_b_ptr[2])
{
    std::set<int> labels;

    // If no pointer, return empty
    if (!p) {
        return labels;
    }

    // Check both chi decay chains
    for (int i = 0; i < 2; ++i) {

        int offset = 10 * i;

        if (p == gen_W_q0_ptr[i])   {labels.insert(offset + 1);}
        if (p == gen_W_q1_ptr[i])   {labels.insert(offset + 2);}
        if (p == gen_b_ptr[i])      {labels.insert(offset + 3);}
        if (p == gen_Z_q0_ptr[i])   {labels.insert(offset + 4);}
        if (p == gen_Z_q1_ptr[i])   {labels.insert(offset + 5);}
        if (p == gen_h_q0_ptr[i])   {labels.insert(offset + 6);}
        if (p == gen_h_q1_ptr[i])   {labels.insert(offset + 7);}
        if (p == gen_t_W_q0_ptr[i]) {labels.insert(offset + 8);}
        if (p == gen_t_W_q1_ptr[i]) {labels.insert(offset + 9);}
        if (p == gen_t_b_ptr[i])    {labels.insert(offset + 10);}

        // Check last copies
        const reco::Candidate* p_lastcopy = findLastCopy(p);
        
        if (p_lastcopy == gen_W_q0_ptr[i])   {labels.insert(offset + 1);}
        if (p_lastcopy == gen_W_q1_ptr[i])   {labels.insert(offset + 2);}
        if (p_lastcopy == gen_b_ptr[i])      {labels.insert(offset + 3);}
        if (p_lastcopy == gen_Z_q0_ptr[i])   {labels.insert(offset + 4);}
        if (p_lastcopy == gen_Z_q1_ptr[i])   {labels.insert(offset + 5);}
        if (p_lastcopy == gen_h_q0_ptr[i])   {labels.insert(offset + 6);}
        if (p_lastcopy == gen_h_q1_ptr[i])   {labels.insert(offset + 7);}
        if (p_lastcopy == gen_t_W_q0_ptr[i]) {labels.insert(offset + 8);}
        if (p_lastcopy == gen_t_W_q1_ptr[i]) {labels.insert(offset + 9);}
        if (p_lastcopy == gen_t_b_ptr[i])    {labels.insert(offset + 10);}
    }

    // Recursively search all mothers
    for (size_t i = 0; i < p->numberOfMothers(); ++i) {

        std::set<int> mother_labels = getTruthLabel(
            p->mother(i),

            gen_W_q0_ptr,
            gen_W_q1_ptr,
            gen_b_ptr,
            gen_Z_q0_ptr,
            gen_Z_q1_ptr,
            gen_h_q0_ptr,
            gen_h_q1_ptr,
            gen_t_W_q0_ptr,
            gen_t_W_q1_ptr,
            gen_t_b_ptr
        );

        for (int mother_label : mother_labels) {
          labels.insert(mother_label);
        }
    }

    return labels;
}

//--------------------------------------------------
// Helper function: calculate delta phi
//--------------------------------------------------

float delta_Phi(float phi1, float phi2) {
  float dphi = phi1 - phi2;
  while (dphi > M_PI) dphi -= 2*M_PI;
  while (dphi <= -M_PI) dphi += 2*M_PI;
  return dphi;
}

//--------------------------------------------------
// Helper function: wrap phi bins
//--------------------------------------------------

int wrapPhiBin(int bin, int nphi) {
    bin %= nphi;
    if (bin < 0)
      bin += nphi;
    return bin;
}

//--------------------------------------------------
// Helper struct: describe cell, with custom operator '=='
//--------------------------------------------------

struct Cell {
  int ieta;
  int iphi;

  bool operator==(const Cell &other) const {
    return ieta == other.ieta && iphi == other.iphi;
  }
};

//--------------------------------------------------
// Helper struct: hash for unordered map of cells
//--------------------------------------------------

struct CellHash {
  std::size_t operator()(const Cell &c) const {
    return std::hash<int>()(c.ieta) ^ (std::hash<int>()(c.iphi) << 1);
  }
};

//----------------------------------------------------------------------
// Old jet sorting algorithm
//----------------------------------------------------------------------

class oldJetSortingAlgorithm {

public:

  oldJetSortingAlgorithm(double jetPtCut, 
                         double akRadius, 
                         double caRadius, 
                         double cosThrust,
                         bool useSlimmedAK8): 
    jetPtCut_(jetPtCut), 
    akRadius_(akRadius), 
    caRadius_(caRadius), 
    cosThrust_(cosThrust),
    useSlimmedAK8_(useSlimmedAK8) 
    {};

  void run(const pat::PackedCandidateCollection& pfcands,
           const pat::JetCollection& ak8Jets);

  // Helper struct

  struct JetRecord {
    TLorentzVector p4;
    int label = -1;
    std::vector<int> pfIndices;
  };

  const std::vector<int>& labels() const {
    return labels_;
  }

  const std::vector<int>& akJet_indices() const {
    return akJet_indices_;
  }

  const std::vector<int>& caJet_indices() const {
    return caJet_indices_;
  }

  const std::vector<JetRecord>& akJets() const {
    return akJets_;
  }

  const std::vector<JetRecord>& caJets() const {
    return caJets_;
  }

  

private:

  // Helper struct

  struct Constituent {
    int pfIndex;                // index in PackedCandidate collection
    int originalAK8;            // index of original AK8 jet
    TLorentzVector p4;          // four-vector (evolves!)
  };

  // Helper functions

  void clear();

  void collectAK8Constituents(const pat::PackedCandidateCollection& pfcands,
                              const pat::JetCollection& ak8Jets);

  void collectAKConstituents(const pat::PackedCandidateCollection& pfcands);

  void computeMPPBoost();

  void boostConstituents();

  void reclusterCA();

  void findThrustAxis();

  void assignCAToChiUsingCos();

  void assignCAToChiUsingMassDiff();

  void enforceChiPtOrdering();

  void assignParticleIndices();

  // Stored state

  std::vector<Constituent> constituents_;

  std::vector<JetRecord> akJets_;

  std::vector<JetRecord> caJets_;

  std::vector<int> labels_;

  std::vector<int> akJet_indices_;

  std::vector<int> caJet_indices_;

  TLorentzVector totalP4_;

  TVector3 betaMPP_;

  TVector3 thrustAxis_;

  std::vector<fastjet::PseudoJet> akfjParticles_;

  std::vector<fastjet::PseudoJet> cafjParticles_;

  double jetPtCut_;

  double akRadius_;

  double caRadius_;

  double cosThrust_;

  bool useSlimmedAK8_;

};

// clear

void oldJetSortingAlgorithm::clear() {

  constituents_.clear();

  akJets_.clear();
  caJets_.clear();

  labels_.clear();
  akJet_indices_.clear();
  caJet_indices_.clear();

  akfjParticles_.clear();
  cafjParticles_.clear();

  totalP4_.SetPxPyPzE(0., 0., 0., 0.);
  betaMPP_.SetXYZ(0., 0., 0.);
  thrustAxis_.SetXYZ(0., 0., 0.);

}

// run

void oldJetSortingAlgorithm::run(
  const pat::PackedCandidateCollection& pfcands,
  const pat::JetCollection& ak8Jets) {

  clear();

  labels_.assign(pfcands.size(), 0);
  akJet_indices_.assign(pfcands.size(), -1);
  caJet_indices_.assign(pfcands.size(), -1);
  constituents_.reserve(pfcands.size());

  if (useSlimmedAK8_) {collectAK8Constituents(pfcands, ak8Jets);}

  else {collectAKConstituents(pfcands);}

  computeMPPBoost();

  boostConstituents();

  reclusterCA();

  findThrustAxis();

  assignCAToChiUsingCos();

  assignCAToChiUsingMassDiff();

  enforceChiPtOrdering();

  assignParticleIndices();

}

// collectAK8Constituents

void oldJetSortingAlgorithm::collectAK8Constituents(
  const pat::PackedCandidateCollection& pfcands,
  const pat::JetCollection& ak8Jets) {

  akJets_.reserve(ak8Jets.size());
    
  for (size_t jetIdx = 0; jetIdx < ak8Jets.size(); ++jetIdx) { // Loop over fatjets

    const auto& jet = ak8Jets[jetIdx];

    if (jet.pt() < jetPtCut_) {continue;}

    JetRecord record;
    record.p4.SetPxPyPzE(
        jet.px(),
        jet.py(),
        jet.pz(),
        jet.energy()
    );

    record.pfIndices.reserve(jet.daughterPtrVector().size());

    for (const auto& daughter : jet.daughterPtrVector()) { // Loop over fatjet constituents

      // PackedCandidate index in the original collection
      const int pfIndex = daughter.key();

      if (pfIndex < 0 || pfIndex >= static_cast<int>(pfcands.size())){
        continue;
      }

      record.pfIndices.push_back(pfIndex);
      akJet_indices_[pfIndex] = static_cast<int>(jetIdx);

      const auto& pfcand = pfcands[pfIndex];

      Constituent c;
      c.pfIndex = pfIndex;
      c.originalAK8 = static_cast<int>(jetIdx);
      c.p4.SetPxPyPzE(
        pfcand.px(),
        pfcand.py(),
        pfcand.pz(),
        pfcand.energy()
      );

      constituents_.push_back(c);
    }

    akJets_.push_back(std::move(record));
  }

}

// collectAKConstituents

void oldJetSortingAlgorithm::collectAKConstituents(
    const pat::PackedCandidateCollection& pfcands) {

  // Convert PF candidates into FastJet pseudojets
  akfjParticles_.clear();

  for (size_t i = 0; i < pfcands.size(); ++i) {

    const auto& pfcand = pfcands[i];

    fastjet::PseudoJet pj(
      pfcand.px(),
      pfcand.py(),
      pfcand.pz(),
      pfcand.energy()
    );

    // Preserve the original PackedCandidate index
    pj.set_user_index(static_cast<int>(i));

    akfjParticles_.push_back(pj);
  }

  // Run anti-kT clustering

  fastjet::JetDefinition jetDef(fastjet::antikt_algorithm, akRadius_);

  fastjet::ClusterSequence clusterSequence(akfjParticles_, jetDef);

  const auto clusteredJets = fastjet::sorted_by_pt(clusterSequence.inclusive_jets(jetPtCut_));

  akJets_.reserve(clusteredJets.size());

  for (size_t jetIdx = 0; jetIdx < clusteredJets.size(); ++jetIdx) {

    const auto& fjJet = clusteredJets[jetIdx];

    if (fjJet.pt() < jetPtCut_) {continue;}

    JetRecord record;
    record.p4.SetPxPyPzE(
        fjJet.px(),
        fjJet.py(),
        fjJet.pz(),
        fjJet.e()
    );

    // This must happen while clusterSequence is alive.
    const auto fjConstituents = fjJet.constituents();
    record.pfIndices.reserve(fjConstituents.size());

    for (const auto& fjConstituent : fjConstituents) {

      const int pfIndex = fjConstituent.user_index();

      if (pfIndex < 0 || pfIndex >= static_cast<int>(pfcands.size())) {
        continue;
      }

      record.pfIndices.push_back(pfIndex);
      akJet_indices_[pfIndex] = static_cast<int>(jetIdx);

      const auto& pfcand = pfcands[pfIndex];

      Constituent c;
      c.pfIndex = pfIndex;
      c.originalAK8 = static_cast<int>(jetIdx);
      c.p4.SetPxPyPzE(
          pfcand.px(),
          pfcand.py(),
          pfcand.pz(),
          pfcand.energy()
      );

      constituents_.push_back(c);
    }

    akJets_.push_back(std::move(record));
  }

}

// computeMPPBoost

void oldJetSortingAlgorithm::computeMPPBoost() {

    totalP4_.SetPxPyPzE(0., 0., 0., 0.);

    for (const auto& constituent : constituents_) {
        totalP4_ += constituent.p4;
    }

    betaMPP_ = totalP4_.BoostVector();

}

// boostConstituents

void oldJetSortingAlgorithm::boostConstituents() {

  for (auto& constituent : constituents_) {
    constituent.p4.Boost(-betaMPP_);
  }

}

// reclusterCA

void oldJetSortingAlgorithm::reclusterCA() {

  cafjParticles_.clear();
  caJets_.clear();

  // Convert constituents into FastJet pseudojets

  for (const auto& constituent : constituents_) {

    fastjet::PseudoJet pj(
      constituent.p4.Px(),
      constituent.p4.Py(),
      constituent.p4.Pz(),
      constituent.p4.E());

    // Preserve the original PackedCandidate index
    pj.set_user_index(constituent.pfIndex);

    cafjParticles_.push_back(pj);

  }

  // Run CA reclustering

  fastjet::JetDefinition jetDef(fastjet::cambridge_algorithm, caRadius_);

  fastjet::ClusterSequence clusterSequence(cafjParticles_, jetDef);

  std::vector<fastjet::PseudoJet> caJets_tmp = fastjet::sorted_by_pt(clusterSequence.inclusive_jets());

  for (const auto& caJet_tmp : caJets_tmp) {

    JetRecord record;

    record.p4.SetPxPyPzE(
        caJet_tmp.px(),
        caJet_tmp.py(),
        caJet_tmp.pz(),
        caJet_tmp.e()
    );

    // Extract while clusterSequence is alive.
    for (const auto& constituent : caJet_tmp.constituents()) {
      record.pfIndices.push_back(constituent.user_index());
    }

    caJets_.push_back(std::move(record));
  }

}

// findThrustAxis

void oldJetSortingAlgorithm::findThrustAxis() {

  if (constituents_.empty()) {
    thrustAxis_.SetXYZ(0,0,0);
    return;
  }

  // Initial guess: direction of hardest constituent
  auto maxIt = std::max_element(
    constituents_.begin(),
    constituents_.end(),
    [](const Constituent& a, const Constituent& b) {
      return a.p4.P() < b.p4.P();
    }
  );

  TVector3 axis = maxIt->p4.Vect().Unit();

  // Numerically solve
  const int maxIterations = 100;
  const double tolerance = 1e-6;

  for (int iter = 0; iter < maxIterations; ++iter) {

    TVector3 newAxis(0,0,0);

    // Construct signed momentum sum
    for (const auto& c : constituents_) {

      TVector3 p = c.p4.Vect();

      if (p.Dot(axis) >= 0)
        newAxis += p;
      else
        newAxis -= p;
    }

    if (newAxis.Mag2() == 0.) {
      break;
    }

    newAxis = newAxis.Unit();

    // Check convergence
    if ((newAxis - axis).Mag() < tolerance) {
      axis = newAxis;
      break;
    }

    axis = newAxis;
  }

  thrustAxis_ = axis;

}

// assignCAToChiUsingCos

void oldJetSortingAlgorithm::assignCAToChiUsingCos() {

  for (size_t i = 0; i < caJets_.size(); i++) {

    TVector3 jetDir(caJets_[i].p4.Px(), caJets_[i].p4.Py(), caJets_[i].p4.Pz());

    jetDir = jetDir.Unit();

    double cosChi = jetDir.Dot(thrustAxis_);

    if (cosChi > cosThrust_) {
      caJets_[i].label = 1;
    }
    else if (cosChi < -cosThrust_) {
      caJets_[i].label = 2;
    }
    else {
      caJets_[i].label = 0; // Leave it for assignCAToChiUsingMassDiff()
    }
  }

}

// assignCAToChiUsingMassDiff

void oldJetSortingAlgorithm::assignCAToChiUsingMassDiff() {

  if (caJets_.empty()) {
    return;
  }

  TLorentzVector fixedChi0;
  TLorentzVector fixedChi1;

  std::vector<size_t> ambiguousIndices;
  ambiguousIndices.reserve(caJets_.size());

  // Sum already-fixed 4vecs
  for (size_t i = 0; i < caJets_.size(); ++i) {

    if (caJets_[i].label == 1) {
      fixedChi0 += caJets_[i].p4;
    }
    else if (caJets_[i].label == 2) {
      fixedChi1 += caJets_[i].p4;
    }
    else {
      ambiguousIndices.push_back(i);
    }
  }

  const size_t nAmbiguous = ambiguousIndices.size();

  // if none ambiguous, we're done

  if (nAmbiguous == 0) {
    return;
  }

  // Protect the bit shift

  if (nAmbiguous >= std::numeric_limits<size_t>::digits) {
    throw std::runtime_error("Too many ambiguous CA jets for exhaustive mass assignment");
  }

  const size_t nMasks = size_t{1} << nAmbiguous;

  double bestScore = std::numeric_limits<double>::max();
  size_t bestMask = 0;
  bool foundValidAssignment = false;

  // Test only the 2^(number of ambiguous jets) possible assignments

  for (size_t mask = 0; mask < nMasks; ++mask) {

    TLorentzVector chi0 = fixedChi0;
    TLorentzVector chi1 = fixedChi1;

    for (size_t k = 0; k < nAmbiguous; ++k) {

      const size_t jetIndex = ambiguousIndices[k];

      if ((mask >> k) & size_t{1}) {
        chi1 += caJets_[jetIndex].p4;
      }
      else {
        chi0 += caJets_[jetIndex].p4;
      }
    }

    // Require both reconstructed chis to contain energy

    if (chi0.E() <= 0. || chi1.E() <= 0.) {
      continue;
    }

    const double m0 = chi0.M();
    const double m1 = chi1.M();
    const double minMass = std::min(m0, m1);

    if (minMass <= 0.) {
      continue;
    }

    const double score = std::abs(m0 - m1) / minMass;

    if (score < bestScore) {
      bestScore = score;
      bestMask = mask;
      foundValidAssignment = true;
    }
  }

  // Do nothing if no valid assignment

  if (!foundValidAssignment) {
    return;
  }

  // Apply the best assignment only to ambiguous jets

  for (size_t k = 0; k < nAmbiguous; ++k) {

    const size_t jetIndex = ambiguousIndices[k];
    const bool assignToChi1 = ((bestMask >> k) & size_t{1}) != 0;

    caJets_[jetIndex].label = assignToChi1 ? 2 : 1;
  }

}

// enforceChiPtOrdering

void oldJetSortingAlgorithm::enforceChiPtOrdering() {

  TLorentzVector chi0;
  TLorentzVector chi1;

  for (const auto& jet : caJets_) {

    if (jet.label == 1)
      chi0 += jet.p4;

    else if (jet.label == 2)
      chi1 += jet.p4;

  }

  chi0.Boost(betaMPP_);
  chi1.Boost(betaMPP_);

  // Already in desired convention
  if (chi0.Pt() >= chi1.Pt())
    return;
  
  // Swap chi labels
  for (auto& jet : caJets_) {

    if (jet.label == 1)
      jet.label = 2;

    else if (jet.label == 2)
      jet.label = 1;

  }

}

void oldJetSortingAlgorithm::assignParticleIndices() {

  std::fill(labels_.begin(), labels_.end(), 0);
  std::fill(caJet_indices_.begin(), caJet_indices_.end(), -1);

  for (size_t i = 0; i < caJets_.size(); ++i) {

    const JetRecord& jet = caJets_[i];

    for (int pfIndex : jet.pfIndices) {

      if (pfIndex >= 0 && pfIndex < static_cast<int>(labels_.size())) {
        labels_[pfIndex] = jet.label;
        caJet_indices_[pfIndex] = static_cast<int>(i);
      }

    }

  }

}

//--------------------------------------------------
// Ntuplizer class declaration
//--------------------------------------------------

class ExistingOptimizationNtuplizer: public edm::one::EDAnalyzer<edm::one::SharedResources>{
  public:
    explicit ExistingOptimizationNtuplizer(const edm::ParameterSet&);
    ~ExistingOptimizationNtuplizer() override {}

    static void fillDescriptions(edm::ConfigurationDescriptions&);

  private:
    void beginJob() override;
    void analyze(const edm::Event&, const edm::EventSetup&) override;
    void printEventDebug();
    void printMotherChain(const reco::Candidate* p);
    void printAncestryGraph(const reco::Candidate* p,const reco::Candidate* b0,const reco::Candidate* b1,int depth = 0,std::set<const reco::Candidate*>* visited = nullptr);
    

  //------------------------------------
  // Tokens
  //------------------------------------

  edm::EDGetTokenT<pat::PackedCandidateCollection> packedPFToken_;
  edm::EDGetTokenT<pat::JetCollection> ak4Token_;
  edm::EDGetTokenT<pat::JetCollection> ak8Token_;
  edm::EDGetTokenT<pat::METCollection> metToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  edm::EDGetTokenT<double> rhoToken_;
  edm::EDGetTokenT<std::vector<reco::GenParticle>> genToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genAK4JetToken_;
  edm::EDGetTokenT<reco::GenJetCollection> genAK8JetToken_;

  //------------------------------------
  // TTree
  //------------------------------------

  TTree* tree_;
  std::vector<float> particle_pt;
  std::vector<float> particle_eta;
  std::vector<float> particle_phi;
  std::vector<float> particle_energy;
  std::vector<float> particle_mass;
  
  std::vector<float> particle_px;
  std::vector<float> particle_py;
  std::vector<float> particle_pz;
  
  std::vector<int> particle_charge;
  std::vector<int> particle_pdgId;
  
  std::vector<float> particle_dz;
  std::vector<float> particle_dxy;
  
  std::vector<float> particle_puppiWeight;
  std::vector<int> particle_fromPV;
  std::vector<int> particle_pvAssociationQuality;

  std::vector<float> particle_vx;
  std::vector<float> particle_vy;
  std::vector<float> particle_vz;

  std::vector<int> particle_oldAlgoLabel;
  std::vector<int> particle_oldAlgoAK4Label;
  std::vector<int> particle_oldAlgoAK8Index;
  std::vector<int> particle_oldAlgoCA8Index;
  
  std::vector<int> particle_newAlgoLabel;
  std::vector<int> particle_newAlgoAKIndex;
  std::vector<int> particle_newAlgoCAIndex;

  std::vector<float> ak_pt;
  std::vector<float> ak_eta;
  std::vector<float> ak_phi;
  std::vector<float> ak_mass;

  std::vector<float> ak8_pt;
  std::vector<float> ak8_eta;
  std::vector<float> ak8_phi;
  std::vector<float> ak8_mass;

  std::vector<float> ak8_softdropMass;

  std::vector<float> ak8_tau1;
  std::vector<float> ak8_tau2;
  std::vector<float> ak8_tau3;
  std::vector<float> ak8_tau4;

  std::vector<float> ak4_pt;
  std::vector<float> ak4_eta;
  std::vector<float> ak4_phi;
  std::vector<float> ak4_mass;

  std::vector<float> ak4_jetArea;

  ParticleKinematics gen_Suu;
  ParticleKinematics gen_chi[2];

  ParticleKinematics gen_W[2];
  ParticleKinematics gen_W_q0[2];
  ParticleKinematics gen_W_q1[2];
  ParticleKinematics gen_b[2];

  ParticleKinematics gen_Z[2];
  ParticleKinematics gen_Z_q0[2];
  ParticleKinematics gen_Z_q1[2];

  ParticleKinematics gen_h[2];
  ParticleKinematics gen_h_q0[2];
  ParticleKinematics gen_h_q1[2];

  ParticleKinematics gen_t[2];
  ParticleKinematics gen_t_W[2];
  ParticleKinematics gen_t_W_q0[2];
  ParticleKinematics gen_t_W_q1[2];
  ParticleKinematics gen_t_b[2];

  UInt_t run;
  UInt_t lumi;
  ULong64_t event;
  float HT;
  float MET_pt;
  float MET_phi;
  float rho;
  int nPV;
  unsigned int nParticles;
  float PV_x;
  float PV_y;
  float PV_z;
  bool isMC_;
  bool passesEventSelection_;
  double jetPtCut_;
  double eventJetPtCut_;
  unsigned int minEventJets_;
  double akRadius_;
  double caRadius_;
  double cosThrust_;

  std::vector<int> particle_match_pdgid;
  std::vector<float> particle_match_dr2;
  std::vector<std::vector<int>> particle_truthLabel;
};

//------------------------------------
// Constructor
//------------------------------------

ExistingOptimizationNtuplizer::ExistingOptimizationNtuplizer(
    const edm::ParameterSet& iConfig)
{
    packedPFToken_ = consumes<pat::PackedCandidateCollection>(
        iConfig.getParameter<edm::InputTag>("packedPFCandidates"));

    ak4Token_ = consumes<pat::JetCollection>(
        iConfig.getParameter<edm::InputTag>("ak4Jets"));

    ak8Token_ = consumes<pat::JetCollection>(
        iConfig.getParameter<edm::InputTag>("ak8Jets"));

    metToken_ = consumes<pat::METCollection>(
        iConfig.getParameter<edm::InputTag>("met"));

    vertexToken_ = consumes<reco::VertexCollection>(
        iConfig.getParameter<edm::InputTag>("vertices"));

    rhoToken_ = consumes<double>(
        iConfig.getParameter<edm::InputTag>("rho"));

    genToken_ = consumes<reco::GenParticleCollection>(
        iConfig.getParameter<edm::InputTag>("genParticles"));

    genAK4JetToken_ = consumes<reco::GenJetCollection>(
        iConfig.getParameter<edm::InputTag>("genJets"));

    genAK8JetToken_ = consumes<reco::GenJetCollection>(
        iConfig.getParameter<edm::InputTag>("genAK8Jets"));

    isMC_ = iConfig.getParameter<bool>("isMC");

    jetPtCut_ = iConfig.getParameter<double>("jetPtCut");

    eventJetPtCut_ = iConfig.getParameter<double>("eventJetPtCut");

    minEventJets_ = iConfig.getParameter<unsigned int>("minEventJets");

    akRadius_ = iConfig.getParameter<double>("akRadius");

    caRadius_ = iConfig.getParameter<double>("caRadius");

    cosThrust_ = iConfig.getParameter<double>("cosThrust");
}

//------------------------------------
// printEventDebug
//------------------------------------

void ExistingOptimizationNtuplizer::printEventDebug() {

  std::cout << "\n================ EVENT DEBUG ================\n";

  // Event info
  std::cout << "Run/Lumi/Event: "
            << run << " / "
            << lumi << " / "
            << event << "\n";

  std::cout << "isMC: " << isMC_ << "\n";
  std::cout << "jetPtCut:  " << jetPtCut_  << "\n";
  std::cout << "eventJetPtCut: " << eventJetPtCut_
            << " minEventJets: " << minEventJets_ << "\n";
  std::cout << "akRadius:  " << akRadius_  << "\n";
  std::cout << "caRadius:  " << caRadius_  << "\n";
  std::cout << "cosThrust: " << cosThrust_ << "\n";
  std::cout << "HT: " << HT
            << " MET_pt: " << MET_pt
            << " MET_phi: " << MET_phi
            << " rho: " << rho << "\n";

  std::cout << "nPV: " << nPV
            << " PV: ("
            << PV_x << ", "
            << PV_y << ", "
            << PV_z << ")\n";

  std::cout << "nParticles: " << nParticles << "\n";


  // PF particles
  std::cout << "\n---------- PF PARTICLES ----------\n";

  for (unsigned int i = 0; i < particle_pt.size(); i++) {

    std::cout << "Particle " << i << ": "
              << "pt=" << particle_pt[i]
              << " eta=" << particle_eta[i]
              << " phi=" << particle_phi[i]
              << " E=" << particle_energy[i]
              << " m=" << particle_mass[i]
              << " px=" << particle_px[i]
              << " py=" << particle_py[i]
              << " pz=" << particle_pz[i]
              << "\n";

    std::cout << "  charge=" << particle_charge[i]
              << " pdgId=" << particle_pdgId[i]
              << " dz=" << particle_dz[i]
              << " dxy=" << particle_dxy[i]
              << "\n";

    std::cout << "  puppiWeight=" << particle_puppiWeight[i]
              << " fromPV=" << particle_fromPV[i]
              << " pvQuality=" << particle_pvAssociationQuality[i]
              << "\n";

    std::cout << "  vertex=("
              << particle_vx[i] << ", "
              << particle_vy[i] << ", "
              << particle_vz[i] << ")"
              << " particle_oldAlgoLabel=" << particle_oldAlgoLabel[i]
              << " particle_oldAlgoAK4Label=" << particle_oldAlgoAK4Label[i]
              << " particle_oldAlgoAK8Index=" << particle_oldAlgoAK8Index[i]
              << " particle_oldAlgoCA8Index=" << particle_oldAlgoCA8Index[i]
              << "\n";

    std::cout << "  matchPDG=" << particle_match_pdgid[i]
              << " matchDR2=" << particle_match_dr2[i]
              << " particle_newAlgoLabel=" << particle_newAlgoLabel[i]
              << " particle_newAlgoAKIndex=" << particle_newAlgoAKIndex[i]
              << " particle_newAlgoCAIndex=" << particle_newAlgoCAIndex[i]
              << " truthLabels=";

    for (auto label : particle_truthLabel[i])
      std::cout << label << " ";

    std::cout << "\n";
  }


  // AK8 jets
  std::cout << "\n---------- AK8 JETS ----------\n";

  for (unsigned int i = 0; i < ak8_pt.size(); i++) {

    std::cout << "AK8 Jet " << i
              << ": pt=" << ak8_pt[i]
              << " eta=" << ak8_eta[i]
              << " phi=" << ak8_phi[i]
              << " mass=" << ak8_mass[i]
              << "\n";

    std::cout << "  softdropMass="
              << ak8_softdropMass[i]
              << " tau1=" << ak8_tau1[i]
              << " tau2=" << ak8_tau2[i]
              << " tau3=" << ak8_tau3[i]
              << " tau4=" << ak8_tau4[i]
              << "\n";
  }


  // AK4 jets
  std::cout << "\n---------- AK4 JETS ----------\n";

  for (unsigned int i = 0; i < ak4_pt.size(); i++) {

    std::cout << "AK4 Jet " << i
              << ": pt=" << ak4_pt[i]
              << " eta=" << ak4_eta[i]
              << " phi=" << ak4_phi[i]
              << " mass=" << ak4_mass[i]
              << " area=" << ak4_jetArea[i]
              << "\n";
  }


  // Generator truth
  auto printKin = [](const std::string& name,
                     const ParticleKinematics& p) {

    std::cout << name
              << ": pt=" << p.pt
              << " eta=" << p.eta
              << " phi=" << p.phi
              << " mass=" << p.mass
              << "\n";
  };


  std::cout << "\n---------- GEN TRUTH ----------\n";

  printKin("Suu", gen_Suu);

  for (int i = 0; i < 2; i++) {

    std::cout << "\nDecay chain " << i << "\n";

    printKin("chi", gen_chi[i]);

    printKin("W", gen_W[i]);
    printKin("W_q0", gen_W_q0[i]);
    printKin("W_q1", gen_W_q1[i]);
    printKin("b", gen_b[i]);

    printKin("Z", gen_Z[i]);
    printKin("Z_q0", gen_Z_q0[i]);
    printKin("Z_q1", gen_Z_q1[i]);

    printKin("h", gen_h[i]);
    printKin("h_q0", gen_h_q0[i]);
    printKin("h_q1", gen_h_q1[i]);

    printKin("t", gen_t[i]);
    printKin("t_W", gen_t_W[i]);
    printKin("t_W_q0", gen_t_W_q0[i]);
    printKin("t_W_q1", gen_t_W_q1[i]);
    printKin("t_b", gen_t_b[i]);
  }

  std::cout << "==============================================\n\n";
}

//------------------------------------
// beginJob
//------------------------------------

void ExistingOptimizationNtuplizer::beginJob(){
  usesResource("TFileService");

  edm::Service<TFileService> fs;

  tree_ = fs->make<TTree>("Events", "Events");

  // Event information
  tree_->Branch("run",   &run);
  tree_->Branch("lumi",  &lumi);
  tree_->Branch("event", &event);

  tree_->Branch("HT",      &HT);
  tree_->Branch("MET_pt",  &MET_pt);
  tree_->Branch("MET_phi", &MET_phi);
  tree_->Branch("rho",     &rho);
  tree_->Branch("nPV",     &nPV);
  tree_->Branch("PV_x",    &PV_x);
  tree_->Branch("PV_y",    &PV_y);
  tree_->Branch("PV_z",    &PV_z);
  tree_->Branch("nParticles", &nParticles);
  tree_->Branch("isMC",    &isMC_);
  tree_->Branch("passesEventSelection", &passesEventSelection_);
  tree_->Branch("jetPtCut",    &jetPtCut_);
  tree_->Branch("eventJetPtCut", &eventJetPtCut_);
  tree_->Branch("minEventJets", &minEventJets_);
  tree_->Branch("akRadius",    &akRadius_);
  tree_->Branch("caRadius",    &caRadius_);
  tree_->Branch("cosThrust",   &cosThrust_);

  // Particle information
  tree_->Branch("particle_pt",     &particle_pt);
  tree_->Branch("particle_eta",    &particle_eta);
  tree_->Branch("particle_phi",    &particle_phi);
  tree_->Branch("particle_energy", &particle_energy);
  tree_->Branch("particle_mass", &particle_mass);
  tree_->Branch("particle_px", &particle_px);
  tree_->Branch("particle_py", &particle_py);
  tree_->Branch("particle_pz", &particle_pz);

  tree_->Branch("particle_charge", &particle_charge);
  tree_->Branch("particle_pdgId",  &particle_pdgId);

  tree_->Branch("particle_dz",  &particle_dz);
  tree_->Branch("particle_dxy", &particle_dxy);

  tree_->Branch("particle_puppiWeight",          &particle_puppiWeight);
  tree_->Branch("particle_fromPV",               &particle_fromPV);
  tree_->Branch("particle_pvAssociationQuality", &particle_pvAssociationQuality);

  tree_->Branch("particle_vx", &particle_vx);
  tree_->Branch("particle_vy", &particle_vy);
  tree_->Branch("particle_vz", &particle_vz);

  tree_->Branch("particle_match_pdgid", &particle_match_pdgid);
  tree_->Branch("particle_match_dr2", &particle_match_dr2);
  tree_->Branch("particle_truthLabel", &particle_truthLabel);

  tree_->Branch("particle_oldAlgoLabel", &particle_oldAlgoLabel);
  tree_->Branch("particle_oldAlgoAK4Label", &particle_oldAlgoAK4Label);
  tree_->Branch("particle_oldAlgoAK8Index", &particle_oldAlgoAK8Index);
  tree_->Branch("particle_oldAlgoCA8Index", &particle_oldAlgoCA8Index);

  tree_->Branch("particle_newAlgoLabel", &particle_newAlgoLabel);
  tree_->Branch("particle_newAlgoAKIndex", &particle_newAlgoAKIndex);
  tree_->Branch("particle_newAlgoCAIndex", &particle_newAlgoCAIndex);

  // AK jets
  tree_->Branch("ak_pt",   &ak_pt);
  tree_->Branch("ak_eta",  &ak_eta);
  tree_->Branch("ak_phi",  &ak_phi);
  tree_->Branch("ak_mass", &ak_mass);

  // AK8 jets
  tree_->Branch("ak8_pt",   &ak8_pt);
  tree_->Branch("ak8_eta",  &ak8_eta);
  tree_->Branch("ak8_phi",  &ak8_phi);
  tree_->Branch("ak8_mass", &ak8_mass);

  tree_->Branch("ak8_softdropMass", &ak8_softdropMass);

  tree_->Branch("ak8_tau1", &ak8_tau1);
  tree_->Branch("ak8_tau2", &ak8_tau2);
  tree_->Branch("ak8_tau3", &ak8_tau3);
  tree_->Branch("ak8_tau4", &ak8_tau4);

  // AK4 jets
  tree_->Branch("ak4_pt",   &ak4_pt);
  tree_->Branch("ak4_eta",  &ak4_eta);
  tree_->Branch("ak4_phi",  &ak4_phi);
  tree_->Branch("ak4_mass", &ak4_mass);

  tree_->Branch("ak4_jetArea", &ak4_jetArea);

  // Target info
  branchParticle(tree_, "gen_Suu", gen_Suu);

  for (int i = 0; i < 2; ++i) {
    branchParticle(tree_, "gen_chi" + std::to_string(i), gen_chi[i]);

    branchParticle(tree_, "gen_W" + std::to_string(i), gen_W[i]);
    branchParticle(tree_, "gen_W" + std::to_string(i) + "_q0", gen_W_q0[i]);
    branchParticle(tree_, "gen_W" + std::to_string(i) + "_q1", gen_W_q1[i]);
    branchParticle(tree_, "gen_b" + std::to_string(i), gen_b[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i), gen_Z[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i) + "_q0", gen_Z_q0[i]);
    branchParticle(tree_, "gen_Z" + std::to_string(i) + "_q1", gen_Z_q1[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i), gen_h[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i) + "_q0", gen_h_q0[i]);
    branchParticle(tree_, "gen_h" + std::to_string(i) + "_q1", gen_h_q1[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i), gen_t[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W", gen_t_W[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W_q0", gen_t_W_q0[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_W_q1", gen_t_W_q1[i]);
    branchParticle(tree_, "gen_t" + std::to_string(i) + "_b", gen_t_b[i]);
  }
}

//------------------------------------
// Event loop
//------------------------------------

void ExistingOptimizationNtuplizer::analyze(const edm::Event& iEvent,
                          const edm::EventSetup&) {

  // Clear vectors
  particle_pt.clear();
  particle_eta.clear();
  particle_phi.clear();
  particle_energy.clear();
  particle_mass.clear();
  particle_px.clear();
  particle_py.clear();
  particle_pz.clear();
  
  particle_charge.clear();
  particle_pdgId.clear();
  
  particle_dz.clear();
  particle_dxy.clear();
  
  particle_puppiWeight.clear();
  particle_fromPV.clear();
  particle_pvAssociationQuality.clear();

  particle_vx.clear();
  particle_vy.clear();
  particle_vz.clear();

  particle_match_pdgid.clear();
  particle_match_dr2.clear();
  particle_truthLabel.clear();

  particle_oldAlgoLabel.clear();
  particle_oldAlgoAK4Label.clear();
  particle_oldAlgoAK8Index.clear();
  particle_oldAlgoCA8Index.clear();

  particle_newAlgoLabel.clear();
  particle_newAlgoAKIndex.clear();
  particle_newAlgoCAIndex.clear();

  ak_pt.clear();
  ak_eta.clear();
  ak_phi.clear();
  ak_mass.clear();

  ak8_pt.clear();
  ak8_eta.clear();
  ak8_phi.clear();
  ak8_mass.clear();

  ak8_softdropMass.clear();

  ak8_tau1.clear();
  ak8_tau2.clear();
  ak8_tau3.clear();
  ak8_tau4.clear();

  ak4_pt.clear();
  ak4_eta.clear();
  ak4_phi.clear();
  ak4_mass.clear();

  ak4_jetArea.clear();

  HT = 0;
  rho = 0;
  PV_x = PV_y = PV_z = 0;
  passesEventSelection_ = true;

  gen_Suu = {};

  for (int i = 0; i < 2; ++i) {
    gen_chi[i] = {};

    gen_W[i] = {};
    gen_W_q0[i] = {};
    gen_W_q1[i] = {};
    gen_b[i] = {};
    gen_Z[i] = {};
    gen_Z_q0[i] = {};
    gen_Z_q1[i] = {};
    gen_h[i] = {};
    gen_h_q0[i] = {};
    gen_h_q1[i] = {};
    gen_t[i] = {};
    gen_t_W[i] = {};
    gen_t_W_q0[i] = {};
    gen_t_W_q1[i] = {};
    gen_t_b[i] = {};
  }
  
  // Retrieve collections

  edm::Handle<reco::VertexCollection> vertices;
  iEvent.getByToken(vertexToken_, vertices);

  edm::Handle<pat::PackedCandidateCollection> packedPFCands;
  iEvent.getByToken(packedPFToken_, packedPFCands);

  edm::Handle<pat::JetCollection> ak4Jets;
  iEvent.getByToken(ak4Token_, ak4Jets);

  edm::Handle<pat::JetCollection> ak8Jets;
  iEvent.getByToken(ak8Token_, ak8Jets);

  edm::Handle<pat::METCollection> mets;
  iEvent.getByToken(metToken_, mets);

  edm::Handle<double> rhoHandle;
  iEvent.getByToken(rhoToken_, rhoHandle);

  edm::Handle<std::vector<reco::GenParticle>> genParticles;
  iEvent.getByToken(genToken_, genParticles);

  edm::Handle<reco::GenJetCollection> genAK4Jets;
  iEvent.getByToken(genAK4JetToken_, genAK4Jets);

  edm::Handle<reco::GenJetCollection> genAK8Jets;
  iEvent.getByToken(genAK8JetToken_, genAK8Jets);

  // Reserve vector space

  particle_pt.reserve(packedPFCands->size());
  particle_eta.reserve(packedPFCands->size());
  particle_phi.reserve(packedPFCands->size());
  particle_energy.reserve(packedPFCands->size());
  particle_mass.reserve(packedPFCands->size());
  particle_px.reserve(packedPFCands->size());
  particle_py.reserve(packedPFCands->size());
  particle_pz.reserve(packedPFCands->size());
  
  particle_charge.reserve(packedPFCands->size());
  particle_pdgId.reserve(packedPFCands->size());
  
  particle_dz.reserve(packedPFCands->size());
  particle_dxy.reserve(packedPFCands->size());
  
  particle_puppiWeight.reserve(packedPFCands->size());
  particle_fromPV.reserve(packedPFCands->size());
  particle_pvAssociationQuality.reserve(packedPFCands->size());

  particle_vx.reserve(packedPFCands->size());
  particle_vy.reserve(packedPFCands->size());
  particle_vz.reserve(packedPFCands->size());

  particle_oldAlgoLabel.reserve(packedPFCands->size());
  particle_oldAlgoAK4Label.reserve(packedPFCands->size());
  particle_oldAlgoAK8Index.reserve(packedPFCands->size());
  particle_oldAlgoCA8Index.reserve(packedPFCands->size());

  particle_newAlgoLabel.reserve(packedPFCands->size());
  particle_newAlgoAKIndex.reserve(packedPFCands->size());
  particle_newAlgoCAIndex.reserve(packedPFCands->size());

  ak8_pt.reserve(ak8Jets->size());
  ak8_eta.reserve(ak8Jets->size());
  ak8_phi.reserve(ak8Jets->size());
  ak8_mass.reserve(ak8Jets->size());

  ak8_softdropMass.reserve(ak8Jets->size());

  ak8_tau1.reserve(ak8Jets->size());
  ak8_tau2.reserve(ak8Jets->size());
  ak8_tau3.reserve(ak8Jets->size());
  ak8_tau4.reserve(ak8Jets->size());

  ak4_pt.reserve(ak4Jets->size());
  ak4_eta.reserve(ak4Jets->size());
  ak4_phi.reserve(ak4Jets->size());
  ak4_mass.reserve(ak4Jets->size());

  ak4_jetArea.reserve(ak4Jets->size());

  // Fill event information

  run = iEvent.id().run();
  lumi = iEvent.luminosityBlock();
  event = iEvent.id().event();
  nPV = vertices->size();

  // Determine AK8 jet membership for each PF candidate
  std::vector<int> ak8Membership(packedPFCands->size(), -1);

  for (size_t jetIdx = 0; jetIdx < ak8Jets->size(); ++jetIdx) {

    const auto& jet = (*ak8Jets)[jetIdx];

    for (const auto& daughter : jet.daughterPtrVector()) {

      if (daughter.key() < ak8Membership.size())
          ak8Membership[daughter.key()] = jetIdx;
    }
  }

  // Fill ak8 jet information
  for (const auto& jet : *ak8Jets) {

    ak8_pt.push_back(jet.pt());
    ak8_eta.push_back(jet.eta());
    ak8_phi.push_back(jet.phi());
    ak8_mass.push_back(jet.mass());

    if (jet.hasUserFloat("ak8PFJetsPuppiSoftDropMass")) {
      ak8_softdropMass.push_back(jet.userFloat("ak8PFJetsPuppiSoftDropMass"));
    }
    else {
      ak8_softdropMass.push_back(-1.);
    }

    ak8_tau1.push_back(jet.userFloat("NjettinessAK8Puppi:tau1"));
    ak8_tau2.push_back(jet.userFloat("NjettinessAK8Puppi:tau2"));
    ak8_tau3.push_back(jet.userFloat("NjettinessAK8Puppi:tau3"));
    ak8_tau4.push_back(jet.userFloat("NjettinessAK8Puppi:tau4"));

    // Only run if want to see all userFloats
    //for (const auto& name : jet.userFloatNames()) std::cout << name << std::endl;
    //for (const auto& pair : jet.getPairDiscri()) std::cout << pair.first << " : " << pair.second << std::endl;
    
  }

  // Determine AK4 jet membership for each PF candidate
  std::vector<int> ak4Membership(packedPFCands->size(), -1);

  for (size_t jetIdx = 0; jetIdx < ak4Jets->size(); ++jetIdx) {

    const auto& jet = (*ak4Jets)[jetIdx];

    for (const auto& daughter : jet.daughterPtrVector()) {

      if (daughter.key() < ak4Membership.size())
          ak4Membership[daughter.key()] = jetIdx;
    }
  }

  // Fill ak4 jet information
  HT = 0;

  for (const auto& jet : *ak4Jets){

    ak4_pt.push_back(jet.pt());
    ak4_eta.push_back(jet.eta());
    ak4_phi.push_back(jet.phi());
    ak4_mass.push_back(jet.mass());

    ak4_jetArea.push_back(jet.jetArea());

    HT += jet.pt();

    // Only run if want to see all userFloats
    //for (const auto& name : jet.userFloatNames()) std::cout << name << std::endl;
    //for (const auto& pair : jet.getPairDiscri()) std::cout << pair.first << std::endl;

  }

  // Fill particle information
  for (const auto& p : *packedPFCands) {

    particle_pt.push_back(p.pt());
    particle_eta.push_back(p.eta());
    particle_phi.push_back(p.phi());
    particle_energy.push_back(p.energy());
    particle_mass.push_back(p.mass());
    particle_px.push_back(p.px());
    particle_py.push_back(p.py());
    particle_pz.push_back(p.pz());
  
    particle_charge.push_back(p.charge());
    particle_pdgId.push_back(p.pdgId());
  
    particle_dz.push_back(p.dz());
    particle_dxy.push_back(p.dxy());
  
    particle_puppiWeight.push_back(p.puppiWeight());
    particle_fromPV.push_back(p.fromPV());
    particle_pvAssociationQuality.push_back(p.pvAssociationQuality());

    particle_vx.push_back(p.vx());
    particle_vy.push_back(p.vy());
    particle_vz.push_back(p.vz());
  }

  for (long unsigned int i = 0; i < packedPFCands->size(); ++i) {
    particle_oldAlgoAK4Label.push_back(ak4Membership[i]);
  } // ak8 membership done later

  nParticles = particle_pt.size();

  // Fill MET information
  if (!mets->empty()) {
    MET_pt = mets->front().pt();
    MET_phi = mets->front().phi();
  } else {
    MET_pt = 0.f;
    MET_phi = 0.f;
  }

  // Fill rho
  rho = *rhoHandle;

  // Fill PV location
  if (!vertices->empty()) {
    PV_x = vertices->front().x();
    PV_y = vertices->front().y();
    PV_z = vertices->front().z();
  } else {
    PV_x = PV_y = PV_z = 0.f;
  }

  // Find and fill target info
  // Loop over genParticles to find Suu
  // Record pointers to 1) record kinematic vars and 2) to match with PF candidates
  const reco::Candidate* gen_Suu_ptr = nullptr;
  const reco::Candidate* gen_chi_ptr[2] = {nullptr, nullptr};

  for (const auto &p : *genParticles) {

    if (std::abs(p.pdgId()) == 9936661) { // The Suu pdgID

      gen_Suu_ptr = findLastCopy(&p);
      fillParticle(gen_Suu_ptr, gen_Suu);

      // Find the ordering (by pt) of chis
      if (gen_Suu_ptr->numberOfDaughters() != 2) {
        continue;
      }

      if (std::abs(gen_Suu_ptr->daughter(0)->pdgId()) != 9936662 || std::abs(gen_Suu_ptr->daughter(1)->pdgId()) != 9936662) { // The chi pdgID
        continue;
      }

      const reco::Candidate* tmp_gen_chi_ptr_0 = findLastCopy(gen_Suu_ptr->daughter(0));
      const reco::Candidate* tmp_gen_chi_ptr_1 = findLastCopy(gen_Suu_ptr->daughter(1));

      if (tmp_gen_chi_ptr_0->pt() > tmp_gen_chi_ptr_1->pt()) {
        gen_chi_ptr[0] = tmp_gen_chi_ptr_0;
        gen_chi_ptr[1] = tmp_gen_chi_ptr_1;
      }

      else {
        gen_chi_ptr[1] = tmp_gen_chi_ptr_0;
        gen_chi_ptr[0] = tmp_gen_chi_ptr_1;
      }  
      
      fillParticle(gen_chi_ptr[0], gen_chi[0]);
      fillParticle(gen_chi_ptr[1], gen_chi[1]);

      break;
    }
  }

  // Loop over genParticles to find decays
  // Record pointers to 1) record kinematic vars and 2) to match with PF candidates
  const reco::Candidate* gen_W_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_W_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_W_q1_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_b_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_Z_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_Z_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_Z_q1_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_h_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_h_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_h_q1_ptr[2] = {nullptr, nullptr};

  const reco::Candidate* gen_t_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_q0_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_W_q1_ptr[2] = {nullptr, nullptr};
  const reco::Candidate* gen_t_b_ptr[2] = {nullptr, nullptr};

  for (int idx = 0; idx < 2; ++idx) { // for each chi

    const reco::Candidate* gen_chi_ptr_temp = gen_chi_ptr[idx];

    if (!gen_chi_ptr_temp) continue;
    
    for (unsigned int i=0; i<gen_chi_ptr_temp->numberOfDaughters(); i++) { // for each chi daughter

      const reco::Candidate* daughter = findLastCopy(gen_chi_ptr_temp->daughter(i));

      // determine if decay is Wb, ht, or Zt
      if (std::abs(daughter->pdgId()) == 24) { // if daughter is W

        gen_W_ptr[idx] = daughter;
        fillParticle(gen_W_ptr[idx], gen_W[idx]);

        if (gen_W_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_W_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_W_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_W_q0_ptr[idx] = findLastCopy(gen_W_ptr[idx]->daughter(0));
        gen_W_q1_ptr[idx] = findLastCopy(gen_W_ptr[idx]->daughter(1));

        fillParticle(gen_W_q0_ptr[idx], gen_W_q0[idx]);
        fillParticle(gen_W_q1_ptr[idx], gen_W_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 5) { // if daughter is b

        gen_b_ptr[idx] = daughter;
        fillParticle(gen_b_ptr[idx], gen_b[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 23) { // if daughter is Z

        gen_Z_ptr[idx] = daughter;
        fillParticle(gen_Z_ptr[idx], gen_Z[idx]);

        if (gen_Z_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_Z_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_Z_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_Z_q0_ptr[idx] = findLastCopy(gen_Z_ptr[idx]->daughter(0));
        gen_Z_q1_ptr[idx] = findLastCopy(gen_Z_ptr[idx]->daughter(1));

        fillParticle(gen_Z_q0_ptr[idx], gen_Z_q0[idx]);
        fillParticle(gen_Z_q1_ptr[idx], gen_Z_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 25) { // if daughter is h

        gen_h_ptr[idx] = daughter;
        fillParticle(gen_h_ptr[idx], gen_h[idx]);

        if (gen_h_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_h_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_h_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_h_q0_ptr[idx] = findLastCopy(gen_h_ptr[idx]->daughter(0));
        gen_h_q1_ptr[idx] = findLastCopy(gen_h_ptr[idx]->daughter(1));

        fillParticle(gen_h_q0_ptr[idx], gen_h_q0[idx]);
        fillParticle(gen_h_q1_ptr[idx], gen_h_q1[idx]);

      }

      else if (std::abs(daughter->pdgId()) == 6) { // if daughter is t

        gen_t_ptr[idx] = daughter;
        fillParticle(gen_t_ptr[idx], gen_t[idx]);

        if (gen_t_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_t_ptr[idx]->daughter(0)->pdgId()) == 24 && std::abs(gen_t_ptr[idx]->daughter(1)->pdgId()) == 5) {
          gen_t_W_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(0));
          gen_t_b_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(1));
        }

        else if (std::abs(gen_t_ptr[idx]->daughter(0)->pdgId()) == 5 && std::abs(gen_t_ptr[idx]->daughter(1)->pdgId()) == 24) {
          gen_t_W_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(1));
          gen_t_b_ptr[idx] = findLastCopy(gen_t_ptr[idx]->daughter(0));
        }

        else continue;

        fillParticle(gen_t_W_ptr[idx], gen_t_W[idx]);
        fillParticle(gen_t_b_ptr[idx], gen_t_b[idx]);

        if (gen_t_W_ptr[idx]->numberOfDaughters() != 2) continue;

        if (std::abs(gen_t_W_ptr[idx]->daughter(0)->pdgId()) > 5 ||
            std::abs(gen_t_W_ptr[idx]->daughter(1)->pdgId()) > 5) continue;

        gen_t_W_q0_ptr[idx] = findLastCopy(gen_t_W_ptr[idx]->daughter(0));
        gen_t_W_q1_ptr[idx] = findLastCopy(gen_t_W_ptr[idx]->daughter(1));

        fillParticle(gen_t_W_q0_ptr[idx], gen_t_W_q0[idx]);
        fillParticle(gen_t_W_q1_ptr[idx], gen_t_W_q1[idx]);

      }
    } // end loop over chi daughters
  } // end loop over chis

  // Find and fill truth information on each PF particle's ancestry, using delta R matching between PF cands and generator particles.
  // Build spatial hash map taking in eta, phi of PF candidate and outputting list of gen particles within 3x3 cell grid of that region.
  
  // Cell size constants, delta R^2 limit, testing values for optimal truth matching
  constexpr float ETA_BIN = 0.2;
  constexpr float PHI_BIN = 0.2;
  constexpr float DR2_MAX = ETA_BIN * PHI_BIN;
  int nphi = std::ceil(2 * M_PI / PHI_BIN);

  // Build eta-phi map for status=1 gen particles
  std::unordered_map<Cell, std::vector<const reco::GenParticle*>, CellHash> genGrid;

  for (const auto &gen : *genParticles) {

    if (gen.status() != 1) continue;

    float eta = gen.eta();
    float phi = gen.phi();

    Cell c;
    c.ieta = std::floor(eta / ETA_BIN);
    c.iphi = wrapPhiBin(std::floor(phi / PHI_BIN), nphi);

    genGrid[c].push_back(&gen);
  }

  // Loop over PF cands, search nearby bins
  for (size_t ipf = 0; ipf < packedPFCands->size(); ++ipf) {

    const pat::PackedCandidate &pf = (*packedPFCands)[ipf];

    float pf_eta = pf.eta();
    float pf_phi = pf.phi();

    Cell base;
    base.ieta = std::floor(pf_eta / ETA_BIN);
    base.iphi = std::floor(pf_phi / PHI_BIN);

    const reco::GenParticle* bestGen = nullptr;
    float bestDR2 = DR2_MAX;

    // search 3x3 neighborhood
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {

        Cell c;
        c.ieta = base.ieta + di;
        c.iphi = wrapPhiBin(base.iphi + dj, nphi);

        auto it = genGrid.find(c);
        if (it == genGrid.end()) continue;

        for (const auto *gen : it->second) {

          float dEta = pf_eta - gen->eta();
          float dPhi = delta_Phi(pf_phi, gen->phi());
          float dR2  = dEta*dEta + dPhi*dPhi;

          if (dR2 < bestDR2) {
            bestDR2 = dR2;
            bestGen = gen;
          }
        }
      }
    }

    // store result
    if (bestGen) {
      particle_match_pdgid.push_back(bestGen->pdgId());
      particle_match_dr2.push_back(bestDR2);
    } else {
      particle_match_pdgid.push_back(0); // unmatched
      particle_match_dr2.push_back(-1.);
    }

    // recursively find ancestry
    std::vector<int> truthLabel_vec;
    std::set<int> truthLabel_set;

    if (bestGen) {
      truthLabel_set = getTruthLabel(bestGen,

                                    gen_W_q0_ptr, 
                                    gen_W_q1_ptr, 
                                    gen_b_ptr,
                                    gen_Z_q0_ptr, 
                                    gen_Z_q1_ptr,
                                    gen_h_q0_ptr, 
                                    gen_h_q1_ptr,
                                    gen_t_W_q0_ptr, 
                                    gen_t_W_q1_ptr, 
                                    gen_t_b_ptr);
      
      for (int truthLabel: truthLabel_set) {
        truthLabel_vec.push_back(truthLabel);
      }
      if (truthLabel_vec.size() == 0) {
        truthLabel_vec.push_back(0);
      }
    }
    else {
      truthLabel_vec.push_back(0);
    }

    particle_truthLabel.push_back(truthLabel_vec);

  } // end loop over pf cands

  // Run old jet sorting algorithm using slimmedJetsAK8.

  // particle_oldAlgoAK8Index is filled separately from the complete input
  // collection; oldAlg.akJet_indices() contains only selected input jets.
  oldJetSortingAlgorithm oldAlg(jetPtCut_, akRadius_, caRadius_, cosThrust_, true);

  oldAlg.run(*packedPFCands, *ak8Jets);

  particle_oldAlgoLabel = oldAlg.labels();
  // Membership in the complete slimmedJetsAK8 collection.
  // This intentionally includes jets below jetPtCut_ so that the evaluation
  // can rescan the AK8 pT threshold. Membership above jetPtCut_ remains the same.
  particle_oldAlgoAK8Index = ak8Membership;
  particle_oldAlgoCA8Index = oldAlg.caJet_indices();

  // push_back already done by corrected AK8 jets

  // Run old jet sorting algorithm using custom jets
  oldJetSortingAlgorithm newAlg(jetPtCut_, akRadius_, caRadius_, cosThrust_, false);

  newAlg.run(*packedPFCands, *ak8Jets);

  // minEventJets_ == 0 is the legacy, single-threshold behavior: no event
  // selection is applied. A positive value opts into the two-threshold mode.
  // In that mode, require minEventJets_ reclustered AK jets above the high
  // event threshold, while every AK jet above jetPtCut_ (the lower collection
  // threshold used by collectAKConstituents) enters the subsequent steps.
  if (minEventJets_ > 0) {
    const auto nHighPtAKJets = std::count_if(
        newAlg.akJets().begin(), newAlg.akJets().end(),
        [this](const oldJetSortingAlgorithm::JetRecord& jet) {
          return jet.p4.Pt() >= eventJetPtCut_;
        });
    passesEventSelection_ =
        nHighPtAKJets >= static_cast<decltype(nHighPtAKJets)>(minEventJets_);
  }

  particle_newAlgoLabel = newAlg.labels();
  particle_newAlgoAKIndex = newAlg.akJet_indices();
  particle_newAlgoCAIndex = newAlg.caJet_indices();

  for (const auto& jet : newAlg.akJets()) {
    ak_pt.push_back(jet.p4.Pt());
    ak_eta.push_back(jet.p4.Eta());
    ak_phi.push_back(jet.p4.Phi());
    ak_mass.push_back(jet.p4.M());
  }

  // Check assignments are correct size
  const size_t nPF = particle_pt.size();

  assert(particle_truthLabel.size()       == nPF);

  assert(particle_oldAlgoLabel.size()     == nPF);
  assert(particle_oldAlgoAK8Index.size()  == nPF);
  assert(particle_oldAlgoCA8Index.size()   == nPF);

  assert(particle_newAlgoLabel.size()     == nPF);
  assert(particle_newAlgoAKIndex.size()   == nPF);
  assert(particle_newAlgoCAIndex.size()   == nPF);

  // Fill every analyzed event. Rejected events must remain in the tree so the
  // evaluator can measure end-to-end selection/reconstruction efficiency.
  tree_->Fill();

} // end analyze()

//------------------------------------
// Parameter descriptions
//------------------------------------

void ExistingOptimizationNtuplizer::fillDescriptions(
    edm::ConfigurationDescriptions& descriptions) {

  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>(
    "packedPFCandidates",
    edm::InputTag("packedPFCandidates"));

  desc.add<edm::InputTag>(
    "vertices",
    edm::InputTag("offlineSlimmedPrimaryVertices"));

  desc.add<edm::InputTag>(
    "ak4Jets",
    edm::InputTag("slimmedJets"));

  desc.add<edm::InputTag>(
    "ak8Jets",
    edm::InputTag("slimmedJetsAK8"));

  desc.add<edm::InputTag>(
    "met",
    edm::InputTag("slimmedMETs"));

  desc.add<edm::InputTag>(
    "rho",
    edm::InputTag("fixedGridRhoFastjetAll"));

  desc.add<edm::InputTag>(
    "genParticles",
    edm::InputTag("prunedGenParticles"));

  desc.add<edm::InputTag>(
    "genJets",
    edm::InputTag("slimmedGenJets"));

  desc.add<edm::InputTag>(
    "genAK8Jets",
    edm::InputTag("slimmedGenJetsAK8"));

  desc.add<bool>("isMC", true);

  desc.add<double>("jetPtCut", 300.0);

  desc.add<double>("eventJetPtCut", 300.0);

  // Zero disables the event-level selection and preserves the original
  // single-threshold ntuplizer behavior.
  desc.add<unsigned int>("minEventJets", 0);

  desc.add<double>("akRadius", 0.8);

  desc.add<double>("caRadius", 0.8);

  desc.add<double>("cosThrust", 0.85);

  descriptions.add(
    "ExistingOptimizationrNtuplizer",
    desc);
}

//------------------------------------
// Define plugin
//------------------------------------

DEFINE_FWK_MODULE(ExistingOptimizationNtuplizer);
