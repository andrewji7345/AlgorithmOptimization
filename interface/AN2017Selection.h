#ifndef SuuAnalysis_ExistingOptimization_AN2017Selection_h
#define SuuAnalysis_ExistingOptimization_AN2017Selection_h

#include <memory>
#include <vector>
#include <cstdint>
#include "TLorentzVector.h"

namespace edm {
class ParameterSet;
class ConsumesCollector;
class Event;
}
class TLorentzVector;

// Nominal UL17 MC object selection and weights from AN-23-067.
// One instance per analyzer stream; evaluate() must precede reclustered corrections.
class AN2017Selection {
public:
  struct EventInfo {
    bool trigger = false, filters = false, leptonVeto = false, jetVeto = false, baseline = false;
    double ht = 0., weight = 1.;
    double btagWeight = 1.;
    bool btagWeightFallback = false;
    unsigned nAK4 = 0, nAK8 = 0, nHeavyAK8 = 0, nBTags = 0;
    std::vector<double> weightVariations;
    double referenceWeight = 1.;
    std::vector<double> referenceWeightVariations;
    std::vector<std::uint8_t> btagWeightVariationFallback;
    std::vector<TLorentzVector> referenceConstituents;
    std::vector<float> btagJetPt, btagJetEta, btagJetDiscriminator;
  };

  AN2017Selection(const edm::ParameterSet&, edm::ConsumesCollector&&);
  ~AN2017Selection();
  EventInfo evaluate(const edm::Event&);
  double correctReclusteredJet(const TLorentzVector& raw, double radius);
  bool passesReclusteredJetVeto(const TLorentzVector& corrected) const;

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

#endif
