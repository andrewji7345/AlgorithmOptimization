#ifndef SuuAnalysis_ExistingOptimization_AN2017BTagWeight_h
#define SuuAnalysis_ExistingOptimization_AN2017BTagWeight_h

#include <cmath>
#include <limits>

namespace an2017 {
// Method 1a event probabilities, including the final analysis's event-level
// fallback (combinedROOT/rootProcessor.C:1234-1236 at 8ef9b506).
// Do not sanitize individual jets: that changes the reference prescription.
class BTagWeight {
public:
  void add(bool tagged, double efficiency, double sf) {
    if (tagged) {
      mcTagged_ *= efficiency;
      dataTagged_ *= sf * efficiency;
    } else {
      mcUntagged_ *= 1. - efficiency;
      dataUntagged_ *= 1. - sf * efficiency;
    }
  }
  double raw() const {
    const double denominator = mcTagged_ * mcUntagged_;
    return denominator == 0. ? std::numeric_limits<double>::quiet_NaN()
                             : dataTagged_ * dataUntagged_ / denominator;
  }
  bool fallback() const {
    const double value = raw();
    return !std::isfinite(value) || value < 0. || value > 100.;
  }
  double value() const { return fallback() ? 1. : raw(); }

private:
  double mcTagged_ = 1., mcUntagged_ = 1., dataTagged_ = 1., dataUntagged_ = 1.;
};
}  // namespace an2017

#endif
