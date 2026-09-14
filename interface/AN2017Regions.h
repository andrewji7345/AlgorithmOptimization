#ifndef SuuAnalysis_ExistingOptimization_AN2017Regions_h
#define SuuAnalysis_ExistingOptimization_AN2017Regions_h

#include <cmath>
#include <cstdint>

namespace an2017 {
// Shared strict AN-23-067 region boundaries. Zero denotes no analysis region.
// Kept independent of ROOT/CMSSW for direct boundary and exclusivity tests.
enum Region : std::uint8_t { none = 0, SR = 1, CR = 2, AT1b = 3, AT0b = 4 };
inline Region region(bool baseline, bool valid, bool jetVeto, unsigned btags,
                     unsigned n300a, unsigned n300b, unsigned n50a, unsigned n50b,
                     double mass100a, double mass100b) {
  if (!baseline || !valid || !jetVeto) return none;
  const bool tagA = n300a >= 2, tagB = n300b >= 2;
  const bool antiA = n50a == 0 && std::isfinite(mass100a) && mass100a >= 0. && mass100a < 150.;
  const bool antiB = n50b == 0 && std::isfinite(mass100b) && mass100b >= 0. && mass100b < 150.;
  if (tagA && tagB) return btags ? SR : CR;
  if ((tagA && antiB) || (tagB && antiA)) return btags ? AT1b : AT0b;
  return none;
}
}  // namespace an2017
#endif
