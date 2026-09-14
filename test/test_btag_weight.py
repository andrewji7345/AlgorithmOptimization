"""Execute the production C++ event-weight helper on boundary cases."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class BTagWeightTest(unittest.TestCase):
    def test_reference_event_fallback_and_probability_products(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "C++ compiler required for the production helper test")
        source = r'''
#include "interface/AN2017BTagWeight.h"
#include <cassert>
#include <cmath>
int main() {
  an2017::BTagWeight nominal;
  nominal.add(true, .7, .9); nominal.add(false, .2, 1.1);
  assert(std::abs(nominal.value() - .9 * .78 / .8) < 1.e-12);
  assert(!nominal.fallback());
  an2017::BTagWeight saturated;
  saturated.add(true, .6, .8); saturated.add(false, 1., 1.18876);
  assert(saturated.fallback() && saturated.value() == 1.);
  an2017::BTagWeight zero;
  zero.add(true, 0., 1.); assert(zero.fallback() && zero.value() == 1.);
  an2017::BTagWeight negative;
  negative.add(false, .9, 1.2); assert(negative.fallback());
  // The reference tests the final event weight, not each factor separately.
  negative.add(false, .9, 1.2);
  assert(!negative.fallback() && std::abs(negative.value() - .64) < 1.e-12);
  an2017::BTagWeight boundary;
  boundary.add(true, .5, 100.); assert(!boundary.fallback());
  boundary.add(true, .5, 1.01); assert(boundary.fallback());
  an2017::BTagWeight zeroWeight;
  zeroWeight.add(false, .5, 2.);
  assert(!zeroWeight.fallback() && zeroWeight.value() == 0.);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "test.cc").write_text(source)
            subprocess.run([compiler, "-std=c++17", "-I", str(REPO), str(path / "test.cc"),
                            "-o", str(path / "test")], check=True, capture_output=True)
            subprocess.run([str(path / "test")], check=True, capture_output=True)
