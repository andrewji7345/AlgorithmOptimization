"""Exercise production region boundaries and exclusivity with real C++ code."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]


class ReferenceRegionTest(unittest.TestCase):
    def test_region_boundaries_and_exclusivity(self):
        compiler = shutil.which("g++")
        self.assertIsNotNone(compiler, "C++ compiler required for region helper validation")
        source = r'''
#include "interface/AN2017Regions.h"
#include <cassert>
#include <limits>
int main() {
  using namespace an2017;
  assert(region(true,true,true,1,2,2,2,2,600,600)==SR);
  assert(region(true,true,true,0,2,2,2,2,600,600)==CR);
  assert(region(true,true,true,1,2,0,2,0,600,0)==AT1b);
  assert(region(true,true,true,0,2,0,2,0,600,0)==AT0b);
  assert(region(true,true,true,1,0,2,0,2,0,600)==AT1b);
  assert(region(true,true,true,0,0,2,0,2,0,600)==AT0b);
  assert(region(false,true,true,1,2,2,2,2,600,600)==none);
  assert(region(true,false,true,1,2,2,2,2,600,600)==none);
  assert(region(true,true,false,1,2,2,2,2,600,600)==none);
  assert(region(true,true,true,1,1,2,2,2,600,600)==none);
  assert(region(true,true,true,1,2,0,2,1,600,0)==none);
  assert(region(true,true,true,1,2,0,2,0,600,150)==none);
  assert(region(true,true,true,1,2,0,2,0,600,149.999)==AT1b);
  assert(region(true,true,true,1,2,0,2,0,600,-1)==none);
  assert(region(true,true,true,1,2,0,2,0,600,std::numeric_limits<double>::quiet_NaN())==none);
  // Sweep physically consistent count combinations: tag/anti-tag selections
  // cannot overlap and swapping superjet ordering cannot change the region.
  for(unsigned a50=0;a50<6;++a50) for(unsigned b50=0;b50<6;++b50)
    for(unsigned a300=0;a300<=a50;++a300) for(unsigned b300=0;b300<=b50;++b300)
      for(unsigned btags=0;btags<4;++btags) {
        const auto r=region(true,true,true,btags,a300,b300,a50,b50,0,0);
        assert(r==region(true,true,true,btags,b300,a300,b50,a50,0,0));
        if(r==SR || r==CR) assert(a300>=2 && b300>=2);
        if(r==AT1b || r==AT0b) assert((a300>=2 && b50==0) || (b300>=2 && a50==0));
        if(r!=none) assert((r==SR || r==AT1b)==(btags>0));
      }
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "regions.cc").write_text(source)
            subprocess.run([compiler, "-std=c++17", "-I", str(REPO), str(path / "regions.cc"),
                            "-o", str(path / "regions")], check=True, capture_output=True)
            subprocess.run([str(path / "regions")], check=True, capture_output=True)
