"""Calibrated signal/background ntuples; same scan options as compact cfg."""
import runpy
import sys
from pathlib import Path

for argument in sys.argv[1:]:
    if argument.startswith("analysisMode=") and argument.split("=", 1)[1].lower() not in ("true", "1"):
        raise ValueError("runSensitivityScan_cfg.py requires analysisMode=True")
if not any(argument.startswith("analysisMode=") for argument in sys.argv[1:]):
    sys.argv.append("analysisMode=True")
globals().update(runpy.run_path(str(Path(__file__).with_name("runCompactOptimizationScan_cfg.py"))))
