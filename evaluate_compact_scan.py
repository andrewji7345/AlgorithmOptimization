#!/usr/bin/env python3
"""Current scan evaluation entry point: weighted signal/background sensitivity.

Use --campaign and --configuration; see docs/sensitivity_objective.md.
The former signal-only retention/regret evaluator is archived at
legacy/evaluate_compact_scan_physicality.py.
"""
from evaluate_sensitivity import main

if __name__ == "__main__":
    raise SystemExit(main())
