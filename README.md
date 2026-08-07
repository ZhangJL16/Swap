# Swap Certified UAV Research Bundle

The active research package is in `review_bundle/`.

```bash
cd review_bundle
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

See `review_bundle/README.md` for multi-step Generator-SAC and comparison commands.
The current phase adds Center-Only/Random-in-Generator ablations, deterministic scenario-family
generation, held-out manifest checks, and paper-oriented synthetic result tables.
The checked-in mission gate, repaired 2k matrix, and five-seed 10k matrix are synthetic software
and empirical evidence only, not calibrated real-flight safety evidence.
The current RL-contribution and held-out gates both pass their synthetic evidence criteria. The
central result is negative but important: Center-Only matches Generator-SAC on the current open and
obstacle missions, so demonstrated task competence is primarily attributable to the verified
task-oriented center rather than learned residual optimization.
