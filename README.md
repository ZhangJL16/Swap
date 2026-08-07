# Swap Certified UAV Research Bundle

The active research package is in `review_bundle/`.

```bash
cd review_bundle
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

See `review_bundle/README.md` for multi-step Generator-SAC and comparison commands.
The checked-in mission gate, repaired 2k matrix, and five-seed 10k matrix are synthetic software
and empirical evidence only, not calibrated real-flight safety evidence.
