# Swap Certified UAV Research Bundle

The active research package is in `review_bundle/`.

```bash
cd review_bundle
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m unittest discover -v -s tests -p 'test_*.py'
```

See `review_bundle/README.md` for multi-step Generator-SAC and comparison commands.
All checked-in calibration and experiments are synthetic software evidence, not real-flight safety
evidence.
