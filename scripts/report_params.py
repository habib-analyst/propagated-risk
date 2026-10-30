"""Report fitted IE-Prop edge params per preregistered train seed.
Deterministic refit (fit_models is deterministic); used only for the
PREREG M1 check (d0/d1 separation), never for metric selection.
Usage: python3 scripts/report_params.py  (run from build dir)
"""
import sys, json
sys.path.insert(0, "src")
from propagated_risk import generate_traces, fit_models, TYPES

out = {}
for ts in (2101, 2102, 2103, 2104, 2105):
    models = fit_models(generate_traces(ts, 1200))
    p = models["edge"]["ieprop"]
    out[str(ts)] = {t: [float(p["d0"][i]), float(p["d1"][i])]
                    for i, t in enumerate(TYPES)}
print(json.dumps(out, indent=2))
with open("results/fitted_params.json", "w") as f:
    json.dump(out, f, indent=2)
