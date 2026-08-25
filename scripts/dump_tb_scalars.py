"""Dump selected TB scalars to console as a table."""
import sys
from pathlib import Path
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "C:/tmp/asl_exp/tensorboard/run_full_v37"
keys = sys.argv[2:] or [
    "val/upsnr", "val/subset_consistency", "val/cnr_pred", "val/cnr_ref",
    "val/psnr_ref", "val/l1_b", "val/hfen", "val/lapvar",
]

ea = EventAccumulator(run, size_guidance={"scalars": 0})
ea.Reload()
avail = set(ea.Tags().get("scalars", []))

cols = [k for k in keys if k in avail]
missing = [k for k in keys if k not in avail]
if missing:
    print(f"# missing tags: {missing}", file=sys.stderr)

per_step = defaultdict(dict)
for k in cols:
    for ev in ea.Scalars(k):
        per_step[ev.step][k] = ev.value

short = [k.split("/")[-1] for k in cols]
print(f"{'step':>6}  " + "  ".join(f"{s:>9}" for s in short))
for step in sorted(per_step):
    row = per_step[step]
    print(f"{step:>6}  " + "  ".join(
        f"{row.get(k, float('nan')):>9.4f}" for k in cols))
