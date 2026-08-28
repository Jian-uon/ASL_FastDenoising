#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Print the subject IDs of one data split, exactly as the training loader partitions them.

Scripts that walk the dataset directory themselves (eval_cbf.py globbed `sub-*` and kept the
first N alphabetically) silently evaluate on training subjects. This reproduces the split the
loader actually uses -- same subject ordering, same ratios, same MONAI `partition_dataset`
shuffle -- so those scripts can be pointed at the held-out subjects with `--subjects`.

The partition depends only on the order and length of the subject list, not on the cached
volumes, so nothing is loaded here and it runs in a second on a login node.

Usage:
  python scripts/dump_split.py --config env/hpc/configs/server_v35_joint.yml --split test
  python scripts/dump_split.py --config ... --split test --out test_subjects.txt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monai.data import partition_dataset

from config.conf_data import Config
from dataio.data_generator import DatasetGenerator


def main() -> int:
    p = argparse.ArgumentParser("print the subject IDs of one split")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out", default=None, help="also write the IDs to this file, one per line")
    p.add_argument("--sep", default="\n", help="separator for stdout ('\n' or ' ')")
    a = p.parse_args()

    cfg = Config(a.config)
    subjects = DatasetGenerator(cfg).generate_dataset()
    dl = cfg.data_loading
    parts = partition_dataset(
        data=subjects,
        ratios=[dl.train_ratio, dl.val_ratio, dl.test_ratio],
        shuffle=dl.shuffle,
    )
    ids = sorted(d["subject_id"] for d in parts[["train", "val", "test"].index(a.split)])

    sizes = " ".join("%s=%d" % (n, len(x)) for n, x in zip(("train", "val", "test"), parts))
    print("# %s  (%s, shuffle=%s)" % (sizes, os.path.basename(a.config), dl.shuffle), file=sys.stderr)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ids) + "\n")
        print("# wrote %d ids -> %s" % (len(ids), a.out), file=sys.stderr)
    sys.stdout.write(a.sep.join(ids) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
