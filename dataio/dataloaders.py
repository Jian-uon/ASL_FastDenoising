import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.conf_data import Config
from dataio.data_generator import DatasetGenerator
from utils.utils import collate_varlen
from dataio.data_classes import (
    get_pre_asl_transform,
    build_cached_subjects,
    ASLTwoSetDataset2DFlat,
)

from monai.data import DataLoader, partition_dataset


def get_asl_2d_loaders(
    config: Config,
    modes=("train", "val"),
    TA_range=(3, 6),          # val/test setA size: stable in-loop selection; setB=12-TA>=6 keeps the uMSE 3-way split valid
    train_TA_range=(2, 10),   # TRAIN setA size: wider, matches the 2-10 inference range (setB>=2 is fine for the N2N mean target)
    TB_range=(6, 9),
    asl_hw=128,
    asl_z=48,
    t1_hw=128,
    t1_z=48,
    cache_rate=1.0,
    cache_workers=0,   # 0 = single-process; avoids Windows spawn deadlock
    num_workers=0,     # DataLoader workers. 0 = single-process (was hardcoded);
                       #   >0 parallelises the per-slice loader/collate so the GPU
                       #   doesn't wait on the main thread. Forced to 0 on Windows.
    pre_tf=None,
    slice_context=0,   # 2.5D: 2*ctx+1 adjacent z-slices per frame (0 = 2D)
):
    subjects = DatasetGenerator(config).generate_dataset()

    pre_tf = pre_tf or get_pre_asl_transform(
        asl_hw=asl_hw,
        asl_z=asl_z,
        t1_hw=t1_hw,
        t1_z=t1_z,
        intensity="percentile", p_lo=1, p_hi=99,
    )

    cached_subjects = build_cached_subjects(
        subjects_list=subjects, pre_tf=pre_tf,
        cache_rate=cache_rate, cache_workers=cache_workers,
    )

    train_sub, val_sub, test_sub = partition_dataset(
        data=cached_subjects,
        ratios=[config.data_loading.train_ratio, config.data_loading.val_ratio, config.data_loading.test_ratio],
        shuffle=config.data_loading.shuffle,
    )
    split_map = {"train": train_sub, "val": val_sub, "test": test_sub}

    datasets = {m: ASLTwoSetDataset2DFlat(
                    split_map[m],
                    TA_range=(train_TA_range if m == "train" else TA_range),
                    TB_range=TB_range, slice_context=slice_context) for m in modes}

    # num_workers>0 parallelises the per-slice loader + collate (big win on a Linux
    # compute node whose GPU would otherwise wait on single-process data loading).
    # Windows uses the 'spawn' start method, which deadlocks with this MONAI/collate
    # stack, so force single-process there regardless of the requested value.
    nw = int(num_workers)
    if os.name == "nt" and nw > 0:
        print(f"[loaders] Windows detected -> forcing num_workers=0 (was {nw}; spawn can deadlock).")
        nw = 0

    loaders = {}
    for m in modes:
        dl_kwargs = dict(
            batch_size=config.data_loading.batch_size,
            shuffle=(m == "train"),
            num_workers=nw,
            collate_fn=collate_varlen,
        )
        if nw > 0:
            # only valid with workers; prefetch_factor with num_workers=0 raises.
            dl_kwargs.update(persistent_workers=True, pin_memory=True, prefetch_factor=4)
        loaders[m] = DataLoader(datasets[m], **dl_kwargs)
    return loaders
