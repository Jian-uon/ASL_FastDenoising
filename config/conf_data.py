import os
import yaml
from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class DatasetConfig:
    root_path: str
    subject_prefix: str
    raw_dir: str


@dataclass
class ASL_Denoiser_TrainParams:
    max_steps: int
    eval_every: int
    asl_hw: int
    asl_z: int
    t1_hw: int
    t1_z: int
    lr: float
    weight_decay: float
    lambda_cos: float
    w_n2n: float
    w_anat_roi: float
    w_anat_ratio: float = 0.0   # deprecated (no dual-branch structure loss)
    w_disentangle: float = 0.0  # deprecated (no shared/private split)
    w_tv: float = 0.0
    w_eps: float = 0.0    # deprecated (no VAE)
    w_global: float = 0.0  # deprecated (no PoE)
    w_seg: float = 0.1            # T1 GM/WM/CSF segmentation aux (multi-task)
    w_contrast: float = 0.1       # contrast-aware boundary L1 in ASL branch
    contrast_boost: float = 4.0   # boundary weight multiplier in contrast_weighted_l1
    w_bg: float = 0.5             # background suppression
    w_grad: float = 0.5           # gradient/edge loss
    w_ssim: float = 0.2           # 1-SSIM loss
    w_sure: float = 0.0           # SURE-based no-GT denoising regularizer
    sure_eps: float = 1e-3        # SURE finite-difference perturbation step
    sure_warmup_steps: int = 30   # ramp w_sure 0 → target over first N steps
    sure_anneal_start: int = 0    # after this step, decay w_sure → sure_anneal_to (0 disables)
    sure_anneal_to: float = 0.0   # target w_sure after anneal completes (linear from anneal_start to max_steps)


@dataclass
class DataLoadingConfig:
    batch_size: int
    num_workers: int
    shuffle: bool
    train_ratio: float
    val_ratio: float
    test_ratio: float


class Config:
    def __init__(self, config_path: str):
        print("config_path: ", config_path)
        with open(config_path, 'r', encoding="utf8") as f:
            config_dict = yaml.safe_load(f)
        self.dataset = DatasetConfig(**config_dict['dataset'])
        self.data_loading = DataLoadingConfig(**config_dict['data_loading'])
        self.asl_denoiser_train_params = ASL_Denoiser_TrainParams(**config_dict['asl_denoiser_train_params'])
