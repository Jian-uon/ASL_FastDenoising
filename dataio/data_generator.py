# data_generator.py
# 读取 Config，遍历数据根目录，为每个 subject 汇总 ASL/T1 NIfTI 路径
import os
from typing import List, Dict

from config.conf_data import Config


class DatasetGenerator:
    def __init__(self, config: Config):
        self.config = config

    def _subject_dirs(self) -> List[str]:
        root = self.config.dataset.root_path
        prefix = self.config.dataset.subject_prefix
        return sorted([
            d for d in os.listdir(root)
            if d.startswith(prefix) and os.path.isdir(os.path.join(root, d))
        ])

    def generate_dataset(self) -> List[Dict[str, str]]:
        data = []
        for sid in self._subject_dirs():
            raw_dir = os.path.join(self.config.dataset.root_path, sid, self.config.dataset.raw_dir)
            paths = {
                'asl_diff': os.path.join(raw_dir, 'asldata_diff.nii.gz'),
                't1':       os.path.join(raw_dir, 't1_in_asl.nii.gz'),
                'gm':       os.path.join(raw_dir, 'gm_asl.nii.gz'),
                'wm':       os.path.join(raw_dir, 'wm_asl.nii.gz'),
                'csf':      os.path.join(raw_dir, 'csf_asl.nii.gz'),
                # M0 is needed at evaluation only, to put sCoV on the CBF scale: the
                # quantification divides by the M0 MAP, and dividing by a spatially
                # varying field is not a rescaling, so sCoV on dM and on CBF differ.
                'm0':       os.path.join(raw_dir, 'm0.nii.gz'),
                'subject_id': sid,
            }
            missing = [k for k, v in paths.items() if k != 'subject_id' and not os.path.exists(v)]
            if missing:
                print(f"[WARN] Missing files for {sid}: {missing}")
            else:
                data.append(paths)
        return data
