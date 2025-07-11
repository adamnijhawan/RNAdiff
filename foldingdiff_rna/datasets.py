import os
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Optional, Union

import logging

from foldingdiff import utils
from foldingdiff import custom_metrics as cm

TRIM_STRATEGIES = ["leftalign", "randomcrop", "discard"]

RNA_FEATURE_NAMES = [
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi", "pucker_P", "pucker_tm"
]
RNA_IS_ANGULAR = [
    True, True, True, True, True, True, True, True, False
]


class RnaTorsionDataset(Dataset):
    """
    Dataset for RNA torsion angles + sugar puckering loaded from .npy files.
    Each .npy file should be an (L, D) array (L = seq length, D = num features).
    """

    feature_names = {"angles": RNA_FEATURE_NAMES}
    feature_is_angular = {"angles": RNA_IS_ANGULAR}

    def __init__(
        self,
        npy_dir: Union[str, Path],
        pad: int = 512,
        min_length: int = 0,
        trim_strategy: str = "leftalign",
        zero_center: bool = True,
    ):
        super().__init__()
        assert trim_strategy in TRIM_STRATEGIES, f"Invalid trim strategy: {trim_strategy}"

        self.pad = pad
        self.min_length = min_length
        self.trim_strategy = trim_strategy
        self.zero_center = zero_center

        self.filenames = sorted(Path(npy_dir).glob("*.npy"))
        self.data = [np.load(f) for f in self.filenames]
        self.data = [d for d in self.data if d.shape[0] >= min_length]

        self.rng = np.random.default_rng(seed=6489)
        self._length_rng = np.random.default_rng(seed=6489)

        # Zero-center mean calculation
        self.means = None
        if self.zero_center:
            concat = np.concatenate(self.data, axis=0)
            self.means = cm.wrapped_mean(concat, axis=0)
            logging.info(f"Zero-centering applied with means: {self.means}")

    def __len__(self):
        return len(self.data)

    def sample_length(self, n=1):
        lengths = [d.shape[0] for d in self.data]
        if n == 1:
            return self._length_rng.choice(lengths)
        return self._length_rng.choice(lengths, size=n)

    def get_masked_means(self):
        return np.copy(self.means) if self.means is not None else None

    def __getitem__(self, index: int, ignore_zero_center: bool = False) -> Dict[str, torch.Tensor]:
        x = self.data[index]

        if self.means is not None and not ignore_zero_center:
            x = x - self.means
            angular_idx = np.where(self.feature_is_angular["angles"])[0]
            x[:, angular_idx] = utils.modulo_with_wrapped_range(x[:, angular_idx], -np.pi, np.pi)

        # Padding/trimming
        l = min(self.pad, x.shape[0])
        attn_mask = torch.zeros(self.pad)
        attn_mask[:l] = 1.0

        if x.shape[0] < self.pad:
            x = np.pad(x, ((0, self.pad - x.shape[0]), (0, 0)), mode="constant", constant_values=0)
        elif x.shape[0] > self.pad:
            if self.trim_strategy == "leftalign":
                x = x[:self.pad]
            elif self.trim_strategy == "randomcrop":
                start = self.rng.integers(0, x.shape[0] - self.pad)
                x = x[start:start + self.pad]
            elif self.trim_strategy == "discard":
                raise ValueError("Sequences longer than pad are not supported in 'discard' mode.")
            else:
                raise ValueError(f"Unknown trim strategy: {self.trim_strategy}")

        x = torch.from_numpy(x).float()
        position_ids = torch.arange(0, self.pad, dtype=torch.long)

        return {
            "angles": x,
            "attn_mask": attn_mask,
            "position_ids": position_ids,
            "lengths": torch.tensor(l, dtype=torch.int64),
        }


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("npy_dir", type=str, help="Directory of .npy files")
    args = parser.parse_args()

    dset = RnaTorsionDataset(args.npy_dir, pad=128)
    print(f"Loaded {len(dset)} sequences")
    item = dset[0]
    for k, v in item.items():
        print(f"{k}: {v.shape if hasattr(v, 'shape') else v}")
