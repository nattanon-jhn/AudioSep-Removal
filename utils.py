import os
import datetime
import json
import logging
import librosa
import pickle
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from models.audiosep import AudioSep
from models.resunet import ResUNet30

# ---------------------------------------------------------------------
# 🔧 PATCH สำหรับ PyTorch 2.6+: อนุญาต numpy types ที่ใช้ใน checkpoint เก่า
# ---------------------------------------------------------------------
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray.scalar,
    np.dtype,
    np.ndarray,
    np.generic,
])
# ---------------------------------------------------------------------


# ============================================================
# Logging / warnings
# ============================================================
def ignore_warnings():
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.functional")
    warnings.filterwarnings(
        "ignore",
        message=r"Some weights of the model checkpoint at roberta-base.*"
    )


def create_logging(log_dir, filemode="w"):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0
    while os.path.isfile(os.path.join(log_dir, f"{i1:04d}.log")):
        i1 += 1

    log_path = os.path.join(log_dir, f"{i1:04d}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


# ============================================================
# Audio utils
# ============================================================
def float32_to_int16(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -1, 1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: np.ndarray) -> np.ndarray:
    return (x / 32767.0).astype(np.float32)


def energy(x: torch.Tensor) -> torch.Tensor:
    return torch.mean(x ** 2)


# ============================================================
# YAML
# ============================================================
def parse_yaml(config_yaml: str) -> Dict:
    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


# ============================================================
# Metrics
# ============================================================
def calculate_sdr(ref: np.ndarray, est: np.ndarray, eps=1e-10) -> float:
    noise = est - ref
    num = max(np.mean(ref ** 2), eps)
    den = max(np.mean(noise ** 2), eps)
    return 10.0 * np.log10(num / den)


def calculate_sisdr(ref: np.ndarray, est: np.ndarray) -> float:
    eps = np.finfo(ref.dtype).eps
    ref = ref.reshape(-1, 1)
    est = est.reshape(-1, 1)

    alpha = (ref.T @ est) / (ref.T @ ref + eps)
    e_true = alpha * ref
    e_res = est - e_true

    return 10 * np.log10((e_true**2).sum() / (e_res**2).sum() + eps)


# ============================================================
# Statistics
# ============================================================
class StatisticsContainer:
    def __init__(self, statistics_path):
        self.statistics_path = statistics_path
        self.backup_statistics_path = (
            os.path.splitext(statistics_path)[0]
            + "_"
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + ".pkl"
        )
        self.statistics_dict = {"train": [], "test": []}

    def append(self, steps, statistics, split="train", flush=True):
        statistics["steps"] = steps
        self.statistics_dict[split].append(statistics)
        if flush:
            self.flush()

    def flush(self):
        with open(self.statistics_path, "wb") as f:
            pickle.dump(self.statistics_dict, f)
        with open(self.backup_statistics_path, "wb") as f:
            pickle.dump(self.statistics_dict, f)


# ============================================================
# 🔧 Model factory (สำคัญ)
# ============================================================
def get_model_class(model_type: str):
    if model_type == "ResUNet30":
        return ResUNet30
    else:
        raise NotImplementedError(f"Unknown model_type: {model_type}")


# ============================================================
# Separation model builders
# ============================================================
def get_ss_model(config_yaml: str) -> nn.Module:
    configs = parse_yaml(config_yaml)

    SsModel = get_model_class(configs["model"]["model_type"])
    return SsModel(
        input_channels=configs["model"]["input_channels"],
        output_channels=configs["model"]["output_channels"],
        condition_size=configs["model"]["condition_size"],
    )


def load_ss_model(
    configs: Dict,
    checkpoint_path: str,
    query_encoder: nn.Module,
) -> AudioSep:
    """
    Load AudioSep Lightning checkpoint (PyTorch 2.6+ safe)
    """

    SsModel = get_model_class(configs["model"]["model_type"])
    ss_model = SsModel(
        input_channels=configs["model"]["input_channels"],
        output_channels=configs["model"]["output_channels"],
        condition_size=configs["model"]["condition_size"],
    )

    pl_model = AudioSep.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        waveform_mixer=None,
        query_encoder=query_encoder,
        loss_function=None,
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
        map_location="cpu",
        weights_only=False,   # 🔧 สำคัญมากสำหรับ checkpoint เก่า
    )

    return pl_model
    
# ============================================================
# Addition for Evaluation
# ============================================================
def get_mean_sdr_from_dict(sdr_dict):
    r"""Calculate mean SDR from a dictionary.
    
    Args:
        sdr_dict (dict): {class_id: sdr_value}
        
    Returns:
        mean_sdr (float)
    """
    sdrs = []
    for class_id in sdr_dict.keys():
        if not np.isnan(sdr_dict[class_id]):
            sdrs.append(sdr_dict[class_id])
    
    return np.mean(sdrs)
