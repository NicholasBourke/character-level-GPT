import json
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import CharGPT2
from train import train_epoch

device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class ModelConfig:
    L: int = 512
    K: int = 256
    D: int = 512
    n_layer: int = 12
    n_head: int = 2
    dropout: float = 0.2
    bias: bool = False

@dataclass
class TrainConfig:
    bsz: int = 16
    # eval_bsz: int = 128
    total_steps: int = 48000
    epoch_steps: int = 12000
    eval_steps: int = 32
    save_step: int = 800





### CYCLE TRAINING

mcfg = ModelConfig()
tcfg = TrainConfig()

lr = (1e-4, 1e-1)
mtm = (0.8, 0.9)
wd = 1e-5

load_step = 44800

folder = f"drive/MyDrive/CharGPT/cps/SGDCM 1cycle T={tcfg.total_steps} B={tcfg.bsz}"
tag = f"clr=[{lr[0]:.0e}, {lr[1]:.0e}] cm=[{mtm[1]:.2f}, {mtm[0]:.2f}] wd={wd:.0e}"

model = CharGPT2(mcfg).to(device)

no_decay = [p for name, p in model.named_parameters() if name.endswith("bias") or name.endswith("gain")]
decay = [p for name, p in model.named_parameters() if name.endswith("weight")]
optimizer = torch.optim.SGD([{"params": decay, "weight_decay": wd}, {"params": no_decay}], lr=lr[1], momentum=mtm[1])

scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr[0], max_lr=lr[1],
                                              step_size_up=int(tcfg.total_steps/2),
                                              base_momentum=mtm[0], max_momentum=mtm[1])

if load_step > 0:
    checkpoint = torch.load(f"{folder}/{tag}/{load_step}.pth")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    lrs, mtms, tr_losses, ev_losses, accs = train_epoch(model, optimizer, scheduler, tcfg,
                                                        load_step, save_path=f"{folder}/{tag}")
else:
    lrs, mtms, tr_losses, ev_losses, accs = train_epoch(model, optimizer, scheduler, tcfg, save_path=f"{folder}/{tag}")

tracking = {"lr": lrs, "mtm": mtms, "trl": tr_losses, "evl": ev_losses, "acc": accs}
with open(f"{folder}/{tag}/{load_step}-{load_step + tcfg.epoch_steps}.json", "w") as file:
    json.dump(tracking, file)