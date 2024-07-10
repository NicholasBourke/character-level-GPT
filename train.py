import math
import time
import json
import numpy as np
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"



def min_sec(t):
    sec = t % 60
    min = (t-sec)/60
    return f"{min:2.0f}:{sec:2.0f}"


def get_batch(split, data_dir, L, B):

    databytes = open(data_dir+split, "rb").read()
    data = np.array(list(databytes))

    ix = torch.randint(len(data) - L, (B,))
    x = torch.stack([torch.from_numpy((data[i:i+L]).astype(np.int64)) for i in ix])

    if split == "train":
        y = torch.stack([torch.from_numpy((data[i+1:i+1+L]).astype(np.int64)) for i in ix])

    if split in ("test", "val"):
        y = torch.tensor([data[i+L] for i in ix]).to(int)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, y = x.to(device), y.to(device)

    return x, y


def train_epoch(model, optimizer, scheduler, cfg, data_dir, load_step=0, save_path=None):

    model.train()

    tracking = {
        "lr": [],
        "mtm": [],
        "train_loss": [],
        "val_loss": [],
        "acc": []
    }

    t0 = time.time()
    for step in range(cfg.epoch_steps):

        x, y = get_batch("train", data_dir, model.cfg.L, cfg.bsz)
        logits = model(x)
        tr_loss = F.cross_entropy(logits.view(-1, model.cfg.K), y.view(-1))
        tr_loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        mtm = optimizer.param_groups[0]["momentum"]

        scheduler.step()
        optimizer.zero_grad()

        if (step+1) % cfg.eval_steps == 0:
            tracking["lr"].append(lr)
            tracking["mtm"].append(mtm)
            tracking["train_loss"].append(tr_loss.item())

            model.eval()
            ev_loss_total = 0.0
            acc_total = 0.0
            for _ in range(cfg.eval_steps):
                x, y = get_batch("val", data_dir, model.cfg.L, cfg.bsz)
                logits = model(x)
                ev_loss = F.cross_entropy(logits[:, -1, :], y)
                ev_loss_total += ev_loss.item()

                pred = logits[:, -1, :].argmax(dim=1)
                acc = torch.sum(torch.where(pred == y, 1, 0)).item() / y.size(0)
                acc_total += acc

            tracking["val_loss"].append(ev_loss_total / cfg.eval_steps)
            tracking["acc"].append(acc_total / cfg.eval_steps)

            elapsed = time.time()-t0
            print(f"[{min_sec(elapsed)}] {step + load_step + 1}/{load_step + cfg.epoch_steps}: loss = {tr_loss:.3f}")

            model.train()

        if (step+1) % cfg.save_step == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, f"{save_path}/{load_step + step + 1}.pth")
            print("MODEL SAVED")

    return tracking


def create_dataloader(split, data_dir, L, B):
    databytes = open(f"{data_dir}/{split}", "rb").read()
    data = torch.tensor(list(databytes))

    n = data.size()[0] - L
    r = n % B
    m = int((n-r)/B)

    dataloader = [(torch.stack([data[k*B+i: k*B+i+L] for i in range(B)]),
                data[k*B+L: k*B+L+B]) for k in range(m)]
    
    dataloader.append((torch.stack([data[n-r+i: n-r+i+L] for i in range(r)]),
                    data[-r:]))

    return dataloader


def inference(dataloader, model, print_step):
    model.eval()

    loss_total = 0.0
    acc_total = 0.0
    t0 = time.time()
    for step, batch in enumerate(dataloader):

        x, y = batch
        logits = model(x)
        loss = F.cross_entropy(logits[:, -1, :], y)
        loss_total += loss.item()

        pred = logits[:, -1, :].argmax(dim=1)
        acc = torch.sum(torch.where(pred == y, 1, 0)).item() / y.size(0)
        acc_total += acc

        if (step+1) % print_step == 0:
            elapsed = time.time()-t0
            print(f"[{min_sec(elapsed)}] {step+1}/{len(dataloader)}: loss = {loss_total/(step+1):.3f}, acc = {acc_total/(step+1) * 100:.2f}%")

    return loss_total / len(dataloader), acc_total / len(dataloader)

