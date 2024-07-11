import time
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from model import CharGPT

@dataclass
class ModelConfig:
    L: int = 512            # context length
    K: int = 256            # vocabulary size
    D: int = 512            # model dimension (hidden vector size)
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




def min_sec(t):
    sec = t % 60
    min = (t-sec)/60
    return f"{min:2.0f}:{sec:2.0f}"


def get_batch(split, data_dir, context_length, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    databytes = open(data_dir+split, "rb").read()
    data = torch.tensor(list(databytes)).to(device)

    ix = torch.randint(data.size(0) - context_length, (batch_size,))
    x = torch.stack([data[i: i + context_length] for i in ix])

    if split == "train":
        y = torch.stack([data[i + 1: i + 1 + context_length] for i in ix])

    if split in ("test", "val"):
        y = torch.stack([data[i + context_length] for i in ix]).to(int)

    return x, y


def create_dataloader(split, data_dir, context_length, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    databytes = open(f"{data_dir}/{split}", "rb").read()
    data = torch.tensor(list(databytes)).to(device)

    n = data.size()[0] - context_length
    r = n % batch_size
    m = int((n-r)/batch_size)

    dataloader = [
        (torch.stack([data[k*batch_size+i: k*batch_size + i + context_length] for i in range(batch_size)]),
         data[k*batch_size + context_length: k*batch_size + context_length + batch_size])
    for k in range(m)]
    
    if r > 0:
        dataloader.append(
            (torch.stack([data[n - r + i: n - r + i + context_length] for i in range(r)]),
             data[-r:])
        )

    return dataloader


def training(model, optimizer, scheduler, cfg, data_dir, save_dir, checkpoint=None):

    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        tracking = checkpoint["tracking"]
        start = checkpoint["step_count"]
    else:
        tracking = {"learning rate": [], "momentum": [], "train loss": [], "val loss": [], "accuracy": []}
        start = 0

    model.train()

    t0 = time.time()
    for step in range(start, start + cfg.epoch_steps):
        x, y = get_batch("train", data_dir, model.cfg.L, cfg.bsz)
        logits = model(x)
        train_loss = F.cross_entropy(logits.view(-1, model.cfg.K), y.view(-1))
        train_loss.backward()
        optimizer.step()

        lr = optimizer.param_groups[0]["lr"]
        mtm = optimizer.param_groups[0]["momentum"]

        scheduler.step()
        optimizer.zero_grad()

        if (step+1) % cfg.eval_step == 0:
            tracking["learning rate"].append(lr)
            tracking["momentum"].append(mtm)
            tracking["train loss"].append(train_loss.item())

            model.eval()
            val_loss_total = 0.0
            acc_total = 0.0
            for _ in range(cfg.eval_steps):
                x, y = get_batch("val", data_dir, model.cfg.L, cfg.bsz)
                logits = model(x)
                val_loss = F.cross_entropy(logits[:, -1, :], y)
                val_loss_total += val_loss.item()

                pred = logits[:, -1, :].argmax(dim=1)
                acc = torch.sum(torch.where(pred == y, 1, 0)).item() / y.size(0)
                acc_total += acc

            tracking["val loss"].append(val_loss_total / cfg.eval_steps)
            tracking["accuracy"].append(acc_total / cfg.eval_steps)

            elapsed = time.time()-t0
            print(f"[{min_sec(elapsed)}] {step + 1}/{start + cfg.epoch_steps}: loss = {train_loss:.3f}")

            model.train()

        if (step+1) % cfg.save_step == 0:
            torch.save({
                "step_count": step + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "tracking": tracking
            },f"{save_dir}/{step + 1}.pth")
            print("MODEL SAVED")

    return tracking


def full_inference(dataloader, model, print_step):
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


def accuracy_test(params_dir, data_dir, batch_size, num_batches):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mcfg = ModelConfig()
    model = CharGPT(mcfg).to(device)
    model.load_state_dict(torch.load(params_dir + "model_state_dict.pth"))
    model.eval()

    acc_total = 0.0
    for _ in range(num_batches):
        x, y = get_batch("test", data_dir, model.cfg.L, batch_size)
        logits = model(x)
        pred = logits[:, -1, :].argmax(dim=1)
        acc = torch.sum(torch.where(pred == y, 1, 0)).item() / y.size(0)
        acc_total += acc

    print(f"Model was {acc_total / num_batches * 100:.2f} accurate over {batch_size * num_batches} random examples.")
