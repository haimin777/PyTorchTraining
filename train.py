import os
import sys
import re
import shutil
import json
import pandas as pd
from utils import build_dataloaders_from_two_dfs
from utils import TrainCfg, build_model, accuracy
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Literal, Dict, Any


def _safe_ckpt_path(cfg: TrainCfg, epoch_idx: int, val_loss: float, val_acc: float) -> str:
    """Build a filesystem-safe checkpoint path using the template."""
    fname = cfg.ckpt_tmpl.format(epoch=epoch_idx, val_loss=val_loss, val_acc=val_acc)
    # Sanitize any odd characters (just in case)
    fname = re.sub(r'[^A-Za-z0-9._=+-]', '-', fname)
    return os.path.join(cfg.ckpt_dir, fname)


def fit(
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainCfg,
) -> Dict[str, Any]:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    device = torch.device(cfg.device)

    model = build_model(cfg).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        fused=True if "cuda" in cfg.device and torch.cuda.is_available() else False,
    )

    # ---- LR schedulers ----
    steps_per_epoch = len(train_loader)
    if cfg.scheduler == "cosine":
        if cfg.cosine_step_per_batch:
            T_max = cfg.epochs * steps_per_epoch
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max, eta_min=cfg.cosine_eta_min
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.cosine_eta_min
            )
    elif cfg.scheduler == "cosine_warm_restarts":
        # You can step this per-batch with fractional epoch input
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=cfg.cwr_T0 * (steps_per_epoch if cfg.cosine_step_per_batch else 1),
            T_mult=cfg.cwr_Tmult, eta_min=cfg.cwr_eta_min
        )
    else:
        scheduler = None

    scaler = torch.amp.GradScaler(enabled=cfg.amp)

    best_acc = -1.0
    best_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.epochs):
        # ---------- Train ----------
        model.train()
        running_loss = 0.0
        running_acc = 0.0

        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()

            if cfg.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            # Scheduler stepping
            if scheduler is not None:
                if cfg.scheduler == "cosine":
                    if cfg.cosine_step_per_batch:
                        scheduler.step()
                elif cfg.scheduler == "cosine_warm_restarts":
                    # warm restarts want fractional epoch when stepping per batch
                    if cfg.cosine_step_per_batch:
                        scheduler.step(epoch + step / steps_per_epoch)

            running_loss += loss.item()
            running_acc += accuracy(logits, yb)

        epoch_train_loss = running_loss / steps_per_epoch
        epoch_train_acc = running_acc / steps_per_epoch

        # Epoch-step for cosine if configured that way
        if scheduler is not None and cfg.scheduler == "cosine" and not cfg.cosine_step_per_batch:
            scheduler.step()

        # ---------- Validate ----------
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                val_loss += loss.item()
                val_acc += accuracy(logits, yb)

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_acc / len(val_loader)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] "
            f"lr={lr_now:.2e}  "
            f"train: loss={epoch_train_loss:.4f} acc={epoch_train_acc:.4f}  "
            f"val: loss={epoch_val_loss:.4f} acc={epoch_val_acc:.4f}"
        )

        ckpt_path = _safe_ckpt_path(cfg, epoch_idx=epoch + 1,
                                        val_loss=epoch_val_loss, val_acc=epoch_val_acc)
        torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "cfg": cfg.__dict__,
                    "epoch": epoch + 1,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc,
                },
                ckpt_path,
            )
        # Optional: keep a rolling "last.pt"
        shutil.copyfile(ckpt_path, os.path.join(cfg.ckpt_dir, "last.pt"))
    return {"model": model, "history": history, "best_path": best_path, "best_val_acc": best_acc}



def main(dataset_dir, config_path):

    with open(config_path) as f:

        config = json.load(f)

    trn_path = config["trn_csv"]
    tst_path = config["tst_csv"]
    batch_size = config['batch_size']
    img_size = config['img_size']
    model_type = config['model']
    pad = config['pad']
    debug = config['debug']
    train_df = pd.read_csv(trn_path)
    val_df = pd.read_csv(tst_path)
    if debug:
        train_df = train_df[:200]
        val_df = val_df[:100]

    cfg = TrainCfg(
        num_classes=1,
        epochs=config['epochs'],
        lr=3e-4,
        scheduler="cosine_warm_restarts",  # "cosine" for monotonic decay, or "none"
        cwr_T0=5, cwr_Tmult=2, cwr_eta_min=1e-6,
        cosine_step_per_batch=True,  # keep True for smooth cosine curves
        pretrained=True,
        freeze_backbone=False,
        label_smoothing=0.05,
        amp=True,
        grad_clip_norm=1.0,
        compile_model=False,
        ckpt_dir="checkpoints",
        ckpt_name="effb0_best.pt",
    )

    train_df['paths'] = train_df['paths'].apply(lambda x: os.path.join(dataset_dir, x))
    val_df['paths'] = val_df['paths'].apply(lambda x: os.path.join(dataset_dir, x))

    train_loader, val_loader, label_to_index = build_dataloaders_from_two_dfs(
        train_df,
        val_df,
        batch_size=batch_size,
        img_size=img_size,
        aug=True,
        num_workers=os.cpu_count() // 2 if os.cpu_count() else 4,
        pin_memory=torch.cuda.is_available(),
        balance_sampler=True,  # set False if classes are balanced
    )

    print("Classes:", {v: k for k, v in label_to_index.items()})
    xb, yb = next(iter(train_loader))
    print("Batch shapes:", xb.shape, yb.shape)

    results = fit(train_loader, val_loader, cfg)

    



if __name__ == "__main__":

    main(sys.argv[1], sys.argv[2])