"""
训练 1D-CNN 手势分类器。

使用:
    python train.py
    python train.py --epochs 80 --batch-size 64

产出:
    checkpoints/best.pt          验证最优权重
    checkpoints/last.pt          最后一轮权重
    checkpoints/train_log.csv    每轮 loss/acc
"""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (BATCH_SIZE, CKPT_DIR, EPOCHS, LR,
                    NUM_CLASSES, VAL_RATIO, VOCAB)
from dataset import HandSignDataset, list_all_samples, split_by_label
from model import SignNet, count_params


def make_loaders(batch_size: int):
    files = list_all_samples()
    if not files:
        raise RuntimeError("data/ 目录里一个 npz 都没有, 先跑 record.py 录数据")

    train_files, val_files = split_by_label(files, VAL_RATIO)
    train_ds = HandSignDataset(train_files, augment=True)
    val_ds = HandSignDataset(val_files, augment=False)

    print(f"samples: train={len(train_ds)} val={len(val_ds)}")

    # 每类样本统计
    counts = np.zeros(NUM_CLASSES, dtype=int)
    for f in train_files:
        counts[int(f.name.split('_', 1)[0])] += 1
    for i, c in enumerate(counts):
        print(f"  [{i}] {VOCAB[i]:8s} train={c}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)
    return train_loader, val_loader, counts


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    per_class_total = np.zeros(NUM_CLASSES, dtype=int)
    per_class_correct = np.zeros(NUM_CLASSES, dtype=int)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        for yi, pi in zip(y.cpu().numpy(), pred.cpu().numpy()):
            per_class_total[yi] += 1
            if yi == pi:
                per_class_correct[yi] += 1
    return loss_sum / max(total, 1), correct / max(total, 1), per_class_correct, per_class_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_loader, val_loader, counts = make_loaders(args.batch_size)

    model = SignNet(num_classes=NUM_CLASSES).to(device)
    print(f"model params: {count_params(model):,}")

    # 类别不均衡时, 用样本数倒数当权重, 避免少样本类被忽略
    class_w = 1.0 / np.maximum(counts, 1)
    class_w = class_w / class_w.mean()
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_w_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    log_path = CKPT_DIR / "train_log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.time()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0
            pbar = tqdm(train_loader, desc=f"ep {epoch:03d}/{args.epochs}", leave=False)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * x.size(0)
                tr_correct += (logits.argmax(1) == y).sum().item()
                tr_total += x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.3f}")
            scheduler.step()

            tr_loss /= max(tr_total, 1)
            tr_acc = tr_correct / max(tr_total, 1)
            va_loss, va_acc, pc_c, pc_t = evaluate(model, val_loader, device, criterion)

            print(f"ep {epoch:03d} | tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} "
                  f"| va_loss {va_loss:.3f} va_acc {va_acc:.3f} | {time.time()-t0:.1f}s")
            w.writerow([epoch, tr_loss, tr_acc, va_loss, va_acc])
            f.flush()

            torch.save({"model": model.state_dict(),
                        "vocab": VOCAB,
                        "epoch": epoch}, CKPT_DIR / "last.pt")
            if va_acc > best_acc:
                best_acc = va_acc
                torch.save({"model": model.state_dict(),
                            "vocab": VOCAB,
                            "epoch": epoch,
                            "val_acc": va_acc}, CKPT_DIR / "best.pt")
                print(f"  [best] saved  (val_acc={va_acc:.3f})")

    # 最终每类准确率报告 (用 best 权重重新评估)
    print("\n=== final per-class (val) ===")
    ckpt = torch.load(CKPT_DIR / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    _, _, pc_c, pc_t = evaluate(model, val_loader, device, criterion)
    for i in range(NUM_CLASSES):
        acc = pc_c[i] / pc_t[i] if pc_t[i] else float("nan")
        print(f"  [{i}] {VOCAB[i]:8s}  {pc_c[i]}/{pc_t[i]} = {acc:.2f}")

    print(f"\nbest val acc: {best_acc:.3f}")
    print(f"checkpoint:   {CKPT_DIR/'best.pt'}")


if __name__ == "__main__":
    main()
