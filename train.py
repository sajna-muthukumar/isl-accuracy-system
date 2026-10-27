import argparse, json, os, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    train_dir, val_dir = data / "train", data / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise SystemExit("Expected data/train/<class>/ and data/val/<class>/ folders. See README.md.")

    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((128,128)), transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tf)
    if len(train_ds.classes) < 2:
        raise SystemExit("Need at least two class folders to train a classifier.")
    if train_ds.classes != val_ds.classes:
        raise SystemExit(f"Train/validation classes differ: {train_ds.classes} vs {val_ds.classes}")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Small CNN suitable as a transparent baseline.
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, len(train_ds.classes))
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/classes.json", "w", encoding="utf-8") as f:
        json.dump(train_ds.classes, f, indent=2)

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); train_loss=0.0; train_correct=0; train_total=0
        for x,y in train_dl:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            logits=model(x); loss=criterion(logits,y)
            loss.backward(); optimizer.step()
            train_loss += loss.item()*y.size(0)
            train_correct += (logits.argmax(1)==y).sum().item(); train_total += y.size(0)
        model.eval(); val_correct=0; val_total=0; val_loss=0.0
        with torch.no_grad():
            for x,y in val_dl:
                x,y=x.to(device),y.to(device)
                logits=model(x); loss=criterion(logits,y)
                val_loss += loss.item()*y.size(0)
                val_correct += (logits.argmax(1)==y).sum().item(); val_total += y.size(0)
        train_acc=train_correct/max(train_total,1); val_acc=val_correct/max(val_total,1)
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss={train_loss/max(train_total,1):.4f} "
              f"| train_acc={train_acc:.4f} | val_loss={val_loss/max(val_total,1):.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_val:
            best_val=val_acc
            torch.save({"state_dict":model.state_dict(),"num_classes":len(train_ds.classes),
                        "input_size":128}, "artifacts/best_model.pt")
    print(f"Training complete. Best validation accuracy: {best_val:.4f}")
    print("Checkpoint: artifacts/best_model.pt")
if __name__ == "__main__":
    main()
