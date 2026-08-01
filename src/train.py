"""
Script de treinamento do modelo de classificação de imagens astronômicas.

Uso:
    python src/train.py --model scratch --epochs 15
    python src/train.py --model transfer --epochs 10   (recomendado com GPU)
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="scratch",
                        choices=["scratch", "transfer"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir, img_size=args.img_size, batch_size=args.batch_size)
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(args.model, num_classes=len(class_names),
                         img_size=args.img_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3)

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    best_val_acc = 0.0
    best_path = os.path.join(args.weights_dir, f"best_model_{args.model}.pt")

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "model_type": args.model,
                "img_size": args.img_size,
                "val_acc": val_acc,
            }, best_path)

    elapsed = time.time() - start_time
    print(f"\nTreinamento concluído em {elapsed/60:.1f} min. "
          f"Melhor acurácia de validação: {best_val_acc:.4f}")
    print(f"Pesos salvos em: {best_path}")

    # Salva histórico
    with open(os.path.join(args.results_dir, f"history_{args.model}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Gráfico de loss/acurácia
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history["train_loss"], label="Treino")
    axes[0].plot(history["val_loss"], label="Validação")
    axes[0].set_title("Loss por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Treino")
    axes[1].plot(history["val_acc"], label="Validação")
    axes[1].set_title("Acurácia por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, f"training_curves_{args.model}.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Gráfico de treinamento salvo em: {plot_path}")


if __name__ == "__main__":
    main()
