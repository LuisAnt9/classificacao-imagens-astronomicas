"""
Script de avaliação do modelo treinado no conjunto de teste.
Gera: matriz de confusão, relatório de métricas (precisão, recall, F1) e
amostras visuais de acertos/erros.

Uso:
    python src/evaluate.py --model scratch
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import build_model


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(cm, class_names, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe real")
    ax.set_title("Matriz de Confusão")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Matriz de confusão salva em: {out_path}")


def plot_sample_predictions(model, dataset, class_names, device, out_path, n=12):
    import torchvision.transforms as T

    model.eval()
    indices = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.flatten()

    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )

    for ax, idx in zip(axes, indices):
        image, label = dataset[idx]
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = torch.argmax(output, dim=1).item()

        img_show = inv_normalize(image).permute(1, 2, 0).clamp(0, 1).numpy()
        ax.imshow(img_show)
        real_name = class_names[label]
        pred_name = class_names[pred]
        color = "green" if pred == label else "red"
        ax.set_title(f"Real: {real_name}\nPrevisto: {pred_name}", color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Amostras de predições salvas em: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="scratch",
                         choices=["scratch", "transfer"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--weights_dir", type=str, default="weights")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.join(args.weights_dir, f"best_model_{args.model}.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    _, _, test_loader, _ = get_dataloaders(
        data_dir=args.data_dir, img_size=img_size, batch_size=args.batch_size)

    model = build_model(args.model, num_classes=len(class_names),
                         img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    preds, labels = get_predictions(model, test_loader, device)

    print("\n=== Relatório de Classificação (conjunto de teste) ===")
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(report)

    with open(os.path.join(args.results_dir, f"classification_report_{args.model}.txt"), "w") as f:
        f.write(report)

    cm = confusion_matrix(labels, preds)
    plot_confusion_matrix(cm, class_names,
                           os.path.join(args.results_dir, f"confusion_matrix_{args.model}.png"))

    plot_sample_predictions(model, test_loader.dataset, class_names, device,
                             os.path.join(args.results_dir, f"sample_predictions_{args.model}.png"))

    accuracy = (preds == labels).mean()
    print(f"\nAcurácia final no conjunto de teste: {accuracy:.4f}")


if __name__ == "__main__":
    main()
