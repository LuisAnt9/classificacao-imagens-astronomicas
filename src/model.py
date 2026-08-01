"""
Definições de modelo para classificação de imagens astronômicas.

Duas opções disponíveis:
  - SimpleCNN: rede convolucional treinada do zero (funciona em qualquer
    ambiente, inclusive sem acesso à internet).
  - build_transfer_model(): ResNet18 pré-treinada na ImageNet com a camada
    final substituída (transfer learning). Requer internet na primeira
    execução para baixar os pesos pré-treinados (recomendado rodar em
    ambiente com GPU, como Google Colab).
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SimpleCNN(nn.Module):
    """CNN simples treinada do zero, adequada para datasets pequenos."""

    def __init__(self, num_classes: int, img_size: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            # Bloco 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> img_size/2

            # Bloco 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> img_size/4

            # Bloco 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> img_size/8

            # Bloco 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),  # -> saída fixa 4x4 independente do img_size
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_transfer_model(num_classes: int, freeze_backbone: bool = True):
    """Constrói um modelo de transfer learning baseado em ResNet18.

    ATENÇÃO: precisa de acesso à internet na primeira execução para baixar
    os pesos pré-treinados da ImageNet. Recomendado para Google Colab.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Substitui a camada final para o número de classes do problema
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model, weights.transforms()


def build_model(model_type: str, num_classes: int, img_size: int = 128):
    """Fábrica de modelos. model_type: 'scratch' ou 'transfer'."""
    if model_type == "scratch":
        return SimpleCNN(num_classes=num_classes, img_size=img_size)
    elif model_type == "transfer":
        model, _ = build_transfer_model(num_classes=num_classes)
        return model
    else:
        raise ValueError(f"model_type desconhecido: {model_type}")


if __name__ == "__main__":
    # Teste rápido de shape
    m = SimpleCNN(num_classes=6)
    x = torch.randn(2, 3, 128, 128)
    out = m(x)
    print("SimpleCNN output shape:", out.shape)
