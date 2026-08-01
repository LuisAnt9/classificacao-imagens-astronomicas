"""
Script de inferência: classifica uma única imagem usando o modelo treinado.

Uso:
    python src/predict.py --image caminho/para/imagem.jpg --model scratch
"""

import argparse

import torch
from PIL import Image

from dataset import get_transforms
from model import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="scratch",
                         choices=["scratch", "transfer"])
    parser.add_argument("--weights_dir", type=str, default="weights")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = f"{args.weights_dir}/best_model_{args.model}.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    model = build_model(args.model, num_classes=len(class_names),
                         img_size=img_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = get_transforms(img_size)
    image = Image.open(args.image).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print(f"Imagem: {args.image}")
    print(f"Classe prevista: {class_names[pred_idx]} "
          f"(confiança: {probs[pred_idx]*100:.2f}%)")
    print("\nProbabilidades por classe:")
    for name, p in sorted(zip(class_names, probs.tolist()),
                           key=lambda x: -x[1]):
        print(f"  {name:15s}: {p*100:5.2f}%")


if __name__ == "__main__":
    main()
