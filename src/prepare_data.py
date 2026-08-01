"""
Script para organizar o dataset bruto baixado do Kaggle
(Astronomy Image Classification Dataset) na estrutura esperada pelo projeto:

    data/train/<classe>/*.jpg
    data/val/<classe>/*.jpg
    data/test/<classe>/*.jpg

Uso:
    python src/prepare_data.py --raw_dir "space images" --out_dir data
"""

import argparse
import os
import random
import shutil

# Mapeia os nomes originais das pastas do dataset do Kaggle para nomes
# de classe limpos. Ajuste caso os nomes das pastas baixadas sejam diferentes.
FOLDER_MAPPING = {
    "constellation - Google Search": "constellations",
    "constellations": "constellations",
    "cosmos space - Google Search": "cosmos",
    "cosmos": "cosmos",
    "galaxies - Google Search": "galaxies",
    "galaxies": "galaxies",
    "nebula - Google Search": "nebulae",
    "nebulae": "nebulae",
    "planets - Google Search": "planets",
    "planets": "planets",
    "stars - Google Search": "stars",
    "stars": "stars",
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True,
                         help="Pasta com as subpastas originais baixadas do Kaggle")
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    for folder_name in os.listdir(args.raw_dir):
        folder_path = os.path.join(args.raw_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        class_name = FOLDER_MAPPING.get(folder_name, folder_name.lower().strip())

        files = [f for f in os.listdir(folder_path)
                 if os.path.splitext(f)[1].lower() in VALID_EXTS]
        random.shuffle(files)

        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)

        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }

        for split, flist in splits.items():
            out_class_dir = os.path.join(args.out_dir, split, class_name)
            os.makedirs(out_class_dir, exist_ok=True)
            for i, f in enumerate(flist):
                ext = os.path.splitext(f)[1].lower()
                shutil.copy(
                    os.path.join(folder_path, f),
                    os.path.join(out_class_dir, f"{class_name}_{i:04d}{ext}")
                )

        print(f"{class_name}: "
              f"train={len(splits['train'])} "
              f"val={len(splits['val'])} "
              f"test={len(splits['test'])}")

    print(f"\nDataset organizado em: {args.out_dir}/")


if __name__ == "__main__":
    main()
