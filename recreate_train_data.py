"""Add real LSUN images to the downloadable dataset."""
import argparse
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image

MODELS = [
    "ADM",
    "DDPM",
    "Diff-ProjectedGAN",
    "Diff-StyleGAN2",
    "IDDPM",
    "LDM",
    "PNDM",
    "ProGAN",
    "ProjectedGAN",
    "StyleGAN",
]


def main(lmdb_path: Path, img_path: Path, validation_only: bool):
    train_dir = img_path / "train"
    val_dir = img_path / "val"

    # construct distribution plan
    start = 10000  # first 10000 images are used in test dataset
    plan = {}
    for model in MODELS:
        plan.update({start: (train_dir / model / "0_real", 39000)})
        start += 39000
        plan.update({start: (val_dir / model / "0_real", 1000)})
        start += 1000
    plan.update({start: ("STOP", 0)})

    if not validation_only:
        # export images
        env = lmdb.open(
            str(lmdb_path),
            map_size=1099511627776,
            max_readers=100,
            readonly=True,
        )
        with env.begin(write=False) as txn:
            cursor = txn.cursor()
            for i, (key, val) in enumerate(cursor):
                if i < 10000:  # skip first 10000 images used in test dataset
                    continue
                if i in plan:
                    if plan[i][0] == "STOP":
                        break
                    else:
                        out = plan[i][0]
                        print(f"Exporting {plan[i][1]} images to {out}.")
                        out.mkdir()

                # load image and center crop t0 256x256 pixels
                img = Image.open(BytesIO(val))
                w, h = img.size
                assert w == 256 or h == 256
                center_x, center_y = w // 2, h // 2
                img = img.crop(
                    box=(
                        center_x - 128,
                        center_y - 128,
                        center_x + 128,
                        center_y + 128,
                    )
                )
                img.save(out / f"{i:08}.png")

    # validation
    print("Validating...")
    for start, (directory, amount) in plan.items():
        if directory == "STOP":
            continue
        assert (directory / f"{start:08}.png").exists()
        assert (directory / f"{start+amount-1:08}.png").exists()
        assert len(list(directory.iterdir())) == amount
    print("Done!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "lmdb_path", type=Path, help="Path to LSUN Bedroom Train LMDB directory."
    )
    parser.add_argument(
        "img_path", type=Path, help="Path to 'diffusion_model_deepfakes_lsun_bedroom'."
    )
    parser.add_argument("--validation-only", action="store_true")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(**vars(parse_args()))
