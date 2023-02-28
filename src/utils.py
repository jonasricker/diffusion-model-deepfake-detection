from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets.folder import IMG_EXTENSIONS


def estimate_batch_size(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Estimate the maximum batch size (power of 2) for a given model and sample."""
    torch.cuda.empty_cache()
    mem_before = torch.cuda.mem_get_info()[0]
    with torch.no_grad():
        model(
            torch.tile(sample, dims=(4, 1, 1, 1)).cuda()
        )  # batch of four gives more reliable results
    mem_after = torch.cuda.mem_get_info()[0]
    diff = mem_before - mem_after
    max_samples = int(mem_before / (diff / 4))
    smaller_po2 = 2 ** (max_samples.bit_length() - 1)

    return smaller_po2


def has_same_sized_images(directory: Path) -> bool:
    """Check whether all images in directory have same dimensions."""
    size = None
    for file in directory.iterdir():
        if file.suffix in IMG_EXTENSIONS:
            if size is None:
                size = Image.open(file).size
            else:
                if Image.open(file).size != size:
                    return False
    return True
