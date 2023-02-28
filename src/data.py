import os
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from torchvision.datasets.folder import (
    ImageFolder,
    default_loader,
    has_file_allowed_extension,
)

from src.image import center_crop
from src.optimization import with_caching


def array_from_imgdir(
    directory: Path,
    crop_size: int = 256,
    grayscale: bool = False,
    num_samples: int = None,
    num_workers: int = 1,
) -> np.ndarray:
    """
    Given a directory, load all (or num_samples if specified) images and return them as a numpy array.
    Optionally crop and/or apply grayscale conversion.
    """
    paths = []
    for path in directory.iterdir():
        if path.suffix.lower() == ".png":
            paths.append(path)
        if num_samples is not None and len(paths) == num_samples:
            break
    if num_samples and len(paths) < num_samples:
        warnings.warn(f"Found only {len(paths)} images instead of {num_samples}.")

    if grayscale:

        def loader(path):
            return np.array(Image.open(path).convert("L"))

    else:

        def loader(path):
            return np.array(Image.open(path))

    if num_workers == 1:
        array = np.array(list(map(loader, paths)))
    else:
        array = np.array(
            Parallel(n_jobs=num_workers)(delayed(loader)(path) for path in paths)
        )
    if array.shape[1:] != (crop_size, crop_size):
        print(f"Cropping from {array.shape[1:]} to {crop_size, crop_size}.")
    array = center_crop(array, size=crop_size)
    array = array / 127.5 - 1  # scale to [-1, 1]
    return array


@with_caching(keys=["img_dir", "func", "crop_size", "grayscale"])
def apply_to_imgdir(
    img_dir: Path,
    func: Callable,
    crop_size: int = 256,
    grayscale: bool = False,
    num_samples: int = None,
    num_workers: int = 1,
) -> np.ndarray:
    """Convenience function to load images from directory into numpy array and apply function to it."""
    return func(
        array_from_imgdir(
            directory=img_dir,
            grayscale=grayscale,
            crop_size=crop_size,
            num_samples=num_samples,
            num_workers=num_workers,
        )
    )


class SingleClassImageFolder(ImageFolder):
    """
    Similar to ImageFolder, but all images (directly in root or in subfolders) are expected to be of a single class.
    Also returns the filename instead of a label.
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(root, transform, target_transform, loader, is_valid_file)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        return [""], {"": 0}

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self.find_classes(directory)
        elif not class_to_idx:
            raise ValueError(
                "'class_to_index' must have at least one entry to collect any samples."
            )

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError(
                "Both extensions and is_valid_file cannot be None or not None at the"
                " same time"
            )

        if extensions is not None:

            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        target_dir = directory
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                if is_valid_file(fname):
                    path = os.path.join(root, fname)
                    item = path, 0
                    instances.append(item)

        return instances

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, path
