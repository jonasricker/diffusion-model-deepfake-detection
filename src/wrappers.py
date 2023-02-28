import logging
import os
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, ToTensor
from tqdm import tqdm

from detectors.gragnaniello2021.resnet50nodown import (
    LIMIT_SIZE,
    LIMIT_SLIDE,
    resnet50nodown,
)
from detectors.mandelli2022.utils import architectures
from detectors.mandelli2022.utils.python_patch_extractor.PatchExtractor import (
    PatchExtractor,
)
from detectors.wang2020.networks.resnet import resnet50
from src.data import SingleClassImageFolder
from src.utils import estimate_batch_size, has_same_sized_images

logger = logging.getLogger(__name__)


class PredictorWrapper:
    def __init__(
        self,
        name: str,
        model_path: Union[str, Path],
        setup_func: Callable,
        pred_func: Optional[Callable] = None,
        cuda: bool = True,
    ) -> None:
        self.name = name
        self.model_path = Path(model_path)
        self.setup_func = setup_func
        self.pred_func = pred_func
        self.cuda = cuda

        self.model = None
        self.transform = None

    def setup(self) -> None:
        self.model, self.transform = self.setup_func(self.model_path, self.cuda)
        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def predict_dir(
        self,
        directory: Union[str, Path],
        num_workers: int = 0,
        batch_size: Optional[int] = None,
    ) -> Tuple[List, List]:
        if self.model is None:
            raise Exception("You have to call 'setup' before predicting.")
        directory = Path(directory)
        dataset = SingleClassImageFolder(root=directory, transform=self.transform)

        if batch_size is None:
            if self.name == "mandelli2022":
                print(
                    "Cannot determine optimal batch size for Mandelli2022, setting"
                    " to 8."
                )
                batch_size = 8
            else:
                batch_size = (
                    estimate_batch_size(self.model, dataset[0][0])
                    if has_same_sized_images(directory)
                    else 1
                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        logger.info(
            f"samples: {len(dataset)}, image_size: {list(dataset[0][0].shape[-2:])},"
            f" batch_size: {dataloader.batch_size}, num_workers: {num_workers}"
        )

        scores, paths = [], []
        with torch.no_grad():
            for img, path in tqdm(dataloader, desc=self.name):
                if self.cuda:
                    img = img.cuda()
                pred = (
                    self.pred_func(self.model, img)
                    if self.pred_func is not None
                    else self.model(img)
                )
                scores.extend(pred.flatten().tolist())
                paths.extend(path)
        return scores, paths


wang2020_transform = Compose(
    [ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


def wang2020_setup(model_path: Path, cuda: bool) -> Tuple[torch.nn.Module, Callable]:
    model = resnet50(num_classes=1)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    if cuda:
        model = model.cuda()
    return model, wang2020_transform


def wang2020_pred(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return model(batch).sigmoid()


def gragnaniello2021_setup(
    model_path: Path, cuda: bool
) -> Tuple[torch.nn.Module, Callable]:
    if cuda:
        model = resnet50nodown("cuda:0", model_path)
    else:
        model = resnet50nodown("cpu", model_path)
    return model, wang2020_transform


def gragnaniello2021_pred(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = []
    if min(batch.shape[-2:]) > LIMIT_SIZE:
        for img in batch:
            list_logit = list()
            list_weight = list()
            for index0 in range(0, img.shape[-2], LIMIT_SLIDE):
                for index1 in range(0, img.shape[-1], LIMIT_SLIDE):
                    clip = img[
                        ...,
                        index0 : min(index0 + LIMIT_SLIDE, img.shape[-2]),
                        index1 : min(index1 + LIMIT_SLIDE, img.shape[-1]),
                    ]
                    logit = (
                        torch.squeeze(model(clip.cuda()[None, :, :, :])).cpu().numpy()
                    )
                    weight = clip.shape[-2] * clip.shape[-1]
                    list_logit.append(logit)
                    list_weight.append(weight)

            logit = np.mean(np.asarray(list_logit) * np.asarray(list_weight)) / np.mean(
                list_weight
            )
            logits.append(logit)
        return torch.Tensor(logits)
    else:
        return model(batch)


def mandelli2022_setup(
    model_path: Path, cuda: bool
) -> Tuple[List[torch.nn.Module], Callable]:
    device = torch.device("cuda:0") if cuda else torch.device("cpu")
    weights_path_list = [os.path.join(model_path, f"method_{x}.pth") for x in "ABCDE"]

    nets = []
    for i, letter in enumerate("ABCDE"):
        # Instantiate and load network
        network_class = getattr(architectures, "EfficientNetB4")
        net = network_class(n_classes=2, pretrained=False).eval().to(device)
        print(f"Loading model {letter}...")
        state_tmp = torch.load(weights_path_list[i], map_location="cpu")

        if "net" not in state_tmp.keys():
            state = OrderedDict({"net": OrderedDict()})
            [
                state["net"].update({"model.{}".format(k): v})
                for k, v in state_tmp.items()
            ]
        else:
            state = state_tmp
        incomp_keys = net.load_state_dict(state["net"], strict=True)
        print(incomp_keys)
        print("Model loaded!\n")

        nets += [net]

    net_normalizer = net.get_normalizer()  # pick normalizer from last network
    transform = [
        ToTensor(),
        net_normalizer,
    ]
    return nets, Compose(transform)


def mandelli2022_pred(
    model: List[torch.nn.Module], batch: torch.Tensor
) -> torch.Tensor:
    cropper = RandomCrop(size=128)

    img_net_scores = []
    for net_idx, net in enumerate(model):
        if net_idx == 0:
            # only for detector A, extract N = 200 random patches per image
            patch_list = [cropper(batch) for _ in range(200)]
            patch_tensor = torch.stack(patch_list, dim=1).to(batch.device)

        else:
            # for detectors B, C, D, E, extract patches aligned with the 8 x 8 pixel grid:
            # we want more or less 200 patches per img
            stride_0 = ((((batch.shape[-2] - 128) // 20) + 7) // 8) * 8
            stride_1 = (((batch.shape[-1] - 128) // 10 + 7) // 8) * 8
            pe = PatchExtractor(dim=(1, 3, 128, 128), stride=(1, 3, stride_0, stride_1))
            patches = torch.from_numpy(pe.extract(batch.cpu().numpy()))
            patch_list = list(
                patches.reshape(
                    (
                        len(batch),
                        patches.shape[1]
                        * patches.shape[2]
                        * patches.shape[3]
                        * patches.shape[4],
                        3,
                        128,
                        128,
                    )
                )
            )
            patch_tensor = torch.stack(patch_list, dim=0).to(batch.device)

        # Compute scores
        n_patches = patch_tensor.shape[1]
        patch_tensor = patch_tensor.reshape(len(batch) * n_patches, 3, 128, 128)
        with torch.no_grad():
            patch_scores = net(patch_tensor).cpu().numpy()
        patch_scores = patch_scores.reshape(len(batch), n_patches, 2)
        patch_predictions = np.argmax(patch_scores, axis=2)

        scores = []
        for score, prediction in zip(patch_scores, patch_predictions):
            maj_voting = np.any(patch_predictions).astype(int)
            scores_maj_voting = score[:, maj_voting]
            scores.append(
                np.nanmax(scores_maj_voting)
                if maj_voting == 1
                else -np.nanmax(scores_maj_voting)
            )
        scores = torch.Tensor(scores)
        img_net_scores.append(scores)

    # final score is the average among the 5 scores returned by the detectors
    img_score = torch.stack(img_net_scores, dim=1).mean(dim=1)
    return img_score
