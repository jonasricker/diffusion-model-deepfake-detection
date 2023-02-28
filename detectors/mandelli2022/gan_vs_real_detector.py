import os
from collections import OrderedDict

import numpy as np
import torch

torch.manual_seed(21)
import random

random.seed(21)
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')
import albumentations as A
import albumentations.pytorch as Ap
from utils import architectures
from utils.python_patch_extractor.PatchExtractor import PatchExtractor
from PIL import Image

import argparse


class Detector:
    def __init__(self):

        # You need to download the weight.zip file from here https://www.dropbox.com/s/g1z2u8wl6srjh6v/weigths.zip?dl=0
        # and uncompress it into the main folder.
        self.weights_path_list = [os.path.join('weights', f'method_{x}.pth') for x in 'ABCDE']

        # GPU configuration if available
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        self.nets = []
        for i, l in enumerate('ABCDE'):
            # Instantiate and load network
            network_class = getattr(architectures, 'EfficientNetB4')
            net = network_class(n_classes=2, pretrained=False).eval().to(self.device)
            print(f'Loading model {l}...')
            state_tmp = torch.load(self.weights_path_list[i], map_location='cpu')

            if 'net' not in state_tmp.keys():
                state = OrderedDict({'net': OrderedDict()})
                [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
            else:
                state = state_tmp
            incomp_keys = net.load_state_dict(state['net'], strict=True)
            print(incomp_keys)
            print('Model loaded!\n')

            self.nets += [net]

        net_normalizer = net.get_normalizer()  # pick normalizer from last network
        transform = [
            A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
            Ap.transforms.ToTensorV2()
        ]
        self.trans = A.Compose(transform)
        self.cropper = A.RandomCrop(width=128, height=128, always_apply=True, p=1.)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def synth_real_detector(self, img_path: str, n_patch: int = 200):

        # Load image:
        img = np.asarray(Image.open(img_path))

        # Opt-out if image is non conforming
        if img.shape == ():
            print('{} None dimension'.format(img_path))
            return None
        if img.shape[0] < 128 or img.shape[1] < 128:
            print('Too small image')
            return None
        if img.ndim != 3:
            print('RGB images only')
            return None
        if img.shape[2] > 3:
            print('Omitting alpha channel')
            img = img[:, :, :3]

        print('Computing scores...')
        img_net_scores = []
        for net_idx, net in enumerate(self.nets):

            if net_idx == 0:

                # only for detector A, extract N = 200 random patches per image
                patch_list = [self.cropper(image=img)['image'] for _ in range(n_patch)]

            else:

                # for detectors B, C, D, E, extract patches aligned with the 8 x 8 pixel grid:
                # we want more or less 200 patches per img
                stride_0 = ((((img.shape[0] - 128) // 20) + 7) // 8) * 8
                stride_1 = (((img.shape[1] - 128) // 10 + 7) // 8) * 8
                pe = PatchExtractor(dim=(128, 128, 3), stride=(stride_0, stride_1, 3))
                patches = pe.extract(img)
                patch_list = list(patches.reshape((patches.shape[0] * patches.shape[1], 128, 128, 3)))

            # Normalization
            transf_patch_list = [self.trans(image=patch)['image'] for patch in patch_list]

            # Compute scores
            transf_patch_tensor = torch.stack(transf_patch_list, dim=0).to(self.device)
            with torch.no_grad():
                patch_scores = net(transf_patch_tensor).cpu().numpy()
                patch_predictions = np.argmax(patch_scores, axis=1)

            maj_voting = np.any(patch_predictions).astype(int)
            scores_maj_voting = patch_scores[:, maj_voting]
            img_net_scores.append(np.nanmax(scores_maj_voting) if maj_voting == 1 else -np.nanmax(scores_maj_voting))

        # final score is the average among the 5 scores returned by the detectors
        img_score = np.mean(img_net_scores)

        return img_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', help='Path to the test image', required=True)
    args = parser.parse_args()

    img_path = args.img_path

    detector = Detector()
    score = detector.synth_real_detector(img_path)

    print('Image Score: {}'.format(score))

    return 0


if __name__ == '__main__':
    main()
