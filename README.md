# Towards the Detection of Diffusion Model Deepfakes

This this the official repository for [Towards the Detection of Diffusion Model Deepfakes](https://arxiv.org/abs/2210.14571).

## Usage
Code will be released soon.

## Dataset
The dataset used in this work is hosted on [Zenodo](https://zenodo.org/record/7528113).
In total, the dataset contains 50k samples (256x256) for each of the following generators, divided into train, validation, and test set (39k/1k/10k).

| GANs | DMs |
| :--- | :-- |
| ProGAN | DDPM |
| StyleGAN | IDDPM |
| ProjectedGAN | ADM |
| Diffusion-StyleGAN2 | PNDM |
| Diffusion-ProjectedGAN | LDM |

The test set additionally contains 10k real images from [LSUN Bedroom](https://github.com/fyu/lsun) and can be used as-is.
Due to storage limitations, the train and validation set contain only generated images. Real images, required for training, have to be added according to the [instructions given below](#train-and-validation-set).

### Test Set
The test set can be downloaded from [Zenodo](https://zenodo.org/record/7528113). To validate that all images were downloaded correctly, you can run the following command and compare the hash value (computation takes some time).
```
$ find diffusion_model_deepfakes_lsun_bedroom/test/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
68bd7326113b9cde3cd4eadf1e10e5fca77df751  -
```

### Train and Validation Set
To recreate the train and validation set, first download both zip-archives from [Zenodo](https://zenodo.org/record/7528113) and extract them in the same directory. Then, download the entire LSUN Bedroom training data in LMDB format according to the instructions in the [LSUN repository](https://github.com/fyu/lsun). Finally, run [recreate_train_data.py](recreate_train_data.py) (requires packages [lmbd](https://lmdb.readthedocs.io/en/release/) and [Pillow](https://pillow.readthedocs.io/en/stable/)):
```
$ python recreate_train_data.py path/to/bedroom_train_lmbd path/to/diffusion_model_deepfakes_lsun_bedroom
```
The script will distribute real images such that for each generator there is a unique set of 40k real images (39k train, 1k validation) with the correct dimensions. Note that due to the large amount of images this will take some time. For each generator, real images will be in `0_real`, fake images in `1_fake`.

To validate that all images were downloaded and processed correctly, you can run the following commands and compare the hash values (computation takes some time).
```
$ find diffusion_model_deepfakes_lsun_bedroom/train/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
10fd255e5669e76b4e9bde93f672733a5dfa92fa  -

$ find diffusion_model_deepfakes_lsun_bedroom/val/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
c836fe592e49ab630ccc30675694ab32aff8e1a3  -
```
