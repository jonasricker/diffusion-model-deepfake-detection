# Towards the Detection of Diffusion Model Deepfakes

This is the official repository for [Towards the Detection of Diffusion Model Deepfakes](https://arxiv.org/abs/2210.14571).

# Table of Contents
- [Code](#code)
    - [Detector Evaluation](#detector-evaluation)
    - [Feature Space Analysis](#feature-space-analysis)
    - [Spectrum Evolutions](#spectrum-evolutions)
    - [Fakeness Ratings](#fakeness-ratings)
- [Checkpoints](#checkpoints)
- [Dataset](#dataset)

# Code
We provide the source code and instructions on how to recreate the results in the paper. The code is tested with Python 3.8. To install the required packages run `pip install -r requirements.txt`. You probably should install the [version of PyTorch matching your system](https://pytorch.org/get-started/locally/).

The commands below expect a working directory which contains data, models, and to which results will be written. To run the commandas as-is, save the path to this directory in a variable by executing `WORKDIR=path/to/working/directory`.

Instructions for downloading the [Checkpoints](#checkpoints) and [Dataset](#dataset) are given in the corresponding sections. Your working directory should have the following structure:
```bash
workdir/
├── data
│   └── diffusion_model_deepfakes_lsun_bedroom
│       ├── test
│       │   ├── ADM
│       │   ├── ...
│       ├── train  # only required for training your own models
│       │   ├── ADM
│       │   ├── ...
│       └── val  # only required for training your own models
│           ├── ADM
│           ├── ...
└── models
    ├── gragnaniello2021
    │   ├── gandetection_resnet50nodown_progan.pth
    │   └── gandetection_resnet50nodown_stylegan2.pth
    ├── mandelli2022
    │   ├── method_A.pth
    │   ├── method_B.pth
    │   ├── method_C.pth
    │   ├── method_D.pth
    │   └── method_E.pth
    └── wang2020
        ├── blur_jpg_prob0.1.pth
        ├── blur_jpg_prob0.5.pth
        ├── finetuning
        │   ├── ADM
        │   │   └── model_epoch_best.pth
        │   ├── ...
        └── scratch
            ├── ADM
            │   └── model_epoch_best.pth
            ├── ...
```

Calling any of the commands below with `-h` prints further information on the arguments.

## Detector Evaluation
To recreate the results in Table 2 run
```bash
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --predictors wang2020 gragnaniello2021 mandelli2022 --wang2020-model-path wang2020/blur_jpg_prob0.5.pth wang2020/blur_jpg_prob0.1.pth --gragnaniello2021-model-path gragnaniello2021/gandetection_resnet50nodown_progan.pth gragnaniello2021/gandetection_resnet50nodown_stylegan2.pth --metric AUROC PD@5% PD@1%
```

To recreate Figures 1a and 1b run
```bash
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment finetuning --predictors wang2020 --wang2020-model-path wang2020/finetuning/ProGAN wang2020/finetuning/StyleGAN wang2020/finetuning/ProjectedGAN wang2020/finetuning/Diff-StyleGAN2 wang2020/finetuning/Diff-ProjectedGAN wang2020/finetuning/DDPM wang2020/finetuning/IDDPM wang2020/finetuning/ADM wang2020/finetuning/PNDM wang2020/finetuning/LDM wang2020/finetuning/GAN wang2020/finetuning/DM wang2020/finetuning/All --output cfm --metric AUROC
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment finetuning --predictors wang2020 --wang2020-model-path wang2020/finetuning/ProGAN wang2020/finetuning/StyleGAN wang2020/finetuning/ProjectedGAN wang2020/finetuning/Diff-StyleGAN2 wang2020/finetuning/Diff-ProjectedGAN wang2020/finetuning/DDPM wang2020/finetuning/IDDPM wang2020/finetuning/ADM wang2020/finetuning/PNDM wang2020/finetuning/LDM wang2020/finetuning/GAN wang2020/finetuning/DM wang2020/finetuning/All --output cfm --metric PD@1%
```

To recreate Figures 8a and 8b run
```bash
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment scratch --predictors wang2020 --wang2020-model-path wang2020/scratch/ProGAN wang2020/scratch/StyleGAN wang2020/scratch/ProjectedGAN wang2020/scratch/Diff-StyleGAN2 wang2020/scratch/Diff-ProjectedGAN wang2020/scratch/DDPM wang2020/scratch/IDDPM wang2020/scratch/ADM wang2020/scratch/PNDM wang2020/scratch/LDM wang2020/scratch/GAN wang2020/scratch/DM wang2020/scratch/All --output cfm --metric AUROC
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment scratch --predictors wang2020 --wang2020-model-path wang2020/scratch/ProGAN wang2020/scratch/StyleGAN wang2020/scratch/ProjectedGAN wang2020/scratch/Diff-StyleGAN2 wang2020/scratch/Diff-ProjectedGAN wang2020/scratch/DDPM wang2020/scratch/IDDPM wang2020/scratch/ADM wang2020/scratch/PNDM wang2020/scratch/LDM wang2020/scratch/GAN wang2020/scratch/DM wang2020/scratch/All --output cfm --metric PD@1%
```

To recreate the results in Tables 4 and 5 run
```bash
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment noise --predictors gragnaniello2021 --gragnaniello2021-model-path gragnaniello2021/gandetection_resnet50nodown_progan.pth --metric AUROC PD@5% PD@1% --perturbation noise
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment blur --predictors gragnaniello2021 --gragnaniello2021-model-path gragnaniello2021/gandetection_resnet50nodown_progan.pth --metric AUROC PD@5% PD@1% --perturbation blur
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment jpeg --predictors gragnaniello2021 --gragnaniello2021-model-path gragnaniello2021/gandetection_resnet50nodown_progan.pth --metric AUROC PD@5% PD@1% --perturbation jpeg
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM --experiment cropping --predictors gragnaniello2021 --gragnaniello2021-model-path gragnaniello2021/gandetection_resnet50nodown_progan.pth --metric AUROC PD@5% PD@1% --perturbation cropping

python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real DDPM IDDPM ADM PNDM LDM --experiment noise_ft_DM --predictors wang2020 --wang2020-model-path wang2020/finetuning/DM --metric AUROC PD@5% PD@1% --perturbation noise
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real DDPM IDDPM ADM PNDM LDM --experiment blur_ft_DM --predictors wang2020 --wang2020-model-path wang2020/finetuning/DM --metric AUROC PD@5% PD@1% --perturbation blur
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real DDPM IDDPM ADM PNDM LDM --experiment jpeg_ft_DM --predictors wang2020 --wang2020-model-path wang2020/finetuning/DM --metric AUROC PD@5% PD@1% --perturbation jpeg
python evaluate_detectors.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/models $WORKDIR/output --img-dirs Real DDPM IDDPM ADM PNDM LDM --experiment cropping_ft_DM --predictors wang2020 --wang2020-model-path wang2020/finetuning/DM --metric AUROC PD@5% PD@1% --perturbation cropping
```

## Feature Space Analysis
To extract the features from Gragnaniello2021 run
```bash
python detectors/gragnaniello2021/features.py --weights-path $WORKDIR/models/gragnaniello2021/gandetection_resnet50nodown_progan.pth --img-root $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --output-root $WORKDIR/output
python detectors/gragnaniello2021/features.py --weights-path $WORKDIR/models/gragnaniello2021/gandetection_resnet50nodown_stylegan2.pth --img-root $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --output-root $WORKDIR/output
```

To extract the features from Wang2020 using a modified version of the original script run
```bash
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/blur_jpg_prob0.1.pth --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/blur_jpg_prob0.5.pth --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/finetuning/All --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/finetuning/GAN --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/finetuning/DM --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/scratch/All --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/scratch/GAN --output_root $WORKDIR/output
python detectors/wang2020/demo_dir_features.py --dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test --model_path $WORKDIR/models/wang2020/scratch/DM --output_root $WORKDIR/output
```

To calculate the Maximum Mean Discrepancy (MMD) based on extracted features run
```bash
python feature_space_analysis/calculate_MMD.py -b $WORKDIR/output/features
```

The plots can then be recreated using the notebooks `plot_MMD.ipynb` and `plot_tSNE.ipynb`.

## Frequency Analysis
To recreate Figures 3a and 3b run
```bash
python frequency_analysis.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output fft_hp --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN --log --vmin 1e-5 --vmax 1e-1
python frequency_analysis.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output fft_hp --img-dirs Real DDPM IDDPM ADM PNDM LDM --log --vmin 1e-5 --vmax 1e-1
```

To compute DCT spectra, as in Figure 14, replace `fft_hp` with `dct`.

To recreate Figures 4a and 4b run
```bash
python frequency_analysis.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output density --img-dirs Real ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN --log --fraction 0.49 --zoom
python frequency_analysis.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output density --img-dirs Real DDPM IDDPM ADM PNDM LDM --log --fraction 0.49 --zoom
```

To recreate the results in Table 3 run
```bash
python logistic_regression.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom $WORKDIR/output --img-dirs ProGAN StyleGAN ProjectedGAN Diff-StyleGAN2 Diff-ProjectedGAN DDPM IDDPM ADM PNDM LDM
```
Note that to run this evaluation it is required to also have the `train` and `val` directories of the dataset.

## Spectrum Evolutions
Since we analyze the spectrum evolution using ADM, it is necessary to clone the [repository](https://github.com/openai/guided-diffusion) and install the package by calling `pip install .` from within. Also, download the LSUN Bedroom checkpoint and save it to `$WORKDIR/models/dhariwal2021/lsun_bedroom.pt`.

First, compute the reference spectrum from 10k real images by running
```bash
python spectrum_evolutions.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real $WORKDIR/output compute_reference lsun_bedroom_ref
```

To generate samples at different steps of the denoising process run
```bash
python sample_adm_denoising_process.py --model $WORKDIR/models/dhariwal2021/lsun_bedroom.pt --output_dir $WORKDIR/output/spectrum_vs_steps --store_within True --stop 1000 --step 10 --timestep_respacing 1000 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
python sample_adm_denoising_process.py --model $WORKDIR/models/dhariwal2021/lsun_bedroom.pt --output_dir $WORKDIR/output/spectrum_vs_steps --store_within True --stop 100 --step 1 --timestep_respacing 1000 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
```

To recreate Figures 5a and 5b run
```bash
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/denoising-1000_10 $WORKDIR/output --fraction 0.49 spectrum_vs_steps $WORKDIR/output/spectrum_vs_steps/lsun_bedroom_ref-10000_samples.npy
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/denoising-100_1 $WORKDIR/output --fraction 0.49 spectrum_vs_steps $WORKDIR/output/spectrum_vs_steps/lsun_bedroom_ref-10000_samples.npy
```

To plot the spectrum deviation relative to the diffusion process instead of the real spectrum, we first need to generate samples at different steps of the diffusion process by running
```bash
python sample_adm_diffusion_process.py --data_dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real --output_dir $WORKDIR/output/spectrum_vs_steps --stop 1000 --step 10 --timestep_respacing 1000 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
python sample_adm_diffusion_process.py --data_dir $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real --output_dir $WORKDIR/output/spectrum_vs_steps --stop 100 --step 1 --timestep_respacing 1000 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
```

Then, to recreate Figures 15a and 15b run
```bash
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/denoising-1000_10 $WORKDIR/output --fraction 0.49 denoising_vs_diffusion $WORKDIR/output/spectrum_vs_steps/diffusion-1000_10
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/denoising-100_1 $WORKDIR/output --fraction 0.49 denoising_vs_diffusion $WORKDIR/output/spectrum_vs_steps/diffusion-100_1
```

To recreate Figure 16 run
```bash
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/denoising-100_1 $WORKDIR/output --fraction 0.49  classification $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real
```

To sample images using different numbers of sampling steps (without and with DDIM) run
```bash
for step in 2 4 8 16 25 50 100 200 500 1000; do python sample_adm_denoising_process.py --model $WORKDIR/models/dhariwal2021/lsun_bedroom.pt --output_dir $WORKDIR/output/spectrum_vs_steps --timestep_respacing $step --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True; done
for step in 2 4 8 16 25 50 100 200 500 1000; do python sample_adm_denoising_process.py --model $WORKDIR/models/dhariwal2021/lsun_bedroom.pt --output_dir $WORKDIR/output/spectrum_vs_steps --use_ddim True --timestep_respacing $step --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True; done
```

Then, to recreate Figures 18a and 18b run
```bash
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/sampling_steps $WORKDIR/output --fraction 0.49 --ylabel "Denoising Steps" --all-labels spectrum_vs_steps $WORKDIR/output/spectrum_vs_steps/lsun_bedroom_ref-10000_samples.npy
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/sampling_steps_ddim $WORKDIR/output --fraction 0.49 --ylabel "Denoising Steps" --all-labels spectrum_vs_steps $WORKDIR/output/spectrum_vs_steps/lsun_bedroom_ref-10000_samples.npy
```

To recreate Figures 19a and 19b run
```bash
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/sampling_steps $WORKDIR/output --fraction 0.49 --ylabel "Denoising Steps" --all-labels classification $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real
python spectrum_evolutions.py $WORKDIR/output/spectrum_vs_steps/sampling_steps_ddim $WORKDIR/output --fraction 0.49 --ylabel "Denoising Steps" --all-labels classification $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test/Real
```

## Fakeness Ratings
To recreate Figures 11 and 12 run
```bash
python fakeness.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output/evaluate/cache $WORKDIR/output --img-dirs ADM DDPM Diff-ProjectedGAN Diff-StyleGAN2 IDDPM LDM PNDM ProGAN ProjectedGAN StyleGAN
python fakeness.py $WORKDIR/data/diffusion_model_deepfakes_lsun_bedroom/test $WORKDIR/output/evaluate/cache $WORKDIR/output --img-dirs ADM DDPM IDDPM LDM PNDM --detector wang2020_finetuning_DM
```
Note that this script uses the cache generated by `evaluate_detectors.py`, which is why it needs to be called first.

# Checkpoints
Pre-trained models from [Wang2020](https://github.com/peterwang512/CNNDetection), [Gragnaniello2021](https://github.com/grip-unina/GANimageDetection), and [Mandelli2022](https://github.com/polimi-ispl/gan-image-detection) can be downloaded from the corresponding repositories.

Checkpoints for the models we trained ourselves using the architecture from [Wang2020](https://github.com/peterwang512/CNNDetection) are hosted on [Zenodo](https://doi.org/10.5281/zenodo.7673266).
You can re-create our checkpoints using our [Dataset](#dataset) and the instructions from the original repository.

# Dataset
The dataset used in this work is hosted on [Zenodo](https://doi.org/10.5281/zenodo.7528112).
In total, the dataset contains 50k samples (256x256) for each of the following generators trained on LSUN Bedroom, divided into train, validation, and test set (39k/1k/10k).

| GANs | DMs |
| :--- | :-- |
| ProGAN | DDPM |
| StyleGAN | IDDPM |
| ProjectedGAN | ADM |
| Diffusion-StyleGAN2 | PNDM |
| Diffusion-ProjectedGAN | LDM |

The test set additionally contains 10k real images from [LSUN Bedroom](https://github.com/fyu/lsun) and can be used as-is.
Due to storage limitations, the train and validation set contain only generated images. Real images, required for training, have to be added according to the [instructions given below](#train-and-validation-set).

## Test Set
The test set can be downloaded from [Zenodo](https://zenodo.org/record/7528113). To validate that all images were downloaded correctly, you can run the following command and compare the hash value (computation takes some time).
```
$ find diffusion_model_deepfakes_lsun_bedroom/test/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
68bd7326113b9cde3cd4eadf1e10e5fca77df751  -
```

## Train and Validation Set
To recreate the train and validation set, first download both zip-archives from [Zenodo](https://zenodo.org/record/7528113) and extract them in the same directory. Then, download the entire LSUN Bedroom training data in LMDB format according to the instructions in the [LSUN repository](https://github.com/fyu/lsun). Finally, run [recreate_train_data.py](recreate_train_data.py) (requires packages [lmbd](https://lmdb.readthedocs.io/en/release/) and [Pillow](https://pillow.readthedocs.io/en/stable/)):
```
$ python recreate_train_data.py path/to/bedroom_train_lmbd path/to/diffusion_model_deepfakes_lsun_bedroom
```
The script will distribute real images such that for each generator there is a unique set of 40k real images (39k train, 1k validation) with the correct dimensions. Note that due to the large amount of images this will take some time. For each generator, real images will be in `0_real`, fake images in `1_fake`.

The script performs a simple form of validation. You can additionally run the following commands and compare the hash values (computation takes some time).
```
$ find diffusion_model_deepfakes_lsun_bedroom/train/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
10fd255e5669e76b4e9bde93f672733a5dfa92fa  -

$ find diffusion_model_deepfakes_lsun_bedroom/val/ -type f -print0 | sort -z | xargs -0 sha1sum | cut -d " " -f 1 | sha1sum
c836fe592e49ab630ccc30675694ab32aff8e1a3  -
```
