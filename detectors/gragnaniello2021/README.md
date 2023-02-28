## GAN Image Detection
This repository contains the best GAN generated image detector, namely <*ResNet50 NoDown*>, among those presented in the paper:

**Are GAN generated images easy to detect? A critical analysis of the state-of-the-art**  
Diego Gragnaniello, Davide Cozzolino, Francesco Marra, Giovanni Poggi and Luisa Verdoliva.
<br />In IEEE International Conference on Multimedia and Expo (ICME), 2021.

The very same architecture has been trained with images generated either by the Progressive Growing GAN or the StyleGAN2 architecture.
<br />To download the trained weights, run:
```
wget -e robots=off -nd -P ./weights -A .pth -r http://www.grip.unina.it/download/GANdetection
```
or manually download them from [here](http://www.grip.unina.it/download/GANdetection) and put them in the folder *weights*.

### Requirements
- python>=3.6
- numpy>=1.19.4
- pytorch>=1.6.0
- torchvision>=0.7.0
- pillow>=8.0.1

### Test on a folder

To test the network on an image folder and to collect the results in a CSV file, run the following command:

```
python main.py -m weights/gandetection_resnet50nodown_stylegan2.pth -i ./example_images -o out.csv
```

### Traning-set

#### 1) ProGAN Traning-set
For training using ProGAN images, we used the traning-set provided by ["CNN-generated images are surprisingly easy to spot...for now"](https://github.com/PeterWang512/CNNDetection)
 
#### 2) StyleGAN2 Traning-set
For training using StyleGAN2 images, we generated the fake images and used different public datasets for pristine images.
The 720K fake images can be downloaded [here](http://www.grip.unina.it/download/gan_detection_training/).
The number of pristine images for each public dataset is reported in the following table:

|   Dataset   |   Size    | #Images |
|:------------|:---------:|--------:|
| LSUN cat    |  256x256  |  120000 | 
| LSUN church |  256x256  |  120000 |
| LSUN hourse |  256x256  |  120000 |
| LSUN car    |  512x512  |  120000 |
| AFHQ cat    |  512x512  |    4700 |
| AFHQ dog    |  512x512  |    4700 |
| AFHQ wild   |  512x512  |    4700 |
| AnimalWeb   |  512x512  |   14100 |
| BreCaHAD    |  512x512  |    1800 |
| FFHQ        | 1024x1024 |   28800 |
| MetFaces    | 1024x1024 |   14100 |

We downloaded and processed the pristine images following the guide provided by [StyleGAN2-ada](https://github.com/NVlabs/stylegan2-ada).
The links to public datasets are reported in the guide
except [AnimalWeb](https://arxiv.org/pdf/1909.04951.pdf) dataset that can be downloaded [here](https://drive.google.com/u/0/uc?id=13PbHxUofhdJLZzql3TyqL22bQJ3HwDK4&export=download).
