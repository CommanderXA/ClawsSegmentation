# UNet for mechanical Claws segmentation

Deep Neural Network (UNet) that segments the claws on the image.

by `stable-confusion` team

## Data

**[Kaggle Competition](https://www.kaggle.com/competitions/gdsc-nu-ml-hackathon-bts-case-competition/overview)**

Kaggle Competition from NU GDSC and BTS Kazakhstan.

## Results

**4th place** with `~87%` accuracy

## Libraries & Frameworks

- **PyTorch** *(Deep Learning)*
- **polars** *(work with csv)*
- **hydra** *(logging and configuration)*
- **tqdm** *(loading bar)*
- **PIL** *(work with images)*

## Techniques

- Augmentations
- L2 Regularization as weight decay
- MixedPrecision

## Project Setup

Run the following commands in project root directory.

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `sh ./setup.sh`

## Use

All the configuration is located inside `cfg/config.yaml`. This enables you to easily change the configuration of the UNet.

To use the project either:

- download the `unet.pth`
- place it inside the `models` directory

or train the model by yourself using `train.py`. Before training the model you need to download the dataset into the project root directory (leave the file name unchanged), then run `sh setup.sh`.

To get predicitons run `python main.py`, but note, you have to add at least one image into the `inference/imgs` directory.
