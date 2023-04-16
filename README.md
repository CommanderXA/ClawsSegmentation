# UNet for mechanical Claws segmentation

Deep Neural Network (UNet) that segments the claws on the image.

by `stable-confusion` team

## Data

**[Kaggle Competition]**(https://www.kaggle.com/competitions/gdsc-nu-ml-hackathon-bts-case-competition/overview)

Kaggle Competition from NU GDSC and BTS Kazakhstan.

## Results

**4th place** with `~87%` accuracy

## Libraries & Frameworks

- PyTorch
- polars
- tqdm

## Techniques

- Augmentations
- L2 Regularization as weight decay
- MixedPrecision

## Project Setup

Run the following commands in project root directory.

- `source .venv/bin/activate`
- `./setup.sh`

## Use

To use the project either:

- download the `unet.pth`
- place it inside the `models` directory

or train the model by yourself using `train.py`. Before training the model you need to download the dataset inside the project root directory and leave the filenamme unchenged, then run `./setup.sh`.

To get predicitons run `python main.py`, but note, you have to add at least one image into the `inference/imgs` directory
