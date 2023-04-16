import os

import torch
import torchvision
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision.utils import draw_segmentation_masks

import numpy as np
from tqdm import tqdm
import pandas as pd

from model import UNet

losses = []
metrics = []


def train(model: UNet, loader, criterion, scaler, optim, dice, model_file, epochs: int, device, grad_scaler):
    model.train()

    for epoch in range(1, epochs+1):
        with tqdm(iter(loader)) as tepoch:
            tepoch.set_description(f"Epoch: {epoch}")
            # mixed precision training
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=grad_scaler):
                for x, y in tepoch:
                    x, y = x.to(device), y.to(device)
                    prediction = model(x)
                    loss: torch.Tensor = criterion(prediction, y)

                    # backprop and optimize
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                    losses.append(loss.item())

                    # metric = iou(torch.floor(torch.sigmoid(prediction) + .5), torch.floor(y + .5))
                    metric = dice(prediction, y.int())
                    metrics.append(metric.item())

            # save model
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optim.state_dict(),
                "scaler": scaler.state_dict()
            }
            # Write checkpoint as desired, e.g.,
            torch.save(checkpoint, model_file)

            print(f"Loss: {np.mean(losses)}, Accuracy: {np.mean(metrics)}")
            losses.clear()
            metrics.clear()


def validation(model: UNet, loader, device):
    model.eval()
    with torch.no_grad():
        results = []
        images = []

        with tqdm(iter(loader)) as tepoch:
            for x, name in tepoch:
                x = x.to(device)
                y = model(x)
                y = torch.floor(torch.sigmoid(y) + .5)

                claws_with_masks = draw_segmentation_masks(image=(
                    x[0]*255).type(torch.uint8).cpu(), masks=(y[0] > 0).cpu(), alpha=0.7, colors="#1FFF78")
                save_image((claws_with_masks / 255), os.path.join(
                    "data/test/masks", name[0].split('.')[0] + ".png"))

                results.append(claws_with_masks)
                images.append(y)

        grid = make_grid(results)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save("images/test.png")
        results.clear()

        responses = []
        for i, image in enumerate(images):
            # img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            coords = list(np.where(image.view(1, -1).cpu().squeeze(0) > 0)[0])
            short_coords = [str(coords[0])]
            j = 1
            length = 1
            while j != len(coords):
                if coords[j]-1 in coords:
                    length += 1
                else:
                    short_coords.append(str(length))
                    short_coords.append(str(coords[j]))
                    length = 1
                j += 1
            short_coords.append(str(length))

            responses.append([i] + [" ".join(short_coords)])

        sample = pd.DataFrame(responses, columns=["ImageID", "Expected"])
        sample.to_csv("sample16_new2.csv", index=None)


def inference(model, loader, device):
    assert len(loader) > 0, "Add image(s) into the `inference/imgs` directory"
    model.eval()
    with torch.no_grad():
        results = []
        images = []

        with tqdm(iter(loader)) as tepoch:
            for x, name in tepoch:
                x = x.to(device)
                y = model(x)
                y = torch.floor(torch.sigmoid(y) + .5)

                claws_with_masks = draw_segmentation_masks(image=(
                    x[0]*255).type(torch.uint8).cpu(), masks=(y[0] > 0).cpu(), alpha=0.7, colors="#1FFF78")

                save_image((claws_with_masks / 255), os.path.join(
                    "inference/masks", name[0].split('.')[0] + ".png"))

                results.append(claws_with_masks)
                images.append(y)

        grid = make_grid(results)
        img = torchvision.transforms.ToPILImage()(grid)
        img.save("inference/test.jpg")
        results.clear()
