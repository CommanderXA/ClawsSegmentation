import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics import Dice

import hydra
from omegaconf import DictConfig

from dataset import ClawDataset
from model import UNet
from operations import train, validation


@hydra.main(version_base=None, config_path="conf", config_name="config")
def app(cfg: DictConfig) -> None:

    # ===================================
    # config
    cfg = cfg.config
    epochs = cfg.epochs
    batch_size = cfg.batch_size
    lr = cfg.lr
    weight_decay = cfg.weight_decay
    model_file = cfg.model_file
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    pretrained = cfg.pretrained
    grad_scaler = cfg.grad_scaler
    # ===================================

    log = logging.getLogger(__name__)
    if pretrained:
        checkpoint = torch.load(model_file)

    # mixed precision setup
    dtype = "bfloat16"
    ptdtype = {'float32': torch.float32,
               'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    torch.amp.autocast(device_type="cuda", dtype=ptdtype)

    # model
    model = UNet()
    model = model.to(device)
    model = torch.compile(model)
    if pretrained:
        model.load_state_dict(checkpoint["model"])
    log.info(f"Model parameters amount: {model.get_parameters_amount():,}")

    criterion = nn.BCEWithLogitsLoss()
    # criterion = DiceLoss()
    iou = BinaryJaccardIndex().to(device)
    dice = Dice(average='micro').to(device)
    optim = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=grad_scaler)

    if pretrained:
        optim.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])

    # datasets
    trainset = ClawDataset("data/train.csv")
    testset = ClawDataset("data/test.csv")

    # dataloaders
    trainloader = DataLoader(trainset, num_workers=4,
                             shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, num_workers=1,
                            shuffle=False, batch_size=1)

    train(model, trainloader, criterion, scaler, optim,
          dice, model_file, epochs, device, grad_scaler, log)
    validation(model, testloader, device)


if __name__ == "__main__":
    app()
