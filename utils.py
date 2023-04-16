import os

from torch.utils.data import DataLoader
import polars as pl

from dataset import ClawDataset


def get_files(dir: str):
    onlyfiles = [os.path.join(dir, f) for f in os.listdir(
        dir) if os.path.isfile(os.path.join(dir, f))]
    onlyfiles.sort()
    return onlyfiles


def compose_csv():
    if os.path.exists("data/inference.csv"):
        os.remove("data/inference.csv")
    imgs = get_files("inference/imgs")

    df = pl.DataFrame({
        "id": [i for i in range(len(imgs))],
        "imgs": imgs,
    })

    df.write_csv("data/inference.csv")


def load_sequence() -> DataLoader:
    compose_csv()
    testset = ClawDataset("data/inference.csv")
    testloader = DataLoader(testset, shuffle=False)
    return testloader
