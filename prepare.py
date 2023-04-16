from os import listdir
from os.path import isfile, join
import polars as pl

def get_files(dir: str):
    onlyfiles = [join(dir, f) for f in listdir(dir) if isfile(join(dir, f))]
    onlyfiles.sort()
    return onlyfiles


# training data
imgs = get_files("data/train/imgs")
masks = get_files("data/train/masks")

df = pl.DataFrame({
    "id": [i for i in range(len(imgs))],
    "imgs": imgs,
    "masks": masks,
})

df.write_csv("data/train.csv")

# test data
imgs = get_files("data/test/imgs")

df = pl.DataFrame({
    "id": [i for i in range(len(imgs))],
    "imgs": imgs,
})

df.write_csv("data/test.csv")