import os
from distutils.dir_util import copy_tree
import random
import shutil


def create_sub_folders(dir_name):

    for i in range(102):
        name = "000" + str(i)
        os.mkdir(f"{dir_name}/train/{name[-3:]}")


def move_test_samples(dir_name):

    copy_tree("../data/flowers/test", f"{dir_name}/test")


def sample_and_move_pictures(dir_name, n_samples, RND_SEED):

    random.seed(RND_SEED)

    all_pictures = []
    for folder in os.listdir(f"../data/flowers/train"):
        for img in os.listdir(f"../data/flowers/train/{folder}"):
            all_pictures += [(folder, img)]

    picked = sorted(all_pictures, key=lambda x: x[0])
    if n_samples <= 100:
        picked = [picked[i] for i in range(0, 1020, 10)][:n_samples]

    else:
        picked = random.sample(all_pictures, k=n_samples)

    for folder, img in picked:
        if folder not in os.listdir(f"{dir_name}/train"):
            os.mkdir(f"{dir_name}/train/{folder}")
        shutil.copyfile(
            f"../data/flowers/train/{folder}/{img}", f"{dir_name}/train/{folder}/{img}"
        )


if __name__ == "__main__":

    RND_SEED = 0

    NUMBER_OF_SAMPLES = [5, 10, 50, 100, 500, 1000]

    for n_samples in NUMBER_OF_SAMPLES:

        dir_name = f"flowers_{n_samples}"

        if dir_name in os.listdir("."):
            continue

        os.mkdir(dir_name)
        os.mkdir(f"{dir_name}/train")

        move_test_samples(dir_name)

        sample_and_move_pictures(dir_name, n_samples, RND_SEED)
