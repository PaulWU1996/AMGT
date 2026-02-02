from torch.utils.data import DataLoader
from utils import *
from trainer import AMGT
import random
import os
import numpy as np
import argparse
import torch


parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument(
    "-mode", type=str, default="rgb", help="rgb or flow (or joint for eval)"
)
parser.add_argument(
    "-train", type=str2bool, default="True", help="train or eval"
)
parser.add_argument("-comp_info", type=str)
parser.add_argument("-gpu", type=str, default="0")
parser.add_argument("-dataset", type=str, default="charades")
parser.add_argument(
    "-rgb_root",
    type=str,
    default="/media/faegheh/T71/Charades-Dataset/Charades_v1_rgb_feats_2/",
)
parser.add_argument("-type", type=str, default="original")
parser.add_argument("-lr", type=str, default="0.0001")
parser.add_argument("-epoch", type=str, default=50)
parser.add_argument("-model", type=str, default="PAT")
parser.add_argument("-load_model", type=str, default="False")
parser.add_argument("-batch_size", type=int, default=5)
parser.add_argument("-num_clips", type=str, default=256)
parser.add_argument("-skip", type=int, default=0)
parser.add_argument("-num_layer", type=str, default="False")
parser.add_argument("-unisize", type=str, default="True")
parser.add_argument("-num_classes", type=int, default=157)
parser.add_argument(
    "-annotation_file", type=str, default="./data/charades.json"
)
parser.add_argument("-fine_weight", type=float, default=0.1)
parser.add_argument("-coarse_weight", type=float, default=0.9)
parser.add_argument("-save_logit_path", type=str, default="./save_logit_path")
parser.add_argument("-step_size", type=int, default=7)
parser.add_argument("-gamma", type=float, default=0.1)

args = parser.parse_args()

# set random seed
SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print("Random_SEED:", SEED)

batch_size = args.batch_size
new_loss = AsymetricLoss()

if args.dataset == "charades":
    from charades_dataloader import Charades as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to", args.num_clips)
        from charades_dataloader import collate_fn_unisize

        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.charades_collate_fn_unisize
    else:
        from charades_dataloader import mt_collate_fn as collate_fn


def load_data(train_split, val_split, root):
    # Load Data
    print("load data", root)

    if len(train_split) > 0:
        dataset = Dataset(
            train_split,
            "training",
            root,
            batch_size,
            args.num_classes,
            args.num_clips,
            args.skip,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(
        val_split,
        "testing",
        root,
        batch_size,
        args.num_classes,
        args.num_clips,
        args.skip,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_dataloader.root = root
    dataloaders = {"train": dataloader, "val": val_dataloader}
    datasets = {"train": dataset, "val": val_dataset}

    return dataloaders, datasets


trainer = AMGT(args, new_loss)
