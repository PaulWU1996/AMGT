import time
import argparse
import csv
from torch.autograd import Variable
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from utils import *
from apmeter import APMeter
import os
from Evaluation import print_second_metric
from torch.nn import BCEWithLogitsLoss
# from calflops import calculate_flops

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
    default="/mnt/fast/nobackup/users/pw0036/Charades_v1_rgb_feats_2",
)
parser.add_argument("-type", type=str, default="original")
parser.add_argument("-lr", type=str, default="0.0001")
parser.add_argument("-epoch", type=str, default=50)
parser.add_argument("-model", type=str, default="AMGT")
parser.add_argument("-load_model", type=str, default="False")
parser.add_argument("-batch_size", type=int, default=5)
parser.add_argument("-num_clips", type=str, default=256)
parser.add_argument("-skip", type=int, default=0)
parser.add_argument("-num_layer", type=str, default="False")
parser.add_argument("-unisize", type=str, default="True")
parser.add_argument("-num_classes", type=int, default=157)
parser.add_argument(
    "-annotation_file", type=str, default="/mnt/fast/nobackup/users/pw0036/AMGT/data/charades.json"
)
parser.add_argument("-fine_weight", type=float, default=0.1)
parser.add_argument("-coarse_weight", type=float, default=0.9)
parser.add_argument("-save_logit_path", type=str, default="./save_logit_rgb")
parser.add_argument("-step_size", type=int, default=7)
parser.add_argument("-gamma", type=float, default=0.1)
parser.add_argument("-patience", type=int, default=5)

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


# batch_size = int(args.batch_size)
batch_size = args.batch_size
new_loss = AsymmetricLoss()
# new_loss = FocalLoss2d()

def cosine_similarity_loss(x1, x2):
    cos_sim = F.cosine_similarity(x1, x2, dim=-1)
    loss = (1.0 - cos_sim).mean()
    return loss

def relational_loss(s_logits, t_logits):
    # 计算 T 维度的自相关 (B, T, T)
    s_rel = torch.bmm(s_logits, s_logits.transpose(1, 2))
    t_rel = torch.bmm(t_logits, t_logits.transpose(1, 2))
    # 归一化后计算 MSE
    return F.mse_loss(F.normalize(s_rel, dim=-1), F.normalize(t_rel, dim=-1).detach())

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


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = 0.0
    worse = 0
    for epoch in range(num_epochs):
        since1 = time.time()
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        for (
            model1,
            model2,
            gpu,
            dataloader,
            optimizer,
            sched,
            model_file,
        ) in models:
            _, _, _, _ = train_step(
                model1, model2, gpu, optimizer, dataloader["train"], epoch
            )
            prob_val, val_loss, val_map = val_step(
                model2, gpu, dataloader["val"], epoch
            )
            # sched.step(val_loss)
            sched.step()
            # Time
            print(
                "epoch",
                epoch,
                "Total_Time",
                time.time() - since,
                "Epoch_time",
                time.time() - since1,
            )

            if Best_val_map < val_map:
                Best_val_map = val_map
                worse = 0
            else:
                worse += 1
            print("epoch", epoch, "Best Val Map Update", Best_val_map)
            pickle.dump(
                prob_val,
                open("./save_logit_rgb/" + str(epoch) + ".pkl", "wb"),
                pickle.HIGHEST_PROTOCOL,
            )
            print("logit_saved at:", "./save_logit_rgb/" + str(epoch) + ".pkl")
            print_second_metric(
                "./save_logit_rgb/" + str(epoch) + ".pkl",
                args.annotation_file,
                args.num_classes,
            )
            if worse >= args.patience:
                print("Early stopping at epoch", epoch)
                return


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (
            outputs.data.cpu().numpy()[0],
            probs.data.cpu().numpy()[0],
            data[2].numpy()[0],
            fps,
        )
    return results


def run_network(model, data, gpu):
    #
    inputs, mask, labels, other, hm = data
    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))

    if torch.equal(inputs, labels):
        pass
    else:
        inputs = inputs.squeeze(3).squeeze(3) # B D T
        inputs = inputs.transpose(1, 2) # B T D
    fine_probs, coarse_probs = model(inputs)

    # Logits
    finall_f = torch.stack(
        [args.fine_weight * fine_probs, args.coarse_weight * coarse_probs]
    )
    finall_f = torch.sum(finall_f, dim=0)

    probs_f = F.sigmoid(finall_f) * mask.unsqueeze(2)

    loss_coarse = new_loss(coarse_probs, labels) / torch.sum(mask)
    loss_fine = new_loss(fine_probs, labels) / torch.sum(mask)

    loss = loss_coarse + args.fine_weight * loss_fine

    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return finall_f, loss, probs_f, corr / tot


def train_step(model1, model2, gpu, optimizer, dataloader, epoch):
    model1.train(True)
    model2.train(True)
    num_iter = 0.0

    a_tot_loss = 0.0
    a_error = 0.0
    a_apm = APMeter()
    i_tot_loss = 0.0
    i_error = 0.0
    i_apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1

        inputs, mask, labels, other, hm = data
        data1 = (labels, mask, labels, other, hm)
        a_outputs, loss, probs, err = run_network(model1, data1, gpu)
        a_apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        a_error += err.data
        a_tot_loss += loss.data

        loss.backward()
        optimizer.step()

        model2.classifier.load_state_dict(model1.classifier.state_dict())
        for p in model2.classifier.parameters():
            p.requires_grad = False
        for p in model1.classifier.parameters():
            p.requires_grad = True

        optimizer.zero_grad()
        i_outputs, loss, probs, err = run_network(model2, data, gpu)
        detach_a_outputs = a_outputs.detach()
        loss = loss + cosine_similarity_loss(detach_a_outputs, i_outputs) + relational_loss(detach_a_outputs, i_outputs)

        i_apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        i_error += err.data
        i_tot_loss += loss.data

        loss.backward()
        optimizer.step()

    a_train_map = 100 * a_apm.value().mean()
    print("epoch", epoch, "assist train-map:", a_train_map)
    a_apm.reset()

    i_train_map = 100 * i_apm.value().mean()
    print("epoch", epoch, "inference train-map:", i_train_map)
    i_apm.reset()

    a_epoch_loss = a_tot_loss / num_iter
    i_epoch_loss = i_tot_loss / num_iter

    return a_train_map, a_epoch_loss, i_train_map, i_epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.0
    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu)
        if sum(data[1].numpy()[0]) > 25:
            p1, l1 = sampled_25(
                probs.data.cpu().numpy()[0],
                data[2].numpy()[0],
                data[1].numpy()[0],
            )
            sampled_apm.add(p1, l1)

        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])

        error += err.data
        tot_loss += loss.data

        probs_1 = mask_probs(
            probs.data.cpu().numpy()[0], data[1].numpy()[0]
        ).squeeze()

        full_probs[other[0][0]] = probs_1.T

    epoch_loss = tot_loss / num_iter
    val_map = (
        torch.sum(100 * apm.value())
        / torch.nonzero(100 * apm.value()).size()[0]
    )
    sample_val_map = (
        torch.sum(100 * sampled_apm.value())
        / torch.nonzero(100 * sampled_apm.value()).size()[0]
    )

    print("epoch", epoch, "Full-val-map:", val_map)
    # print('epoch', epoch, 'sampled-val-map:', sample_val_map)
    # print(100 * sampled_apm.value())
    apm.reset()
    sampled_apm.reset()
    return full_probs, epoch_loss, val_map


# # for rgb
if __name__ == "__main__":
    train_split = "./data/charades.json"
    test_split = train_split
    dataloaders, datasets = load_data(train_split, test_split, args.rgb_root)
    print(len(dataloaders["train"]))
    print(len(dataloaders["val"]))

    if not os.path.exists(args.save_logit_path):
        os.makedirs(args.save_logit_path)
    if args.train:

        if args.model == "AMGT":
            print("AMGT")
            from AMGT.network import AssistBranch, InferenceBranch

            assist_model = AssistBranch(
                d_model=1024,
                n_class=args.num_classes,
                n_layers=3,
                n_head=8,
                max_offset=512,
            )
            inference_model = InferenceBranch(
                d_model=1024,
                scale_factor=2,
                depth=3,
                d_state=128,
                d_conv=4,
                expand=2,
                mode="linear",
                align_corners=True,
                n_cls=int(args.num_classes),
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assist_model.to(device)
        inference_model.to(device)

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        optimizer = optim.Adam(
            [
                {
                    "params": assist_model.parameters(),
                    "lr": lr,
                },  # 比如 1e-3
                {
                    "params": inference_model.parameters(),
                    "lr": lr,
                },  # 比如 1e-4
            ]
        )
        lr_sched = optim.lr_scheduler.StepLR(
            optimizer, step_size=int(args.step_size), gamma=args.gamma
        )
        # run(
        #     [(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)],
        #     criterion,
        #     num_epochs=int(args.epoch),
        # )
        run(
            [
                (assist_model,
                inference_model,
                0,
                dataloaders,
                optimizer,
                lr_sched,
                args.comp_info,)
            ],
            criterion,
            num_epochs=int(args.epoch),
        )
