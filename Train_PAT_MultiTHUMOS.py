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
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow (or joint for eval)')
parser.add_argument('-train', type=str2bool, default='True', help='train or eval')
parser.add_argument('-comp_info', type=str)
parser.add_argument('-gpu', type=str, default='0')
parser.add_argument('-dataset', type=str, default='multithomus')
parser.add_argument('-rgb_root', type=str, default='/media/faegheh/T71/multithumos_features/multithumos_features_i3d/')
parser.add_argument('-type', type=str, default='original')
parser.add_argument('-lr', type=str, default='0.0001')
parser.add_argument('-epoch', type=str, default=50)
parser.add_argument('-model', type=str, default='PAT')
parser.add_argument('-load_model', type=str, default='False')
parser.add_argument('-batch_size', type=int, default=1)
parser.add_argument('-num_clips', type=str, default=800)
parser.add_argument('-skip', type=int, default=0)
parser.add_argument('-num_layer', type=str, default='False')
parser.add_argument('-unisize', type=str, default='True')
parser.add_argument('-num_classes', type=int, default=65)
parser.add_argument('-annotations_file', type=str, default='multithumos.json')
parser.add_argument('-fine_weight', type=float, default=0.2)
parser.add_argument('-coarse_weight', type=float, default=0.8)
parser.add_argument('-save_logit_path', type=str, default='./save_logit_path')
parser.add_argument('-step_size', type=int, default=35)
parser.add_argument('-gamma', type=float, default=0.1)
parser.add_argument('-input_type', type=str, default='rgb')

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
print('Random_SEED:', SEED)

batch_size = args.batch_size
new_loss = AsymmetricLoss()

if args.dataset == 'multithomus':
    from multithomus_dataloader import MultiThomus as Dataset

    if str(args.unisize) == "True":
        print("uni-size padd all T to",args.num_clips)
        from multithomus_dataloader import collate_fn_unisize
        collate_fn_f = collate_fn_unisize(args.num_clips)
        collate_fn = collate_fn_f.multithomus_collate_fn_unisize
    else:
        from multithomus_dataloader import mt_collate_fn as collate_fn


def load_data(train_split, val_split, root):
    # Load Data
    print('load data', root)

    if len(train_split) > 0:
        dataset = Dataset(args, args.input_type, 'training', 1.0, int(args.num_clips), int(args.skip), shuffle=True,
                          add_background=False)
        # dataset = Dataset_2(train_split, 'training', rgb_root, flow_root, batch_size, classes, int(args.num_clips), int(args.skip))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True, collate_fn=collate_fn)
        dataloader.root = root
    else:

        dataset = None
        dataloader = None

    val_dataset = Dataset(args, args.input_type, 'testing', 1.0, int(args.num_clips), int(args.skip), shuffle=False,
                          add_background=False)
    # val_dataset = Dataset_2(val_split, 'testing', rgb_root, flow_root, batch_size, classes, int(args.num_clips), int(args.skip))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=2,
                                                 pin_memory=True)
    val_dataloader.root = root
    dataloaders = {'train': dataloader, 'val': val_dataloader}
    datasets = {'train': dataset, 'val': val_dataset}

    return dataloaders, datasets


def run(models, criterion, num_epochs=50):
    since = time.time()
    Best_val_map = 0.
    for epoch in range(num_epochs):
        since1 = time.time()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for model, gpu, dataloader, optimizer, sched, model_file in models:
            _, _ = train_step(model, gpu, optimizer, dataloader['train'], epoch)
            prob_val, val_loss, val_map = val_step(model, gpu, dataloader['val'], epoch)
            # sched.step(val_loss)
            sched.step()
            # Time
            print("epoch", epoch, "Total_Time",time.time()-since, "Epoch_time",time.time()-since1)

            if Best_val_map < val_map:
                Best_val_map = val_map
            print("epoch",epoch,"Best Val Map Update",Best_val_map)
            pickle.dump(prob_val, open('./save_logit_rgb/' + str(epoch) + '.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
            print("logit_saved at:","./save_logit_rgb/" + str(epoch) + ".pkl")
            print_second_metric("./save_logit_rgb/" + str(epoch) + ".pkl", args.annotations_file, args.num_classes, args.dataset)


def eval_model(model, dataloader, baseline=False):
    results = {}
    for data in dataloader:
        other = data[3]
        outputs, loss, probs, _ = run_network(model, data, 0, baseline)
        fps = outputs.size()[1] / other[1][0]

        results[other[0][0]] = (outputs.data.cpu().numpy()[0], probs.data.cpu().numpy()[0], data[2].numpy()[0], fps)
    return results


def run_network(model, data, gpu, phase = 'training'):
    if phase == 'training':
        inputs, mask, labels, other, hm = data
        inputs = inputs.squeeze(3).squeeze(3)
    else:
        inputs, mask, labels, other, hm = data
        inputs = inputs.squeeze(dim=0).squeeze(2).squeeze(2).permute(0, 2, 1)
        labels = labels.squeeze(dim=0)


    # wrap them in Variable
    inputs = Variable(inputs.cuda(gpu))
    mask = Variable(mask.cuda(gpu))
    labels = Variable(labels.cuda(gpu))


    if phase == 'testing':
        if inputs.shape[2] == 1:
            inputs = inputs.permute(2, 1, 0)
            labels = labels.unsqueeze(dim=0)

        else:
            mask = mask.squeeze(dim=0)

    # if phase == 'testing':
    #     if inputs.shape[2] == 1:
    #         inputs = inputs.permute(2, 1, 0)
    #         labels = labels.unsqueeze(dim=0)
    #     else:
    #         mask = mask.squeeze(dim=0)
    fine_probs, coarse_probs = model(inputs)

    # Logit
    finall_f = torch.stack([args.fine_weight * fine_probs, args.coarse_weight * coarse_probs])
    finall_f = torch.sum(finall_f, dim=0)

    probs_f = F.sigmoid(finall_f) * mask.unsqueeze(2)

    loss_coarse = new_loss(coarse_probs, labels)/ torch.sum(mask)
    loss_fine =new_loss(fine_probs, labels)/ torch.sum(mask)

    loss = loss_coarse + loss_fine


    corr = torch.sum(mask)
    tot = torch.sum(mask)

    return finall_f, loss, probs_f, corr / tot

def train_step(model, gpu, optimizer, dataloader, epoch):
    model.train(True)
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    apm = APMeter()
    for data in dataloader:
        optimizer.zero_grad()
        num_iter += 1
        outputs, loss, probs, err = run_network(model, data, gpu, 'training')
        apm.add(probs.data.cpu().numpy()[0], data[2].numpy()[0])
        error += err.data
        tot_loss += loss.data

        loss.backward()
        optimizer.step()

    train_map = 100 * apm.value().mean()
    print('epoch', epoch, 'train-map:', train_map)
    apm.reset()

    epoch_loss = tot_loss / num_iter

    return train_map, epoch_loss


def val_step(model, gpu, dataloader, epoch):
    model.train(False)
    apm = APMeter()
    sampled_apm = APMeter()
    tot_loss = 0.0
    error = 0.0
    num_iter = 0.
    full_probs = {}

    # Iterate over data.
    for data in dataloader:
        num_iter += 1
        other = data[3]

        outputs, loss, probs, err = run_network(model, data, gpu, 'testing')

        if len(data[2].numpy().shape) == 4 :
            data[2] = data[2].squeeze(dim=0)
        for i in range(probs.data.cpu().numpy().shape[0]):
            p = probs.data.cpu().numpy()[i]
            l = data[2].numpy()[i]

            apm.add(p, l)


        error += err.data
        tot_loss += loss.data

        if len(data[1].numpy().shape) == 3:
            data[1] = data[1].squeeze(dim=0)
        probs_1 = mask_probs(probs.data.cpu().numpy()[0], data[1].numpy()[0]).squeeze()

        full_probs[other[0][0]] = probs_1.T

    epoch_loss = tot_loss / num_iter
    val_map = torch.sum(100 * apm.value()) / torch.nonzero(100 * apm.value()).size()[0]
    # sample_val_map = torch.sum(100 * sampled_apm.value()) / torch.nonzero(100 * sampled_apm.value()).size()[0]

    print('epoch', epoch, 'Full-val-map:', val_map)
    # print('epoch', epoch, 'sampled-val-map:', sample_val_map)
    # print(100 * sampled_apm.value())
    apm.reset()
    sampled_apm.reset()
    return full_probs, epoch_loss, val_map

# # for rgb
if __name__ == '__main__':
    train_split = 'multithumos.json'
    test_split = train_split
    dataloaders, datasets = load_data(train_split, test_split, args.rgb_root)
    print(len(dataloaders['train']))
    print(len(dataloaders['val']))

    if not os.path.exists(args.save_logit_path):
        os.makedirs(args.save_logit_path)
    if args.train:

        if args.model == "PAT":
            print("PAT")
            from PAT.PAT_Model import PAT
            num_clips = args.num_clips
            num_classes = args.num_classes
            inter_channels = [512, 512, 512, 512]
            num_block = 4
            # H
            head = 8
            # theta
            mlp_ratio = 8
            # D_0
            in_feat_dim = 1024
            # D_v
            final_embedding_dim = 512

            rgb_model = PAT(inter_channels, num_block, head, mlp_ratio, in_feat_dim, final_embedding_dim, num_classes, num_clips)
            print("loaded", args.load_model)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rgb_model.to(device)

        criterion = nn.NLLLoss(reduce=False)
        lr = float(args.lr)
        optimizer = optim.Adam(rgb_model.parameters(), lr=lr)
        lr_sched = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.step_size), gamma=args.gamma)
        run([(rgb_model, 0, dataloaders, optimizer, lr_sched, args.comp_info)], criterion, num_epochs=int(args.epoch))

