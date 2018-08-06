import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"

import numpy as np
import torch as T
import torch.nn as nn
from dataset import get_pretrain1_loaders
from models import Pretrain_model
from torchvision import transforms
from helper_functions import *
import torch.nn.functional as F
import argparse
import os
import random
import scipy.io

save_file = "data/saved_models/pt_glimpse_model.tar"
train_mat_file = "data/Original/lists/train_list.mat"
test_mat_file = "data/Original/lists/test_list.mat"

patch_size = 96

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=101)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument('--n_c', type=int, default=120)

def str2bool(v):
    if v.lower() == 'true':
        return True
    else:
        return False

parser.add_argument('--resume_training', type=str2bool, default=False)

opt = parser.parse_args()
print(opt)

if not os.path.exists("data/saved_models"):
    os.makedirs("data/saved_models")

def load_mat_files(mat_file):
    mat = scipy.io.loadmat(mat_file)
    return mat['file_list'], mat['labels']

opt.train_files, opt.train_labels = load_mat_files(train_mat_file)
opt.test_files, opt.test_labels = load_mat_files(test_mat_file)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
opt.my_transform = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        normalize
    ])

train_loader, test_loader = get_pretrain1_loaders(opt)

my_model = Pretrain_model(opt)
my_model = get_cuda(my_model)

device_ids = range(T.cuda.device_count())
my_model = nn.DataParallel(my_model, device_ids)

my_trainer = T.optim.Adam(my_model.parameters(), lr=opt.lr, betas=(0.5, 0.999))

def train_model(x_high, x_med, x_low, label_inds):
    out12, out13, out23, out123 = my_model(x_high, x_med, x_low)
    #Pretraining visula network using all combinations of high, medium and low patches ensuring each combination remain independently relevant even if another combination is more informative
    loss12 = F.cross_entropy(out12, label_inds)
    loss13 = F.cross_entropy(out13, label_inds)
    loss23 = F.cross_entropy(out23, label_inds)
    loss123 = F.cross_entropy(out123, label_inds)
    my_trainer.zero_grad()
    (loss12+loss13+loss23+loss123).backward()
    my_trainer.step()
    return loss123.item()

def test_model(x_high, x_med, x_low, label_inds):
    out12, out13, out23, out123 = my_model(x_high, x_med, x_low)

    _, p12 = T.max(out12, 1)
    _, p13 = T.max(out13, 1)
    _, p23 = T.max(out23, 1)
    _, p123 = T.max(out123, 1)

    return (p12 == label_inds).sum().item(), (p13 == label_inds).sum().item(), (p23 == label_inds).sum().item(), (p123 == label_inds).sum().item(), len(label_inds)

# def load_model_from_checkpoint():
#     global my_model, my_trainer
#     checkpoint = T.load(save_file)
#     my_model.load_state_dict(checkpoint['model_dict'])
#     my_trainer.load_state_dict(checkpoint['model_trainer'])
#     return checkpoint['epoch'], checkpoint['best_acc']


def training():
    start_epoch = best_acc = 0
    # if opt.resume_training:
    #     start_epoch = load_model_from_checkpoint()
    for epoch in range(start_epoch, opt.epochs):
        my_model.train()
        total_loss = []
        for x_high, x_med, x_low, labels in train_loader:
            x_high, x_med, x_low = get_cuda(x_high), get_cuda(x_med), get_cuda(x_low)
            labels = get_cuda(labels)
            loss = train_model(x_high, x_med, x_low, labels)
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        print("epoch:", epoch, "T_loss:", '%.3f' % (total_loss))

        my_model.eval()
        test_correct12 = test_correct13 = test_correct23 = test_correct123 = test_total = 0

        for x_high, x_med, x_low, labels in test_loader:
            x_high, x_med, x_low = get_cuda(x_high), get_cuda(x_med), get_cuda(x_low)
            labels = get_cuda(labels)
            with T.autograd.no_grad():
                correct12, correct13, correct23, correct123, total = test_model(x_high, x_med, x_low, labels)

            test_correct12 += correct12
            test_correct13 += correct13
            test_correct23 += correct23
            test_correct123 += correct123
            test_total += total


        acc12 = test_correct12 * 100 / float(test_total)
        acc13 = test_correct13 * 100 / float(test_total)
        acc23 = test_correct23 * 100 / float(test_total)
        acc123 = test_correct123 * 100 / float(test_total)

        print("Acc12:", '%.1f' % (acc12), "Acc13:", '%.1f' % (acc13), "Acc23:", '%.1f' % (acc23), "Acc123:",
              '%.1f' % (acc123))

        if best_acc < acc123:
            best_acc = acc123
            T.save({
                "model_dict": my_model.module.resnet.state_dict(),
            }, save_file)

if __name__ == "__main__":
    training()



