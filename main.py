import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,0"

import numpy as np
import torch.nn as nn
from dataset import get_main_loaders
from torchvision import transforms
from models import Recurrent_Attention
from helper_functions import *
import scipy.io
import argparse
import torchvision.models as models
from helper_functions import *
import torch.nn.functional as F
from PIL import Image, ImageDraw

pretrained_glimpsemodel = "data/saved_models/pt_glimpse_model.tar"
save_path = "data/saved_models/saved_model.tar"
train_mat_file = "data/Original/lists/train_list.mat"
test_mat_file = "data/Original/lists/test_list.mat"
patch_size = 96

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=101)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument('--rnn_hidden', type=int, default=2048)
parser.add_argument('--n_glimpses', type=int, default=1, help="No. of times to refer image before prediction")
parser.add_argument('--n_samples', type=int, default=20, help='No. of bounding boxed images to generate. Should be less than batch_size')
parser.add_argument('--std_dev', type=float, default=0.2)
parser.add_argument('--valid_size', type=float, default=0.3)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--start_size', type=int, default=2)
parser.add_argument('--n_jobs', type=int, default=4, help='For using joblib parallelization')
parser.add_argument('--n_c', type=int, default=120, help="No. of classes")
parser.add_argument('--task', type=str, default="train")

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

#Load filenames & labels of images in train_set and test_set
opt.train_files, opt.train_labels = load_mat_files(train_mat_file)
opt.test_files, opt.test_labels = load_mat_files(test_mat_file)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
opt.my_transform = transforms.Compose([
        transforms.Resize(patch_size),
        transforms.ToTensor(),
        normalize
    ])

train_loader, valid_loader, test_loader = get_main_loaders(opt)

def load_resnet():
    resnet = models.resnet50(pretrained=True)

    resnet.conv1 = nn.Conv2d(3, 64, 5, 1, 2, bias=False)
    resnet = nn.Sequential(*list(resnet.children())[:-2])

    checkpoint = T.load(pretrained_glimpsemodel)
    resnet.load_state_dict(checkpoint["model_dict"])
    # We fix the parameters of resnet and do not train it
    for param in resnet.parameters():
        param.requires_grad = False

    return get_cuda(resnet)

#Load pretrained resnet which is used to extract features from raw pixels; Used in both context network and glimpse network
resnet = load_resnet()

my_model = Recurrent_Attention(resnet, opt)
my_model = get_cuda(my_model)

device_ids = range(T.cuda.device_count())
my_model = nn.DataParallel(my_model, device_ids)

my_trainer = T.optim.Adam(filter(lambda p: p.requires_grad, my_model.parameters()),  lr=opt.lr, betas=(0.5, 0.999))

def reset(batch_size, x_batch):
    #Initializes hidden state and cell state for LSTM;
    h1 = T.zeros(batch_size, opt.rnn_hidden)
    c1 = T.zeros(batch_size, opt.rnn_hidden)
    #Context vector used for getting location of 1st glimpse
    cv = my_model.module.context(x_batch)

    return (get_cuda(h1), get_cuda(c1)), cv

def train_model(x_batch, label_inds, info):
    batch_size = len(x_batch)
    hc1, cv = reset(batch_size, x_batch)
    del x_batch
    log_pi = []
    baselines = []
    #1st time step
    hc1, l, bl, p = my_model(None, hc1, cv, info, last=False)
    baselines.append(bl)
    log_pi.append(p)

    for t in range(opt.n_glimpses-1):
        hc1, l, bl, p = my_model(l, hc1, cv, info, last = False)
        baselines.append(bl)
        log_pi.append(p)
    #last time step
    _, _, _, _, output = my_model(l, hc1, cv, info, last = True)


    baselines = T.cat(baselines, dim=1)
    log_pi = T.cat(log_pi, dim=1)

    _, predicted = T.max(output, 1)
    R = (predicted == label_inds).detach().float()
    R = R.unsqueeze(1).repeat(1, opt.n_glimpses)

    loss_action = F.cross_entropy(output, label_inds)
    loss_baseline = F.mse_loss(baselines, R)
    adjusted_reward = R - baselines.detach()
    loss_reinforce = T.sum(-log_pi*adjusted_reward, dim=1)
    loss_reinforce = T.sum(loss_reinforce, dim=0)

    loss = loss_action + loss_baseline + loss_reinforce

    my_trainer.zero_grad()
    loss.backward()
    my_trainer.step()
    return loss_action.item(), loss_baseline.item(), loss_reinforce.item()


def test_model(x_batch, label_inds, info):
    n_samples = 7                                   #Number of trajectories to average over
    x_batch = x_batch.repeat(n_samples, 1, 1, 1)    #Each image is run for n_sample times
    info = info.repeat(n_samples, 1)
    batch_size = len(x_batch)
    hc1, cv = reset(batch_size, x_batch)
    del x_batch

    hc1, l, _, _ = my_model(None, hc1, cv, info, last=False)

    for t in range(opt.n_glimpses-1):
        hc1, l, _, _ = my_model(l, hc1, cv, info, last = False)

    _, _, _, _, output = my_model(l, hc1, cv, info, last=True)
    output = output.view(n_samples, -1,  output.size(-1))
    output = T.mean(output, dim=0)                  #Mean prediction over n_samples
    _, predicted = T.max(output, 1)
    return (predicted == label_inds).sum().item(), len(label_inds)

def load_model_from_checkpoint():
    global my_model, my_trainer
    checkpoint = T.load(save_path)
    my_model.load_state_dict(checkpoint['model_dict'])
    my_trainer.load_state_dict(checkpoint['model_trainer'])
    return checkpoint['epoch'], checkpoint['best_acc']

def training():
    global my_trainer
    start_epoch = best_acc = 0
    if opt.resume_training:
        start_epoch, best_acc = load_model_from_checkpoint()

    for epoch in range(start_epoch, opt.epochs):

        my_model.train()
        Ta_loss = []
        Tb_loss = []
        Tr_loss = []

        for x_batch, labels, info in train_loader:
            x_batch = get_cuda(x_batch)
            labels = get_cuda(labels)
            la, lb, lr = train_model(x_batch, labels, info)
            Ta_loss.append(la)
            Tb_loss.append(lb)
            Tr_loss.append(lr)

        Ta_loss = np.mean(Ta_loss)
        Tb_loss = np.mean(Tb_loss)
        Tr_loss = np.mean(Tr_loss)

        print("epoch:", epoch, "Ta_loss:", '%.2f'%Ta_loss, "Tb_loss:", '%.2f'%Tb_loss, "Tr_loss:", '%.2f'%Tr_loss)

        if epoch%3 == 0:
            acc = testing(valid_loader)

            if best_acc < acc:
                best_acc = acc
                T.save({
                    "epoch": epoch + 1,
                    "best_acc": best_acc,
                    "model_dict": my_model.state_dict(),
                    "model_trainer": my_trainer.state_dict()
                }, save_path)

def testing(data_loader):
    my_model.eval()
    test_correct = test_total = 0
    for x_batch, labels, info in data_loader:
        x_batch = get_cuda(x_batch)
        labels = get_cuda(labels)
        with T.autograd.no_grad():
            correct, total = test_model(x_batch, labels, info)
        test_correct += correct
        test_total += total

    acc = test_correct*100/float(test_total)
    print("Testing Accuracy:", '%.1f' % (acc))
    return acc

def view_glimpses(batch_size):
    global my_model
    if not os.path.exists("imgs"):
        os.makedirs("imgs")
    checkpoint = T.load(save_path)
    my_model.load_state_dict(checkpoint["model_dict"])
    del checkpoint
    my_model = get_cuda(my_model).eval()
    x1_batch = labels = info = 0
    for x, y, info in test_loader:
        x1_batch, labels, info = x[:batch_size], y[:batch_size], info[:batch_size]
        break
    x1_batch, labels = get_cuda(x1_batch), get_cuda(labels)
    x_batch = x1_batch.repeat(10,1,1,1)
    info1 = info.repeat(10, 1)
    batch_size1 = len(x_batch)
    hc1, cv = reset(batch_size1, x_batch)
    del x_batch, x1_batch
    locs = []
    with T.autograd.no_grad():
        hc1, l, _, _ = my_model(None, hc1, cv, info1, last=False)
        locs.append(l)

        for t in range(opt.n_glimpses-1):
            hc1, l, _, _ = my_model(l, hc1, cv, info1, last = False)
            locs.append(l)

    #Calculate mean location for patch to be extracted
    for i in range(len(locs)):
        loc = locs[i]
        loc = loc.view(10, -1, loc.size(-1))
        loc = loc.mean(dim=0)
        locs[i] = loc.cpu().numpy()

    for j in range(batch_size):                                 #For each image in batch
        img, imgsize = get_image(opt, info[j], False)

        for i in range(len(locs)):                              #For each time step
            loc = locs[i]
            l_denorm = (0.5 * imgsize * (1 + loc[j])).astype(int)
            for k1 in range(1,opt.k+1):                         #For each patch extracted
                patch_size = (imgsize*opt.start_size*k1//4)
                from_x, from_y = l_denorm[0] - (patch_size // 2), l_denorm[1] - (patch_size // 2)
                to_x, to_y = from_x + patch_size, from_y + patch_size
                dr = ImageDraw.Draw(img)
                color = get_color(i)
                dr.rectangle(((from_x, from_y), (to_x, to_y)), outline=color)
        save_name = "imgs/glimpse_%d.jpg" % (j)
        img.save(save_name)


if __name__ == "__main__":
    if opt.task == "train":
        training()
    elif opt.task == "test":
        checkpoint = T.load(save_path)
        my_model.load_state_dict(checkpoint["model_dict"])
        del checkpoint
        testing(test_loader)
    elif opt.task == "view_glimpses":
        view_glimpses(opt.n_samples)