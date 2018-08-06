import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch as T
import random
from torch.utils.data.sampler import SubsetRandomSampler


images_folder = "data/Original/Images"


manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
T.manual_seed(manual_seed)
if T.cuda.is_available():
    T.cuda.manual_seed_all(manual_seed)

class Pretrain1_Dataset(Dataset):
    def __init__(self, file_list, labels, my_transform, istrain):
        self.file_list = file_list
        self.labels = labels
        self.istrain = istrain
        self.my_transform = my_transform


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, ind):
        filename = self.file_list[ind][0][0]
        label = self.labels[ind][0]
        img = Image.open(os.path.join(images_folder, filename)).convert('RGB')
        if self.istrain:
            img = transforms.RandomHorizontalFlip()(img)
        width, height = img.size
        small_size = min(width, height)
        high_res = small_size // 4
        med_res = small_size // 2
        low_img = transforms.RandomCrop(small_size)(img)
        med_img = transforms.CenterCrop(med_res)(low_img)
        high_img = transforms.CenterCrop(high_res)(med_img)
        low_img = self.my_transform(low_img)
        med_img = self.my_transform(med_img)
        high_img = self.my_transform(high_img)

        label = T.LongTensor([label]).squeeze()

        return high_img, med_img, low_img, label-1

def get_pretrain1_loaders(opt):
    my_transform = opt.my_transform

    train_dataset = Pretrain1_Dataset(opt.train_files, opt.train_labels, my_transform, istrain=True)
    test_dataset = Pretrain1_Dataset(opt.test_files, opt.test_labels, my_transform, istrain=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)

    return train_loader, test_loader

class Main_Dataset(Dataset):
    def __init__(self, file_list, labels, my_transform, istrain):
        self.file_list = file_list
        self.labels = labels
        self.istrain = istrain
        self.my_transform = my_transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, ind):
        filename = self.file_list[ind][0][0]
        label = self.labels[ind][0]
        img = Image.open(os.path.join(images_folder, filename)).convert('RGB')
        if self.istrain:
            img = transforms.RandomHorizontalFlip()(img)
        width, height = img.size
        size = min(width, height)
        #If training, chose center of patches to be extracted randomly; If testing, chose the centre of the image as centre of the patches
        if self.istrain:
            from_w, from_h = random.randint(0, width - size), random.randint(0, height - size)
        else:
            from_w, from_h = (width - size) // 2, (height - size) // 2

        img = img.crop((from_w, from_h, from_w + size, from_h + size))
        img = self.my_transform(img)

        label = T.LongTensor([label]).squeeze()
        #info stores information about context image which are be used for extracting glimpses for timestep>1
        info = T.LongTensor([from_w, from_h, ind]).squeeze()
        return img, label-1, info

def get_valid_loader(opt, test_dataset):
    total_imgs = len(test_dataset)
    indices = list(range(total_imgs))
    split = int(np.floor(opt.valid_size*total_imgs))
    random.shuffle(indices)

    valid_idx = indices[:split]
    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_loader = DataLoader(test_dataset, batch_size=64, sampler=valid_sampler, num_workers=opt.num_workers)
    return valid_loader

def get_main_loaders(opt):

    my_transform = opt.my_transform
    train_dataset = Main_Dataset(opt.train_files, opt.train_labels, my_transform, istrain=True)
    test_dataset = Main_Dataset(opt.test_files, opt.test_labels, my_transform, istrain=False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers)
    valid_loader = get_valid_loader(opt, test_dataset)
    test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True, num_workers=opt.num_workers)
    return train_loader, valid_loader, test_loader





