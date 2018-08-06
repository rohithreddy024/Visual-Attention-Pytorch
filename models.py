import torch as T
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch.distributions import Normal
from helper_functions import *

def get_nn_sequentials(n1, n2, out_size):
    seq = nn.Sequential(
        nn.Linear(n1 * 2048, n2*out_size),
        nn.BatchNorm1d(n2*out_size),
        nn.ReLU()
    )
    return seq

class Pretrain_model(nn.Module):
    def __init__(self, opt):
        super(Pretrain_model, self).__init__()
        self.out_size = 1024
        self.resnet = models.resnet50(pretrained=True)
        #Changing the stride of 1st convolution from 2 to 1 to facilitate processing 96x96 images
        self.resnet.conv1 = nn.Conv2d(3, 64, 5, 1, 2, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.lin_fc12 = get_nn_sequentials(2,1, self.out_size)
        self.lin_fc13 = get_nn_sequentials(2,1, self.out_size)
        self.lin_fc23 = get_nn_sequentials(2,1, self.out_size)
        self.lin_fc123 = get_nn_sequentials(3,1, self.out_size)

        self.fc12 = nn.Linear(self.out_size, opt.n_c)
        self.fc13 = nn.Linear(self.out_size, opt.n_c)
        self.fc23 = nn.Linear(self.out_size, opt.n_c)
        self.fc123 = nn.Linear(1*self.out_size, opt.n_c)

    def my_trans(self, out):
        out = F.avg_pool2d(out, 6, 1, 0)
        return out.view(out.size(0), -1)

    def forward(self, x_high, x_med, x_low):
        out_high = self.my_trans(self.resnet(x_high))
        out_med = self.my_trans(self.resnet(x_med))
        out_low = self.my_trans(self.resnet(x_low))
        #Depth concatenation layers
        out12 = T.cat([out_high, out_med], dim=1)
        out13 = T.cat([out_high, out_low], dim=1)
        out23 = T.cat([out_med, out_low], dim=1)
        out123 = T.cat([out_high, out_med, out_low], dim=1)
        #mixing depth concatenated layers
        out12 = self.lin_fc12(out12)
        out13 = self.lin_fc13(out13)
        out23 = self.lin_fc23(out23)
        out123 = self.lin_fc123(out123)

        return self.fc12(out12), self.fc13(out13), self.fc23(out23), self.fc123(out123)

class Glimpse_Network(nn.Module):
    def __init__(self, resnet, opt):
        super(Glimpse_Network, self).__init__()
        self.opt = opt
        self.resnet = resnet

    def my_trans(self, out):
        out = out.view(len(out), 2048, 6, 6)    #bs,2048,6,6
        out = F.avg_pool2d(out, 6, 1, 0)        #bs,2048,1,1
        return out.view(len(out), -1)           #bs,2048

    def forward(self, l_prev, info):
        phi = retina(l_prev, info, self.opt, self.training) #Extracts high, medium and low resolution patches corresponding to a location centre; Also compressed to 96x96 size
        what = []
        for i in range(self.opt.k):
            out = self.my_trans(self.resnet(phi[i]))
            what.append(out)

        what = T.cat(what, dim=1)               #bs,k*2048
        g = what
        return g

class Context_Network(nn.Module):
    def __init__(self, resnet):
        super(Context_Network, self).__init__()
        self.resnet = resnet

    def my_trans(self, out):
        #Output of resnet is 2048x6x6; i.e 36 feature vectors each corresponding to 36 spatial locations. Use 3x3 avgpool to compress into 2048x2x2 i.e 4 spatial locations
        #Specifically, top left, top right, bottom left and bottom right
        out = F.avg_pool2d(out, 3, 3, 0)
        return out

    def forward(self, input):
        cv = self.my_trans(self.resnet(input))  #bs,2048,2,2
        return cv

class Emission_Network(nn.Module):
    def __init__(self, opt):
        super(Emission_Network, self).__init__()
        #For 1st time step, use 1*1 convolution to predict probabilities over possible 4 regions of the context image; Trained using policy gradients
        self.fc1 = nn.Conv2d(2048, 1, 1, 1, 0, bias=False)
        #For rest of the time steps, use softmax attention as mentioned in the paper "https://arxiv.org/abs/1612.01887"
        self.W_v = nn.Linear(2048, 1024, bias=False)
        self.W_g = nn.Linear(opt.rnn_hidden, 1024, bias=False)
        self.W_h = nn.Linear(1024, 1, bias=False)
        # -------------------------------

    def soft_attn(self, cv, h):
        a = self.W_g(h).unsqueeze(1).repeat(1,4,1)      #bs,4,1024
        a = F.tanh(self.W_v(cv) + a)                    #bs,4,1024
        a = self.W_h(a).squeeze()                       #bs,4
        return a

    def forward(self, cv, h, initial, std):
        if initial:                                     #For 1st time step, use simple 1*1 to predict probabilities over 4 regions
            lp = self.fc1(cv).squeeze()                 # bs,2,2
            lp = lp.view(lp.size(0), -1)                #bs,4
        else:                                           #Use softmax attention to predict probabilities over 4 regions based on hidden state
            cv = cv.view(cv.size(0), cv.size(1), -1)    # bs,2048,4
            cv = cv.transpose(1, 2)                     # bs,4,2048
            lp = self.soft_attn(cv, h)                  #bs,4

        lp = F.softmax(lp, dim=1)                       #Compute probabilities
        #lp[:,0] corresponds to the probability that discriminative features are present in top left quadrant
        #lp[:,1] corresponds to top right quadrant
        #lp[:,2] corresponds to bottom left quadrant
        #lp[:,3] corresponds to bottom right quadrant
        lh = (lp[:,1] + lp[:,3] - lp[:,0] - lp[:,2]).unsqueeze(1)   #Formula ensures lh & lv lies between -1 & 1
        lv = (lp[:,0] + lp[:,1] - lp[:,2] - lp[:,3]).unsqueeze(1)
        mu = T.cat([lh, lv], dim=1)
        # mu = F.tanh(self.fc2(lp))
        norm_dist = Normal(mu, scale=std)
        l_t = F.tanh(norm_dist.sample().detach())                   #Sample a location from normal distribution with predicted mean=mu & given standard deviation=std
        return mu, l_t, norm_dist

class Classification_Network(nn.Module):
    def __init__(self, opt):
        super(Classification_Network, self).__init__()
        self.fc = nn.Linear(opt.rnn_hidden, opt.n_c)

    def forward(self, h):
        return self.fc(h)


class Baseline_Network(nn.Module):
    def __init__(self, opt):
        super(Baseline_Network, self).__init__()
        self.fc = nn.Linear(opt.rnn_hidden , 1)

    def forward(self, h):
        return F.relu(self.fc(h))

class Recurrent_Attention(nn.Module):
    def __init__(self, resnet, opt):
        super(Recurrent_Attention, self).__init__()
        self.opt = opt
        self.glimpse = Glimpse_Network(resnet, opt)
        self.core = nn.LSTMCell(opt.k*2048, opt.rnn_hidden)
        self.location = Emission_Network(opt)
        self.context = Context_Network(resnet)
        self.action = Classification_Network(opt)
        self.baseline = Baseline_Network(opt)

    def forward(self,l_prev, hc1_prev, cv, info, last = False):
        initial = False
        if l_prev is None:                  #1st time step; We predict location based on just context vector & context image, and not use LSTM
            hc1 = hc1_prev
            initial = True
        else:                               #For rest of the time steps; We predict location based on glimpses extracted and their respective hidden states
            g = self.glimpse(l_prev, info)  #Extract glimpse based on tuple
            hc1 = self.core(g, hc1_prev)    #Compute hidden state after mixing information of extracted glimpse

        l = bl = log_pi = None
        if last == True:
            output = self.action(hc1[0])    #For the last time step, compute logit vector used for training classification loss
            return hc1, l, bl, log_pi, output
        else:                               #For all the time steps except the last one, perform visual attention and compute location for next glimpse
            mu, l, norm_dist = self.location(cv, hc1[0].detach(), initial, self.opt.std_dev)
            bl = self.baseline(hc1[0].detach()) #Compute baseline for current time step based on hidden state
            log_pi = norm_dist.log_prob(l)      #compute log probabilities used for computing policy gradients
            log_pi = T.sum(log_pi, dim=1).unsqueeze(1)  #Suming over location tuple

        return hc1, l, bl, log_pi








