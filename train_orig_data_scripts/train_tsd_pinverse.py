import os 
import numpy as np
import torch
import torch
import torch.nn as nn

import torch
import torch.nn as nn

from torch.utils.data import Subset, Dataset
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image

import numpy as np
import random

import numpy as np
import matplotlib.pyplot as plt

import csv
from sklearn import svm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from sklearn.linear_model import LinearRegression

import sklearn
from sklearn import datasets

from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt

class NETWORK_F_MLP(nn.Module):
    def __init__(self, input_dim = 784, HIDDEN = 200, out_dim = 200, how_many_layers = 2):
        super(NETWORK_F_MLP, self).__init__()
        self.dim = out_dim
        self.many_layer = how_many_layers
        
        self.fc_list = []
        self.bn_list = []
        
#         self.fc_list.append(nn.Linear(input_dim+20, HIDDEN, bias=True))
        self.fc_list.append(nn.Linear(input_dim, HIDDEN, bias=True))
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        for i in range(0, self.many_layer-1):
            self.fc_list.append(nn.Linear(HIDDEN, HIDDEN, bias=True))
            self.bn_list.append(nn.BatchNorm1d(HIDDEN))
            
        self.fc_list = nn.ModuleList(self.fc_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.fc_final = nn.Linear(HIDDEN, out_dim, bias=True)

    def forward(self, x):
        
        x = x.reshape(x.shape[0], -1)
        
        for i in range(0, self.many_layer):
            x = self.fc_list[i](x)
            x = torch.relu(x)
            x = self.bn_list[i](x)
        
        x = self.fc_final(x)
        x = torch.sigmoid(x)
        return x

class Advanced1DCNN_channel(nn.Module):
    def __init__(self, input_channel=1, num_classes=100, input_size=4000, num_channel=60):
        super(Advanced1DCNN_channel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=11, padding=5),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=11, padding=5),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256 * 15, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        self.fc3 = nn.Linear(512, num_classes)
        
        self.MLP = NETWORK_F_MLP(input_dim = 128*input_channel, HIDDEN = 4000, out_dim = num_classes, how_many_layers = 1)
        
    def forward(self, x):
        bs, channel = x.shape[0], x.shape[1]
        x = x.unsqueeze(2)
        x = x.flatten(0, 1)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
#         print(out.shape)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.sigmoid(out)
        out = out.reshape(bs, channel, -1)
        out = out.flatten(-2, -1)
        out = self.MLP(out)
        return out

class ComplexClassifier(nn.Module):
    def __init__(self, dim_features=128, num_classes = 10):
        super(ComplexClassifier, self).__init__()
#         self.fc1 = nn.Linear(dim_features, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 512)
#         self.bn2 = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(dim_features, num_classes)  # CIFAR-10 has 10 classes

    def forward(self, x):
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.fc2(x)))
#         x = torch.relu(self.bn3(self.fc3(x)))
        x = self.fc4(x)  # No activation, CrossEntropyLoss includes softmax
        # Using a linear network for channel network  
        return x

class MCALoss:
    def __init__(self, emb_size, device):
        self.track_cov_final = torch.zeros((emb_size * 2), requires_grad=False).to(device)
        self.track_cov_estimate_final = torch.zeros((emb_size * 2), requires_grad=False).to(device)
        self.step = 0

    def __call__(self, fmri_f, smri_f):
        return self.mca_loss(fmri_f, smri_f)

    def mca_loss(self, fmri_f, smri_f):
        self.step += 1
        device = fmri_f.device
        emb_size = fmri_f.shape[1]

        RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0]
        RG = (smri_f.T @ smri_f) / smri_f.shape[0]
        P = (fmri_f.T @ smri_f) / smri_f.shape[0]

        input_dim, output_dim = RF.shape[1], RG.shape[1]
        RFG = torch.zeros((input_dim + output_dim, input_dim + output_dim)).to(device)
        RFG[:input_dim, :input_dim] = RF
        RFG[input_dim:, input_dim:] = RG
        RFG[:input_dim, input_dim:] = P
        RFG[input_dim:, :input_dim] = P.T

        self.track_cov_final, self.track_cov_estimate_final = self.calc_track_cov(
            RFG, self.track_cov_final, self.step, emb_size
        )
        cost, tsd = self.cost_trace(RFG, self.track_cov_estimate_final, emb_size)
        return cost.detach(), tsd.detach()

    @staticmethod
    def cost_trace(RFG, track_cov_estimate_final, dim):
        RF_E = track_cov_estimate_final[:dim, :dim]
        RG_E = track_cov_estimate_final[dim:, dim:]
        P_E = track_cov_estimate_final[:dim, dim:]

        RF_EI = torch.inverse(RF_E)
        RG_EI = torch.inverse(RG_E)

        RF = RFG[:dim, :dim]
        RG = RFG[dim:, dim:]
        P = RFG[:dim, dim:]

        COST = (
            -RF_EI @ RF @ RF_EI @ P_E @ RG_EI @ P_E.T
            + RF_EI @ P @ RG_EI @ P_E.T
            - RF_EI @ P_E @ RG_EI @ RG @ RG_EI @ P_E.T
            + RF_EI @ P_E @ RG_EI @ P.T
        )
        TSD = RF_EI @ P_E @ RG_EI @ P_E.T
        return -torch.trace(COST), -torch.trace(TSD)

    @staticmethod
    def calc_track_cov(RP, track_cov, step, dim):
        device = RP.device
        cov = RP + torch.eye((RP.shape[0])).to(device) * 1e-6
        track_cov, cov_estimate = MCALoss.adaptive_estimation(track_cov, 0.5, cov, step)

        return track_cov, cov_estimate

    @staticmethod
    def adaptive_estimation(v_t, beta, square_term, i):
        v_t = beta * v_t + (1 - beta) * square_term.detach()
        return v_t, (v_t / (1 - beta**i))

def create_folder_if_not_exists(folder_path):
    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def fmcat_loss(fmri_f, smri_f):
    eps = torch.eye(fmri_f.shape[1], device=fmri_f.device) * 1e-5
    RF = (fmri_f.T @ fmri_f) / fmri_f.shape[0] + eps
    RG = (smri_f.T @ smri_f) / smri_f.shape[0] + eps
    P = (fmri_f.T @ smri_f) / smri_f.shape[0]

    lhs = torch.linalg.lstsq(RF, P).solution
    rhs = torch.linalg.lstsq(RG, P.T).solution
    tsd = -torch.trace(lhs @ rhs)

    return tsd

def return_cost_trace(RFG, track_cov_estimate_final):
    RF_E = track_cov_estimate_final[:128, :128]
    RG_E = track_cov_estimate_final[128:, 128:]
    P_E = track_cov_estimate_final[:128, 128:]

    RF_EI = torch.inverse(RF_E)
    RG_EI = torch.inverse(RG_E)

    RF = RFG[:128, :128]
    RG = RFG[128:, 128:]
    P = RFG[:128, 128:]

    COST = -RF_EI@RF@RF_EI@P_E@RG_EI@P_E.T \
            + RF_EI@P@RG_EI@P_E.T \
            - RF_EI@P_E@RG_EI@RG@RG_EI@P_E.T \
            + RF_EI@P_E@RG_EI@P.T
    
    TSD = RF_EI@P_E@RG_EI@P_E.T

    return -torch.trace(COST), -torch.trace(TSD)

# for some reasons the adaptive filter is needed
def adaptive_estimation(v_t, beta, square_term, i):
    v_t = beta*v_t + (1-beta)*square_term.detach()
    return v_t, (v_t/(1-beta**i))

def MCA_LOSS_GIVEN_R(RP, track_cov, i, dim):
    cov = RP + torch.eye((RP.shape[0])).cuda()*(.000001)
    track_cov, cov_estimate = adaptive_estimation(track_cov, 0.5, cov, i)

    cov_estimate_f = cov_estimate[:dim, :dim]
    cov_f = cov[:dim, :dim]

    cov_estimate_g = cov_estimate[dim:, dim:]
    cov_g = cov[dim:, dim:]

    LOSS = (torch.linalg.inv(cov_estimate)*cov).sum() - (torch.linalg.inv(cov_estimate_f)*cov_f).sum() -(torch.linalg.inv(cov_estimate_g)*cov_g).sum()
    return track_cov, cov_estimate, LOSS

# torch.cuda.set_device(3)

subject = np.load('./run_experiments_ssl_signal/subject.npy')
label = np.load('./run_experiments_ssl_signal/label.npy')
new_label = np.load('./run_experiments_ssl_signal/new_label.npy')
eeg_data = np.load('./run_experiments_ssl_signal/eeg_data.npy')
emg_data = np.load('./run_experiments_ssl_signal/emg_data.npy')

subject = torch.from_numpy(subject).long()
label = torch.from_numpy(label).long()
new_label = torch.from_numpy(new_label).long()
eeg_data = torch.from_numpy(eeg_data).float()
emg_data = torch.from_numpy(emg_data).float()

SAMPLE_X = eeg_data
SAMPLE_Y = emg_data
label_tensor = new_label

# Generate Data Cross Subject

train_idx = [np.where(subject == k)[0] for k in range(0, 20)]
test_idx = [np.where(subject == k)[0] for k in range(20, 25)]

train_idx = np.concatenate(train_idx)
test_idx = np.concatenate(test_idx)

train_major_label, test_major_label = new_label[train_idx], new_label[test_idx]
train_subtle_label, test_subtle_label = label[train_idx], label[test_idx]
train_subj_label, test_subj_label = subject[train_idx], subject[test_idx]

train_X, test_X = SAMPLE_X[train_idx], SAMPLE_X[test_idx]
train_Y, test_Y = SAMPLE_Y[train_idx], SAMPLE_Y[test_idx]



save_path = './TSD_PINV/'
create_folder_if_not_exists(save_path)

NET_1 = Advanced1DCNN_channel(SAMPLE_X.shape[1], 128, 4000).cuda()
NET_2 = Advanced1DCNN_channel(SAMPLE_Y.shape[1], 128, 4000).cuda()

classifier_major = ComplexClassifier(dim_features = 128, num_classes = 3).cuda()
classifier_subtle = ComplexClassifier(dim_features = 128, num_classes = 11).cuda()
classifier_subj = ComplexClassifier(dim_features = 128, num_classes = 25).cuda()
criterion = nn.CrossEntropyLoss()

beta1 = 0.9
beta2 = 0.999

# lr1 = 0.0005
# lr2 = 0.0005
lr1 = 1e-3
lr2 = 1e-3

optimizer_1 = optim.Adam([
    {'params': NET_1.parameters(), 'lr': lr2, 'betas': (beta1, beta2)},
], amsgrad = True)

optimizer_2 = optim.Adam([
    {'params': NET_2.parameters(), 'lr': lr2, 'betas': (beta1, beta2)},
], amsgrad = True)

optimizer_classifier = optim.Adam([
    {'params': classifier_major.parameters(), 'lr': lr2, 'betas': (beta1, beta2)},
    {'params': classifier_subtle.parameters(), 'lr': lr2, 'betas': (beta1, beta2)},
    {'params': classifier_subj.parameters(), 'lr': lr2, 'betas': (beta1, beta2)},
], amsgrad = True)

save_curve = []
eig_list = []
classifier_error = []

test_primary_error_save = []
test_error_save = []

track_cov_final = torch.zeros((128+128)).cuda()
track_cov_estimate_final = torch.zeros((128+128)).cuda()

loss_fn = MCALoss(128, torch.device('cuda'))

print(f"Training on {len(train_X)} samples", flush=True)

for i in range(0, 15_000 + 1):
    batch_size = 50
    batch_indices = torch.randint(0, train_X.shape[0], (batch_size,))
    input_x = train_X[batch_indices]
    input_y = train_Y[batch_indices]
    
    feature_1 = NET_1(input_x.cuda())
    feature_2 = NET_2(input_y.cuda())
    
    RF = (feature_1.T@feature_1)/feature_1.shape[0]
    RG = (feature_2.T@feature_2)/feature_2.shape[0]
    P = (feature_1.T@feature_2)/feature_2.shape[0]

    
    input_dim, output_dim = RF.shape[1], RG.shape[1]
    RFG = torch.zeros((input_dim+output_dim, input_dim+output_dim)).cuda()
    RFG[:input_dim, :input_dim] = RF
    RFG[input_dim:, input_dim:] = RG
    RFG[:input_dim, input_dim:] = P
    RFG[input_dim:, :input_dim] = P.T
    
    cost, tsd = loss_fn(feature_1, feature_2)
    loss = fmcat_loss(feature_1, feature_2)
    output_major, output_subtle, output_subj = classifier_major(feature_1.detach()), classifier_subtle(feature_1.detach()), classifier_subj(feature_1.detach())
    
    optimizer_1.zero_grad()
    optimizer_2.zero_grad()  
    optimizer_classifier.zero_grad()

    loss.backward()
    
    optimizer_1.step()  
    optimizer_2.step()  
    optimizer_classifier.step()  

    if i%100 == 0:
        print(f"Step {i}, loss: {loss.item()} tsd: {tsd.item()} cost: {cost.item()}")
 
        if i%500 == 0:
            torch.save(NET_1.state_dict(), save_path+"NET_1_iter{0}.pth".format(i))
            torch.save(NET_2.state_dict(), save_path+"NET_2_iter{0}.pth".format(i))
    

