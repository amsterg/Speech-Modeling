import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from math import floor
from torch.utils.tensorboard import SummaryWriter
from yaml import safe_load
import os
import sys

# from src.data.data_utils import ImbalancedDatasetSampler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.getcwd())

from src.data.data_utils import HDF5TorchDataset
import matplotlib.pyplot as plt
import librosa
np.random.seed(42)


class ACCENT_ENCODER(nn.Module):
    def __init__(self,
                 input_shape=(160, 40),
                 load_model=False,
                 epoch=0,
                 dataset_train='gmu_4_train',
                 dataset_val='gmu_4_val',
                 device=torch.device('cpu'),
                 loss_=None):
        super(ACCENT_ENCODER, self).__init__()
        self.input_shape = input_shape
        with open('src/config.yaml', 'r') as f:
            self.config_yml = safe_load(f.read())

        model_save_dir = os.path.join(self.config_yml['MODEL_SAVE_DIR'],
                                      dataset_train)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')

        log_dir = os.path.join(
            self.config_yml['RUNS_DIR'],
            '{}_{}'.format(dataset_train, self.__class__.__name__))
        self.writer = SummaryWriter(log_dir=os.path.join(
            log_dir, "run_{}".format(
                len(os.listdir(log_dir)) if os.path.exists(log_dir) else 0)))

        self.device = device
        self.dataset_train = HDF5TorchDataset(dataset_train, device=device)
        self.dataset_val = HDF5TorchDataset(dataset_val, device=device)

        self.lstm = nn.LSTM(self.config_yml['MEL_CHANNELS'],
                            self.config_yml['HIDDEN_SIZE'],
                            self.config_yml['NUM_LAYERS'],
                            batch_first=True)
        self.linear1 = nn.Linear(self.config_yml['HIDDEN_SIZE'],
                                 self.config_yml['EMBEDDING_SIZE'])
        # self.linear2 = nn.Linear(1, 1)
        # torch.nn.init.constant_(self.linear2.weight, 10.0)
        # torch.nn.init.constant_(self.linear2.bias, -5.0)
        self.W = torch.nn.Parameter(torch.Tensor([10.0]), requires_grad=True)
        self.b = torch.nn.Parameter(torch.Tensor([-5.0]), requires_grad=True)

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch
        self.loss_ = loss_

    def forward(self, frames):

        x, (_, _) = self.lstm(frames)  #lstm out,hidden,

        x = x[:, -1]  #last layer -> embeds

        x = self.linear1(x)

        # x = self.relu(x)

        x = x * torch.reciprocal(torch.norm(x, dim=1, keepdim=True))

        return x

    def loss_fn(self, loss_, embeds):

        lang_count = int(self.config_yml['BATCH_SIZE'] /
                         self.config_yml['UTTR_COUNT'])
        embeds3d = embeds.view(lang_count, self.config_yml['UTTR_COUNT'], -1)

        centroids = torch.mean(embeds3d, dim=1)

        centroids_neg = (torch.sum(embeds3d, dim=1, keepdim=True) -
                         embeds3d) / (self.config_yml['UTTR_COUNT'] - 1)
        cosim_neg = torch.cosine_similarity(embeds,
                                            centroids_neg.view_as(embeds),
                                            dim=1).view(lang_count, -1)
        centroids = centroids.repeat(
            lang_count * self.config_yml['UTTR_COUNT'], 1)

        embeds2de = embeds.unsqueeze(1).repeat_interleave(lang_count, 1).view(
            -1, self.config_yml['EMBEDDING_SIZE'])
        cosim = torch.cosine_similarity(embeds2de, centroids)
        cosim_matrix = cosim.view(lang_count, self.config_yml['UTTR_COUNT'],
                                  -1)
        neg_ix = list(range(lang_count))

        # print(cosim_matrix[neg_ix,:,neg_ix].shape)
        cosim_matrix[neg_ix, :, neg_ix] = cosim_neg

        # sim_matrix = self.linear2(cosim_matrix)
        sim_matrix = (self.W * cosim_matrix) + self.b

        sim_matrix = sim_matrix.view(self.config_yml['BATCH_SIZE'], -1)
        targets = torch.range(0, 3).repeat_interleave(
            self.config_yml['UTTR_COUNT']).long().to(device=self.device)
        ce_loss = loss_(sim_matrix, targets)
        return ce_loss

    def train_loop(self,
                   opt,
                   lr_scheduler,
                   loss_,
                   batch_size=1,
                   gaze_pred=None):

        train_iterator = torch.utils.data.DataLoader(self.dataset_train,
                                                     batch_size=batch_size,
                                                     shuffle=True)

        self.val_iterator = torch.utils.data.DataLoader(self.dataset_val,
                                                        batch_size=batch_size,
                                                        shuffle=True)

        if self.load_model:
            model_pickle = torch.load(self.model_save_string.format(
                self.epoch))
            self.load_state_dict(model_pickle['model_state_dict'])
            opt.load_state_dict(model_pickle['model_state_dict'])
            self.epoch = model_pickle['epoch']
            loss_val = model_pickle['loss']

        for epoch in range(self.epoch, 20000):
            for i, data in enumerate(train_iterator):
                data = data.view(
                    -1,
                    self.config_yml['SLIDING_WIN_SIZE'],
                    self.config_yml['MEL_CHANNELS'],
                )

                opt.zero_grad()

                embeds = self.forward(data)
                loss = self.loss_fn(loss_, embeds)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.parameters(), 3.0)
                opt.step()

                if epoch % 10 == 0:
                    self.writer.add_scalar('Loss', loss.data.item(), epoch)
                    self.writer.add_scalar('Val Loss', self.val_loss(), epoch)

                    torch.save(
                        {
                            'epoch': epoch,
                            'model_state_dict': self.state_dict(),
                            'optimizer_state_dict': opt.state_dict(),
                            'loss': loss,
                        }, self.model_save_string.format(epoch))

    def infer(self, uttrs):
        model_pickle = torch.load(self.model_save_string.format(self.epoch))
        self.load_state_dict(model_pickle['model_state_dict'])

        embeds = self.forward(uttrs).data.numpy()

        return embeds

    def accuracy(self):
        acc = 0
        ix = 0
        for i, data in enumerate(self.val_data):
            uttrs = data[0]
            embeds = self.forward(uttrs)
            #TODO
        return (acc / ix)

    def val_loss(self):
        with torch.no_grad():
            return self.loss_fn(
                self.loss_,
                self.forward(
                    next(iter(self.val_iterator)).view(
                        -1,
                        self.config_yml['SLIDING_WIN_SIZE'],
                        self.config_yml['MEL_CHANNELS'],
                    )))


if __name__ == "__main__":
    device = torch.device('cuda')
    loss_ = torch.nn.CrossEntropyLoss()
    encoder = ACCENT_ENCODER(dataset_train='gmu_4_train',
                             dataset_val='gmu_4_val',
                             device=device,
                             loss_=loss_).to(device=device)
    optimizer = torch.optim.SGD(encoder.parameters(), lr=1e-2)
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None
    encoder.train_loop(optimizer, lr_scheduler, loss_, batch_size=4)
    exit()
