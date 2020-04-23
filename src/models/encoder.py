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

np.random.seed(42)


class ACCENT_ENCODER(nn.Module):
    def __init__(self,
                 input_shape=(84, 84),
                 load_model=False,
                 epoch=0,
                 num_actions=18,
                 data=['images', 'actions', 'fused_gazes'],
                 dataset='combined',
                 val_data='combined',
                 device=torch.device('cpu')):
        super(ACCENT_ENCODER, self).__init__()
        self.data = data
        self.dataset = dataset
        self.input_shape = input_shape
        self.num_actions = num_actions
        with open('src/config.yaml', 'r') as f:
            self.config_yml = safe_load(f.read())

        model_save_dir = os.path.join(self.config_yml['MODEL_SAVE_DIR'],
                                      dataset)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        self.model_save_string = os.path.join(
            model_save_dir, self.__class__.__name__ + '_Epoch_{}.pt')
        self.writer = SummaryWriter(log_dir=os.path.join(
            self.config_yml['RUNS_DIR'], dataset, self.__class__.__name__))
        self.device = device
        self.hdf5dataset = HDF5TorchDataset(dataset)
        self.val_data = val_data
        self.lstm = nn.LSTM(self.config_yml['MEL_CHANNELS'],
                            self.config_yml['HIDDEN_SIZE'],
                            self.config_yml['NUM_LAYERS'],
                            batch_first=True)
        self.linear1 = nn.Linear(self.config_yml['HIDDEN_SIZE'],
                                 self.config_yml['EMBEDDING_SIZE'])

        self.linear2 = nn.Linear(self.config_yml['BATCH_SIZE'],
                                 self.config_yml['BATCH_SIZE']) 

        self.dropout = nn.Dropout()
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax()
        self.load_model = load_model
        self.epoch = epoch

    def forward(self, frames):

        out, (x, _) = self.lstm(frames)  #lstm out,hidden,

        x = x[-1]  #last layer -> embeds

        x = self.linear1(x)

        x = self.relu(x)

        x = x * torch.reciprocal(torch.norm(x, dim=1, keepdim=True))

        return x

    def out_shape(self, layer, in_shape):
        h_in, w_in = in_shape
        h_out, w_out = floor((
            (h_in + 2 * layer.padding[0] - layer.dilation[0] *
             (layer.kernel_size[0] - 1) - 1) / layer.stride[0]) + 1), floor((
                 (w_in + 2 * layer.padding[1] - layer.dilation[1] *
                  (layer.kernel_size[1] - 1) - 1) / layer.stride[1]) + 1)
        return h_out, w_out

    def lin_in_shape(self):
        # TODO create as a wrapper
        # wrapper that gives num params

        # temp written down shape calcer
        pass

    def loss_fn(self, loss_, embdeds):
        centroids = {}
        emb_size = 1
        simililarity_matrix = []
    
        for i,emb_vec in enumerate(embdeds.reshape(4,embdeds.shape[0]//4,-1)):
            emb_size = emb_vec.shape[0]*emb_vec.shape[1]
            centroids[i] = torch.mean(emb_vec, dim=0) 

        cent_calc = lambda emb, cent : ((cent*emb_size)-emb)/(emb_size-1)
        for i,emb in enumerate(embdeds):
            simililarity_matrix.append([torch.cosine_similarity(emb.unsqueeze(0),cent_calc(emb.T, centroids[int(c)].unsqueeze(0))) for c in centroids.keys()])
        simililarity_matrix = torch.tensor(simililarity_matrix).T
        simililarity_matrix = self.linear2(simililarity_matrix).T
        target_matrix = torch.range(0,3)
        target_matrix = torch.repeat_interleave(target_matrix, embdeds.shape[0]//4).long()
        loss = loss_(simililarity_matrix, target_matrix) 
        print(loss)
        return loss

    def train_loop(self,
                   opt,
                   lr_scheduler,
                   loss_,
                   batch_size=1,
                   gaze_pred=None):

        train_data = torch.utils.data.DataLoader(self.hdf5dataset,
                                                 batch_size=batch_size)
        # self.val_data = torch.utils.data.DataLoader(
        #     self.hdf5dataset,
        #     batch_size=batch_size,
        #     sampler=ImbalancedDatasetSampler(self.hdf5dataset))

        if self.load_model:
            model_pickle = torch.load(self.model_save_string.format(
                self.epoch))
            self.load_state_dict(model_pickle['model_state_dict'])
            opt.load_state_dict(model_pickle['model_state_dict'])
            self.epoch = model_pickle['epoch']
            loss_val = model_pickle['loss']

        for epoch in range(self.epoch, 20000):
            for i, data in enumerate(train_data):
                data = data[0]
                opt.zero_grad()

                embeds = self.forward(data)

                loss = self.loss_fn(loss_, embeds)
                loss.backward()
                opt.step()

                if epoch % 10 == 0:
                    self.writer.add_histogram("embeds", embeds)
                    self.writer.add_scalar('Loss', loss.data.item(), epoch)
                    # self.writer.add_scalar('Acc',
                    #                        self.accuracy(x_g, gaze_pred),
                    #                        epoch)
                    # self.writer.add_scalar('Acc', self.accuracy(), epoch)

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
            acc += (acts == y).sum().item()
            ix += y.shape[0]
        return (acc / ix)


if __name__ == "__main__":

    encoder = ACCENT_ENCODER(dataset='gmu_4')
    print(encoder)
    optimizer = torch.optim.Adadelta(encoder.parameters(), lr=1.0, rho=0.95)
    loss_ = torch.nn.CrossEntropyLoss()
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lr_lambda=lambda x: x*0.95)
    lr_scheduler = None
    encoder.train_loop(optimizer, lr_scheduler, loss_, batch_size=1)
    exit()
