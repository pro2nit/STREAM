import torch
from dataset import VidDataset

from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np

from prdc import compute_prdc

class STREAM :
    def __init__(self, num_frame, model=None, num_embed=None) :
        # number of frame per video
        self.num_frame = num_frame
        
        if model == 'swav' :
            # code from : https://github.com/facebookresearch/swav
            self.embedder = torch.hub.load('facebookresearch/swav:main', 'resnet50')
            self.num_embed = 2048
        elif model == 'dinov2' :
            # code from : https://github.com/facebookresearch/dinov2
            self.embedder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.num_embed = 384
        else :
            self.num_embed = num_embed
        
        # calculate coefficient for power's law fitting
        x = torch.arange(1, num_frame//2+1, 1.0).view(1, num_frame//2)
        self.x = x.repeat(self.num_embed, 1, 1).view(-1, num_frame//2)

        x = torch.concat([torch.log(x), torch.ones_like(x)], dim=0)        
        D = torch.mm(x.T, torch.mm(x, x.T).inverse())
        self.D = D.repeat(self.num_embed, 1, 1)
    
    def calculate_coefficient(self, embeds) : # (f, n)
        device = embeds.device
        
        signal = torch.fft.fftn(input=embeds, dim=0)[:1+self.num_frame//2]
        amp = 2 * abs(signal) / (signal.shape[0])
        M = amp[0]
        amp = amp[1:]
        amp = amp + 1e-6 * torch.ones_like(amp).to(device)
        
        y = torch.log(amp).transpose(0, 1).view(-1, 1, amp.shape[0]) # (n, 1, f/2)
        
        # power's law fitting
        C = torch.bmm(y, self.D.to(device)).view(-1, 2) # (n, 2)
        
        return C, M 
        
    def _calculate_skewness(self, embeds) : 
        # calculate skewness of single feature
        device = embeds.device
        
        x = self.x.to(device)
        C, M = self.calculate_coefficient(embeds)
        # coefficient of power's law
        B, A = C[:, 0], torch.exp(C[:, 1])
        
        y_fit = A * torch.float_power(x.T, B)
        A_norm = A / y_fit.sum(dim=0)
        
        cm_3 = torch.sum(torch.float_power(x.T, B+3), dim=0)
        cm_2 = torch.sum(torch.float_power(x.T, B+2), dim=0)
        
        return cm_3 / (torch.sqrt(A_norm) * torch.sqrt(cm_2)), M
    
    def calculate_skewness(self, vid_dir, device='cpu', batch_size=16, num_workers=4) :
        '''
        <vid_dir>
        ├── vid_00000.npy
        ├── vid_00001.npy
        ├── ...
        ├── vid_02046.npy
        └── vid_02047.npy
        
        vid_%05d.npy
        * dtype : np.uint8 (0 ~ 255)
        * shape : (f, h, w, c) 
        '''
        
        embedder = self.embedder
        embedder.eval()
        embedder = embedder.to(device)
        embedder.requires_grad_ = False
        
        print('Calculating Skewness from %s ...' % vid_dir)
        dataset = VidDataset(vid_dir)
        loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        skewness, mean_signal = list(), list()
        
        for batch in tqdm(loader) :
            b, f, c, h, w = batch.shape[:]
            imgs = batch.view(-1, c, h, w).to(device)
            with torch.no_grad() :
                embeds = embedder(imgs).view(b, f, -1)
                for embed in embeds :
                    sk, ms = self._calculate_skewness(embed)
                    
                    skewness.append(sk.cpu())
                    mean_signal.append(ms.cpu())
    
        skewness = torch.stack(skewness)
        mean_signal = torch.stack(mean_signal)
        
        return skewness, mean_signal
    
    def corr(self, y_pred, y_true):
        # calculate correlation between prediction and ground truth
        x = torch.Tensor(y_pred)
        y = torch.Tensor(y_true)
        
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        
        cov = torch.sum(vx * vy)
        
        corr = cov / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-12)
        return corr ** 2

    def stream_T(self, y_pred, y_true):
        x = torch.Tensor(y_pred)
        y = torch.Tensor(y_true)

        dim = x.shape[1]
        size = x.shape[0]
        cor_sq = []

        for iloop in range(dim):
            REAL = y[:, iloop]
            FAKE = x[:, iloop]
            MIN = int(torch.min(torch.min(REAL), torch.min(FAKE)).numpy()) - 1
            MAX = int(torch.max(torch.max(REAL), torch.max(FAKE)).numpy()) + 1
            
            REAL = torch.histogram(REAL, bins = 50, range = (MIN, MAX), density = False)[0]/size
            FAKE = torch.histogram(FAKE, bins = 50, range = (MIN, MAX), density = False)[0]/size

            cor_sq.append(self.corr(FAKE, REAL).numpy())
        return np.mean(cor_sq)

    def stream_S(self, y_pred, y_true) :
        x = torch.Tensor(y_pred)
        y = torch.Tensor(y_true)
        
        # code from : https://github.com/LAIT-CVLab/TopPR
        score = compute_prdc(x, y, 5)
        
        return dict(stream_F = score.get('precision').item(), stream_D = score.get('recall').item())