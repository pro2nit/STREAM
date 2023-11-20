from torch.utils.data import Dataset
import glob, os
from torchvision import transforms
import numpy as np
from PIL import Image
import torch

class VidDataset(Dataset) :
    def __init__(self, vid_dir) :
        self.vids = sorted(glob.glob(os.path.join(vid_dir, '*.npy')))
        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self) :
        return len(self.vids)
    
    def __getitem__(self, idx) :
        vid_pth = self.vids[idx]
        vid = [self.preprocess(Image.fromarray(image.astype(np.uint8))) for image in np.load(vid_pth)]
        vid = torch.stack(vid)
        
        return vid