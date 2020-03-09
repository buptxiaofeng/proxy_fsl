import os.path as osp
import PIL
from PIL import Image

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/cub/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/cub/split')

# This is for the CUB dataset, which does not support the ResNet encoder now
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)
class CUB(Dataset):

    def __init__(self, setname, model_type):
        txt_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1
        self.wnids = []

        image_size = 84
        self.transform = transforms.Compose([
            transforms.Resize(92),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        data_dict = {}

        for l in lines:
            context = l.split(',')
            name = context[0] 
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                
            data.append(path)
            image = Image.open(path).convert("RGB")
            data_dict[path] = self.transform(image)
            image.close()
            label.append(lb)

        self.data_dict = data_dict
        self.data = data
        self.label = label
        self.num_class = np.unique(np.array(label)).shape[0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        #image = self.transform(Image.open(path).convert('RGB'))
        image = self.data_dict[path]
        return image, label            

