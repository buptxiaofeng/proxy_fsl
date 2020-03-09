import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
IMAGE_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/images')
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')

class MiniImageNet(Dataset):
    """ Usage: 
    """
    def __init__(self, setname, model_type):
        assert model_type == "ResNet" or model_type == "ConvNet4" or model_type == "ConvNet6"

        # Transformation
        if model_type == 'ConvNet4' or model_type == "ConvNet6":
            image_size = 84
            if setname == 'train':
                self.transform = transforms.Compose([
                    #transforms.RandomResizedCrop(image_size),
                    transforms.Resize(92),
                    transforms.CenterCrop(image_size),
                    #transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #Lighting(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec']),
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])),
                    ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(92),
                    #transforms.RandomResizedCrop(image_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))
            ])
        elif model_type == 'ResNet':
            #image_size = 80
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                #transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]), 
                                     np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

        csv_path = osp.join(SPLIT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        data_dict = {}
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
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
        self.num_class = len(set(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        #image = Image.open(path).convert("RGB")
        #image = self.transform(image)
        image = self.data_dict[path]
        return image, label

