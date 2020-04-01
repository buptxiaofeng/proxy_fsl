import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json
import numpy
from feat.dataloader.mini_imagenet import MiniImageNet
from feat.dataloader.cub import CUB
from feat.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from feat.models.relation import Relation
from torch.optim.lr_scheduler import StepLR
from evaluation import evaluation
from tqdm import tqdm
import math
from torch.nn import init
import wandb
import os

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
          lr +=[ param_group['lr'] ]
    return lr

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

def train():
    json_file = open("parameters.json")
    parameters = json.load(json_file)
    json_file.close()
    os.environ["CUDA_VISIBLE_DEVICES"] = parameters["gpu_id"]

    train_set = None
    val_set = None
    test_set = None
    image_size = 84
    if "ResNet" in parameters["model_type"]:
        image_size = 224
    if parameters["dataset"] == "CUB":
        train_set = CUB(setname = "train", image_size = image_size, if_augmentation = json.loads(parameters["if_augmentation"].lower()))
        val_set = CUB(setname = 'val', image_size = image_size)
        test_set = CUB(setname = 'test', image_size = image_size)
    elif parameters["dataset"] == "MiniImageNet":
        train_set = MiniImageNet(setname = "train", image_size = image_size, if_agumentation = json.loads(parameters["if_augmentation"].lower()))
        val_set = MiniImageNet(setname = 'val', image_size = image_size)
        test_set = MiniImageNet(setname = 'test', image_size = image_size)
    else:
        raise("dataset parameter value error!")

    train_sampler = CategoriesSampler(train_set.label, n_batch = parameters["num_train"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    train_loader = DataLoader(dataset=train_set, batch_sampler = train_sampler, num_workers= 8, pin_memory = True)
    val_sampler = CategoriesSampler(val_set.label, n_batch = parameters["num_val"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    val_loader = DataLoader(dataset=val_set, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    test_sampler = CategoriesSampler(test_set.label, n_batch = parameters["num_test"], n_cls = parameters["num_way"], n_per = parameters["num_shot"] + parameters["num_query"])
    test_loader = DataLoader(dataset=test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True)

    relation = Relation(model_type = parameters["model_type"], num_shot = parameters["num_shot"], num_way = parameters["num_way"], num_query = parameters["num_query"], proxy_type = parameters["proxy_type"], classifier = parameters["classifier"]).cuda()
    #relation.apply(init_layer)

    optimizer = torch.optim.SGD(relation.parameters(), lr = parameters["sgd_lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = 50, factor = 0.5, min_lr = 0.0001)
    if parameters["dataset"] == "CUB":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience = 20, factor = 0.2, min_lr = 0.0001)

    ce = nn.CrossEntropyLoss().cuda()
    cudnn.benchmark = True

    label = torch.arange(parameters["num_way"]).repeat(parameters["num_query"])
    #one hot encode
    one_hot_label = torch.zeros(len(label), label.max()+1).scatter_(1, label.unsqueeze(1), 1.).float().cuda()
    label = label.cuda()
    max_acc = 0
    max_test_acc = 0
    train_accuracy = 0
    for epoch in range(parameters["num_epochs"]):
        total_rewards = 0
        total_loss = 0
        for i, batch in enumerate(train_loader, 1):
            relation.train()
            data, _ = [_.cuda() for _ in batch]
            p = parameters["num_shot"] * parameters["num_way"]
            support, query = data[:p], data[p:]
            relation_score = relation(support, query)

            loss = ce(-1 * relation_score, label) 
            total_loss += loss.item()
            _, predict_label = torch.min(relation_score, 1)
            rewards = [1 if predict_label[j]==label[j] else 0 for j in range(label.shape[0])]
            total_rewards += numpy.sum(rewards)
            
            relation.zero_grad()
            loss.backward()
            optimizer.step()

            episode = epoch * parameters["num_train"] + i + 1
            if episode % 100 == 0:
                print("episode:", epoch * parameters["num_train"] + i+1,"ce loss", total_loss / float(i + 1))
                train_accuracy = numpy.sum(total_rewards)/1.0/parameters["num_query"] / parameters["num_way"] / parameters["num_train"]
                print('Train Accuracy of the model on the train :{:.2f} %'.format(100 * train_accuracy))
            threshold = 50000
            if parameters["dataset"] == "CUB":
                threshold = 10000
            if (episode % 100 == 0 and episode > threshold) or episode % 1000 == 0:
                acc, _ = evaluation(parameters, relation, val_loader, mode="val")
                if acc > max_acc:
                    max_acc = acc
                    test_acc, _, = evaluation(parameters, relation, test_loader, mode="test")
                    max_test_acc = test_acc
                print("episode:", epoch * parameters["num_train"] + i+1,"max val acc:", max_acc, " max test acc:", max_test_acc)

        scheduler.step(max_acc)
        print("sgd learning rate:", get_learning_rate(optimizer))

if __name__ == "__main__":
    train()
